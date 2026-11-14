"""Model backends with one small interface, so the judge and reply generator don't care where a model runs.

  generate(convs, prefix, ...)              -> sampled continuations
  next_token_logprobs(convs, prefix, cands) -> log-prob of each candidate as the next token

`convs` are chat conversations ([{"role": ..., "content": ...}]). `prefix` is text we pre-fill at the
start of the assistant turn (e.g. "SCAFFOLDING: "), which forces the output format and lets us read
the judge's probability over score labels in a single forward pass.

Backends:
  VLLMBackend   GPU, batched, the default for everything heavy (Kaggle 2x T4).
  HFBackend     transformers on CPU/GPU; exact logits for any candidate. 4-bit reward judge, vLLM fallback, tests.
  OpenAIBackend any OpenAI-compatible API; generate-only (no pre-fill), used for the API sanity judge.
  FakeBackend   deterministic stand-in for unit tests.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

Conv = list[dict[str, str]]


def logsumexp(xs: list[float]) -> float:
    if not xs:
        return float("-inf")
    m = max(xs)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Backend:
    name: str = "base"
    supports_prefill: bool = True

    def generate(self, convs: list[Conv], prefix: str = "", max_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, n: int = 1, seed: int | None = None) -> list[list[str]]:
        raise NotImplementedError

    def next_token_logprobs(self, convs: list[Conv], prefix: str, candidates: list[str]) -> list[dict[str, float]]:
        raise NotImplementedError


class _ChatTemplateMixin:
    tokenizer: Any

    def render(self, conv: Conv, prefix: str) -> str:
        """Chat template + open assistant turn + pre-filled prefix."""
        prompt = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        return prompt + prefix


class VLLMBackend(_ChatTemplateMixin, Backend):
    def __init__(self, model: str, dtype: str = "bfloat16", max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85, tensor_parallel_size: int = 1, seed: int = 0,
                 enable_lora: bool = False, **kwargs: Any):
        from vllm import LLM

        self.name = model
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len, seed=seed,
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size, enable_lora=enable_lora, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, convs, prefix="", max_tokens=256, temperature=0.0, top_p=1.0, n=1, seed=None):
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
        outs = self.llm.generate([self.render(c, prefix) for c in convs], sp, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outs]

    def next_token_logprobs(self, convs, prefix, candidates):
        """Read the top-20 next-token log-probs and pick out the candidate labels.

        Labels are single digits, which Qwen tokenizes as one token each. If a candidate isn't in the
        top 20 its log-prob is -inf; the judge records how much probability mass landed on valid labels
        so a broken prompt format shows up immediately instead of silently.
        """
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        outs = self.llm.generate([self.render(c, prefix) for c in convs], sp, use_tqdm=False)
        results = []
        for out in outs:
            top = out.outputs[0].logprobs[0]  # {token_id: Logprob(logprob, rank, decoded_token)}
            buckets: dict[str, list[float]] = {c: [] for c in candidates}
            for lp in top.values():
                tok = (lp.decoded_token or "").strip()
                if tok in buckets:
                    buckets[tok].append(lp.logprob)
            results.append({c: logsumexp(v) for c, v in buckets.items()})
        return results


class HFBackend(_ChatTemplateMixin, Backend):
    """transformers backend. Exact logits for any candidate token, batched generation.

    Also the fallback when vLLM won't run on a GPU (set TUTORBIAS_BACKEND=hf; see load_backend), and
    the backend of the GRPO reward judge, which needs 4-bit weights to share a 16 GB T4.
      device_map="auto"  splits a model across every visible GPU (a 7B fp16 judge on Kaggle's 2x T4)
      load_in_4bit=True  NF4 weights via bitsandbytes (~5.5 GB for 7B), fp16 compute
    """

    def __init__(self, model: str, device: str | None = None, dtype: str = "auto", batch_size: int = 4,
                 device_map: str | None = None, load_in_4bit: bool = False, **_: Any):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            compute = torch.float16 if torch_dtype in ("auto", torch.float32) else torch_dtype
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute,
                bnb_4bit_use_double_quant=True)
            device_map = device_map or "auto"
            self.name = f"{model} (nf4)"  # recorded in every score row, so 4-bit scores are never mixed up
        if device_map and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(model, device_map=device_map, **kwargs).eval()
            self.device = next(self.model.parameters()).device
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs).to(self.device).eval()
        self.batch_size = batch_size

    def _cand_ids(self, cand: str) -> list[int]:
        ids = set()
        for form in (cand, " " + cand):
            toks = self.tokenizer.encode(form, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise ValueError(f"Candidate {cand!r} is not a single token for {self.name}")
        return sorted(ids)

    def next_token_logprobs(self, convs, prefix, candidates):
        cand_ids = {c: self._cand_ids(c) for c in candidates}
        prompts = [self.render(c, prefix) for c in convs]
        results = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tokenizer(prompts[i : i + self.batch_size], return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.device)
            with self.torch.no_grad():
                # logits_to_keep=1 would save memory but isn't in every transformers version; slice instead.
                logits = self.model(**enc).logits[:, -1, :].float()
            logp = self.torch.log_softmax(logits, dim=-1)
            for row in logp:
                results.append({c: logsumexp([row[j].item() for j in ids]) for c, ids in cand_ids.items()})
        return results

    def generate(self, convs, prefix="", max_tokens=256, temperature=0.0, top_p=1.0, n=1, seed=None):
        if seed is not None:
            self.torch.manual_seed(seed)
        prompts = [self.render(c, prefix) for c in convs]
        results = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tokenizer(prompts[i : i + self.batch_size], return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                    temperature=temperature or None, top_p=top_p if temperature > 0 else None,
                    num_return_sequences=n, pad_token_id=self.tokenizer.pad_token_id,
                )
            new = self.tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            results += [new[j * n : (j + 1) * n] for j in range(len(new) // n)]
        return results


class OpenAIBackend(Backend):
    """OpenAI-compatible chat API. No assistant pre-fill, so the judge parses free text instead."""

    supports_prefill = False

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", **_: Any):
        from openai import OpenAI

        self.name = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env))

    def generate(self, convs, prefix="", max_tokens=256, temperature=0.0, top_p=1.0, n=1, seed=None):
        if prefix:
            raise ValueError("OpenAIBackend does not support assistant pre-fill")
        out = []
        for conv in convs:
            r = self.client.chat.completions.create(model=self.name, messages=conv, max_tokens=max_tokens,
                                                    temperature=temperature, top_p=top_p, n=n, seed=seed)
            out.append([c.message.content or "" for c in r.choices])
        return out


class FakeBackend(Backend):
    """Test double. `logprob_fn(conv, prefix, candidates) -> {cand: logprob}`; `gen_fn(conv, prefix) -> str`."""

    def __init__(self, logprob_fn: Callable | None = None, gen_fn: Callable | None = None, name: str = "fake"):
        self.name = name
        self.logprob_fn = logprob_fn
        self.gen_fn = gen_fn or (lambda conv, prefix: "ok")
        self.calls: list[tuple[str, str]] = []

    def next_token_logprobs(self, convs, prefix, candidates):
        self.calls += [("lp", prefix)] * len(convs)
        return [self.logprob_fn(c, prefix, candidates) for c in convs]

    def generate(self, convs, prefix="", max_tokens=256, temperature=0.0, top_p=1.0, n=1, seed=None):
        self.calls += [("gen", prefix)] * len(convs)
        return [[self.gen_fn(c, prefix) for _ in range(n)] for c in convs]


# vLLM-only constructor arguments, dropped when a vLLM block is run with transformers instead.
_VLLM_ONLY = ("max_model_len", "gpu_memory_utilization", "tensor_parallel_size", "enable_lora", "seed",
              "enforce_eager", "swap_space")


def vllm_spec_to_hf(spec: dict[str, Any]) -> dict[str, Any]:
    """Translate a {"backend": "vllm", ...} block into an equivalent transformers block.

    Same model and dtype; the model is spread over all visible GPUs with device_map="auto" instead of
    vLLM tensor parallelism. Scores match vLLM up to numerical noise; generation is slower.
    """
    out = {k: v for k, v in spec.items() if k not in _VLLM_ONLY}
    out["backend"] = "hf"
    out.setdefault("device_map", "auto")
    out.setdefault("batch_size", 8)
    return out


def load_backend(spec: dict[str, Any]) -> Backend:
    """Build a backend from a config block like {"backend": "vllm", "model": "Qwen/...", ...}.

    TUTORBIAS_BACKEND=hf runs every vLLM block with transformers instead: the escape hatch if a vLLM
    release doesn't support the GPU (vLLM needs compute capability >= 7.0: fine on T4, not on P100).
    """
    spec = dict(spec)
    if spec.get("backend") == "vllm" and os.environ.get("TUTORBIAS_BACKEND", "").lower() == "hf":
        spec = vllm_spec_to_hf(spec)
    kind = spec.pop("backend")
    if kind == "vllm":
        return VLLMBackend(**spec)
    if kind == "hf":
        return HFBackend(**spec)
    if kind == "openai":
        return OpenAIBackend(**spec)
    raise ValueError(f"Unknown backend: {kind}")
