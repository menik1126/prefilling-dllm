import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xxhash
from transformers import AutoTokenizer

from d2f_vllm.config import Config
from d2f_vllm.engine.model_runner import AutoModelRunner
from d2f_vllm.sampling_params import SamplingParams
from d2f_vllm.utils.context import reset_context_diffusion_lm, set_context_diffusion_lm


@dataclass
class FastDLLMEngineOutput:
    text: str
    token_ids: List[int]
    n_diff_steps: int


@dataclass
class _StaticMaskConfig:
    diffusion_block_size: int


@dataclass
class _StaticMaskSeq:
    current_block_mask: torch.Tensor
    diffusion_block_size: int

    @property
    def config(self) -> _StaticMaskConfig:
        return _StaticMaskConfig(diffusion_block_size=self.diffusion_block_size)


@dataclass
class _CachedPrefix:
    prompt_hash: int
    page_ids: List[int]
    prompt_len: int
    last_context_logit: torch.Tensor
    ref_count: int = 1


class _PrefixPageAllocator:
    """Manages page allocation with hash-based prefix sharing for FastDLLMDreamEngine."""

    def __init__(self, num_pages: int, page_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.free_pages: deque = deque(range(num_pages))
        self.page_ref_count: List[int] = [0] * num_pages
        self.hash_to_prefix: Dict[int, _CachedPrefix] = {}

    @staticmethod
    def compute_prompt_hash(prompt_ids: List[int]) -> int:
        h = xxhash.xxh64()
        h.update(np.array(prompt_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    @property
    def num_free_pages(self) -> int:
        return len(self.free_pages)

    def allocate_pages(self, n: int) -> List[int]:
        if n > len(self.free_pages):
            raise RuntimeError(f"Cannot allocate {n} pages, only {len(self.free_pages)} free")
        pages = [self.free_pages.popleft() for _ in range(n)]
        for p in pages:
            self.page_ref_count[p] = 1
        return pages

    def ref_pages(self, pages: List[int]) -> None:
        for p in pages:
            self.page_ref_count[p] += 1

    def release_pages(self, pages: List[int]) -> None:
        for p in pages:
            self.page_ref_count[p] -= 1
            if self.page_ref_count[p] == 0:
                self.free_pages.append(p)

    def lookup_prefix(self, prompt_ids: List[int]) -> Optional[_CachedPrefix]:
        h = self.compute_prompt_hash(prompt_ids)
        return self.hash_to_prefix.get(h)

    def register_prefix(
        self, prompt_ids: List[int], page_ids: List[int], prompt_len: int, last_context_logit: torch.Tensor
    ) -> _CachedPrefix:
        h = self.compute_prompt_hash(prompt_ids)
        entry = _CachedPrefix(
            prompt_hash=h,
            page_ids=list(page_ids),
            prompt_len=prompt_len,
            last_context_logit=last_context_logit.detach(),
            ref_count=0,
        )
        self.hash_to_prefix[h] = entry
        return entry

    def release_prefix(self, prompt_ids: List[int]) -> None:
        h = self.compute_prompt_hash(prompt_ids)
        entry = self.hash_to_prefix.get(h)
        if entry is None:
            return
        entry.ref_count -= 1

    def evict_one(self) -> bool:
        """Evict a cached prefix that has no active users. Returns True if evicted."""
        for h, entry in list(self.hash_to_prefix.items()):
            if entry.ref_count <= 0:
                self.release_pages(entry.page_ids)
                del self.hash_to_prefix[h]
                return True
        return False


class FastDLLMDreamEngine:
    """Offline Fast-DLLM decode path on top of the d2f_vllm Dream runner.

    This first implementation targets the currently strongest ParallelComp
    setting where generation uses one Fast-DLLM block:
    ``[prompt][MASK x block_length]`` with full prompt+MASK prefill, then
    replace-and-denoise the generation slots while keeping prompt KV fixed.
    """

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 8192,
        block_length: int = 32,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.60,
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: int = 1,
        mask_token_id: int = 151666,
        threshold: float = 0.9,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        enforce_eager: bool = True,
        kv_cache_layout: str = "unified",
        master_port: int = 2333,
        shm_name: str = "d2f_vllm_fastdllm",
    ) -> None:
        self.block_length = int(block_length)
        self.mask_token_id = int(mask_token_id)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.top_k = top_k

        cfg = Config(
            model=model,
            model_name="dream",
            model_type="diffusion_lm",
            mask_token_id=self.mask_token_id,
            diffusion_block_size=self.block_length,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens or max_model_len,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            kv_cache_layout=kv_cache_layout,
            master_port=master_port,
            shm_name=shm_name,
        )
        if cfg.kv_cache_layout not in ("unified", "distinct"):
            raise ValueError(f"FastDLLMDreamEngine supports kv_cache_layout='unified' or 'distinct', got '{cfg.kv_cache_layout}'.")
        self.config = cfg
        self.runner = AutoModelRunner.from_config(cfg, 0, [])
        self.model = self.runner.model
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        self.page_size = self.runner.block_size
        self._prefix_cache = _PrefixPageAllocator(
            num_pages=cfg.num_kvcache_blocks, page_size=self.page_size
        )

    def close(self) -> None:
        if getattr(self, "runner", None) is not None:
            self.runner.exit()

    def _ids_tensor(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor(list(ids), dtype=torch.long, device=torch.cuda.current_device())

    def _positions(self, length: int, start: int = 0) -> torch.Tensor:
        return torch.arange(start, start + length, device=torch.cuda.current_device(), dtype=torch.long)

    def _positions_tensor(self, positions: Sequence[int]) -> torch.Tensor:
        return torch.tensor(list(positions), dtype=torch.long, device=torch.cuda.current_device())

    @staticmethod
    def _full_mask(rows: int, cols: Optional[int] = None) -> torch.Tensor:
        cols = rows if cols is None else cols
        return torch.ones((rows, cols), dtype=torch.bool, device=torch.cuda.current_device())

    def _set_full_prefill_context(self, seq_len: int, slot_mapping: torch.Tensor) -> None:
        seq = _StaticMaskSeq(self._full_mask(seq_len), self.block_length)
        seq_lens_ts = torch.tensor([seq_len], dtype=torch.int32, device=torch.cuda.current_device())
        set_context_diffusion_lm(
            True,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping.to(dtype=torch.int32),
            context_lens=torch.tensor([0], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=None,
            seqs=[seq],
            seq_lens=[seq_len],
            seq_lens_ts=seq_lens_ts,
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=True,
        )

    def _set_replace_context(self, context_len: int, block_len: int, slot_mapping: torch.Tensor) -> None:
        if block_len % self.block_length != 0:
            raise ValueError(
                f"d2f_vllm KV loader requires active length to be a multiple of "
                f"diffusion_block_size={self.block_length}; got {block_len}."
            )
        num_pages = math.ceil((context_len + block_len) / self.page_size)
        block_tables = torch.arange(num_pages, dtype=torch.int32, device=torch.cuda.current_device()).view(1, -1)
        if self.config.kv_cache_layout == "distinct":
            mask = self._full_mask(block_len, block_len)
        else:
            mask = self._full_mask(block_len, context_len + block_len)
        seq = _StaticMaskSeq(mask, self.block_length)
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor([0, block_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, context_len + block_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=block_len,
            max_seqlen_k=context_len + block_len,
            slot_mapping=slot_mapping.to(dtype=torch.int32),
            context_lens=torch.tensor([context_len], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=block_tables,
            seqs=[seq],
            seq_lens=[block_len],
            seq_lens_ts=torch.tensor([block_len], dtype=torch.int32, device=torch.cuda.current_device()),
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=True,
        )

    def _forward_prefill(self, ids: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        if len(ids) != len(positions):
            raise ValueError(f"ids/positions length mismatch: {len(ids)} vs {len(positions)}")
        input_ids = self._ids_tensor(ids)
        slot_mapping = torch.arange(len(ids), dtype=torch.int32, device=torch.cuda.current_device())
        self._set_full_prefill_context(len(ids), slot_mapping)
        try:
            hidden = self.model(input_ids, self._positions_tensor(positions))
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    def _page_ids_to_slot_mapping(self, page_ids: List[int], num_tokens: int) -> torch.Tensor:
        slots = []
        for token_idx in range(num_tokens):
            page_idx = token_idx // self.page_size
            offset = token_idx % self.page_size
            slots.append(page_ids[page_idx] * self.page_size + offset)
        return torch.tensor(slots, dtype=torch.int32, device=torch.cuda.current_device())

    def _split_slot_mapping(self, prompt_page_ids: List[int], block_page_ids: List[int],
                            prompt_len: int, block_len: int) -> torch.Tensor:
        """Build slot mapping for prefill: prompt tokens → prompt pages, block tokens → block pages."""
        slots = []
        for i in range(prompt_len):
            page_idx = i // self.page_size
            offset = i % self.page_size
            slots.append(prompt_page_ids[page_idx] * self.page_size + offset)
        for i in range(block_len):
            page_idx = i // self.page_size
            offset = i % self.page_size
            slots.append(block_page_ids[page_idx] * self.page_size + offset)
        return torch.tensor(slots, dtype=torch.int32, device=torch.cuda.current_device())

    def _forward_prefill_paged(self, ids: Sequence[int], positions: Sequence[int],
                               prompt_page_ids: List[int], block_page_ids: List[int],
                               prompt_len: int) -> torch.Tensor:
        if len(ids) != len(positions):
            raise ValueError(f"ids/positions length mismatch: {len(ids)} vs {len(positions)}")
        input_ids = self._ids_tensor(ids)
        block_len = len(ids) - prompt_len
        slot_mapping = self._split_slot_mapping(prompt_page_ids, block_page_ids, prompt_len, block_len)
        self._set_full_prefill_context(len(ids), slot_mapping)
        try:
            hidden = self.model(input_ids, self._positions_tensor(positions))
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    def _forward_replace_block_paged(
        self,
        block_ids: torch.Tensor,
        *,
        prompt_len: int,
        block_page_ids: List[int],
        all_page_ids: List[int],
        block_positions: Sequence[int],
    ) -> torch.Tensor:
        block_len = int(block_ids.numel())
        if block_len != len(block_positions):
            raise ValueError(f"block_ids/block_positions length mismatch: {block_len} vs {len(block_positions)}")
        slot_mapping = self._page_ids_to_slot_mapping(block_page_ids, block_len)
        block_tables = torch.tensor(all_page_ids, dtype=torch.int32, device=block_ids.device).view(1, -1)
        if self.config.kv_cache_layout == "distinct":
            mask = self._full_mask(block_len, block_len)
        else:
            mask = self._full_mask(block_len, prompt_len + block_len)
        seq = _StaticMaskSeq(mask, self.block_length)
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor([0, block_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, prompt_len + block_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=block_len,
            max_seqlen_k=prompt_len + block_len,
            slot_mapping=slot_mapping,
            context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=block_tables,
            seqs=[seq],
            seq_lens=[block_len],
            seq_lens_ts=torch.tensor([block_len], dtype=torch.int32, device=torch.cuda.current_device()),
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=True,
        )
        try:
            hidden = self.model(block_ids.reshape(-1), self._positions_tensor(block_positions))
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    def _forward_replace_block_for_init(
        self,
        *,
        prompt_len: int,
        prompt_page_ids: List[int],
        block_page_ids: List[int],
        suffix_positions: Sequence[int],
    ) -> torch.Tensor:
        """Run a single replace step with all-MASK block to get initial logits (cache hit path)."""
        block_len = self.block_length
        block_ids = torch.full((block_len,), self.mask_token_id, dtype=torch.long, device=torch.cuda.current_device())
        all_page_ids = prompt_page_ids + block_page_ids
        return self._forward_replace_block_paged(
            block_ids,
            prompt_len=prompt_len,
            block_page_ids=block_page_ids,
            all_page_ids=all_page_ids,
            block_positions=suffix_positions,
        )

    @staticmethod
    def _shift_logits(logits: torch.Tensor, last_logit: Optional[torch.Tensor] = None) -> torch.Tensor:
        shifted = torch.empty_like(logits)
        if logits.shape[0] > 1:
            shifted[1:, :] = logits[:-1, :]
        if last_logit is None:
            shifted[0, :] = logits[0, :]
        else:
            shifted[0, :] = last_logit.reshape(-1)
        return shifted

    def _sample_tokens(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        work_logits = logits.float()
        if self.temperature > 0:
            work_logits = work_logits / self.temperature
        if self.top_k is not None:
            top_k = min(int(self.top_k), work_logits.shape[-1])
            kth = torch.topk(work_logits, top_k, dim=-1).values[..., -1, None]
            work_logits = work_logits.masked_fill(work_logits < kth, torch.finfo(work_logits.dtype).min)
        if self.top_p is not None and self.top_p < 1:
            sorted_logits, sorted_indices = torch.sort(work_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > float(self.top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(work_logits, dtype=torch.bool)
            remove.scatter_(-1, sorted_indices, sorted_remove)
            work_logits = work_logits.masked_fill(remove, torch.finfo(work_logits.dtype).min)
        probs = F.softmax(work_logits, dim=-1)
        if self.temperature > 0:
            sampled = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1)
            return confidence, sampled
        confidence, sampled = probs.max(dim=-1)
        return confidence, sampled

    @torch.inference_mode()
    def generate_token_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        prompt_positions: Optional[Sequence[int]] = None,
        stop_token_ids: Optional[Iterable[int]] = None,
    ) -> FastDLLMEngineOutput:
        if max_new_tokens <= 0:
            return FastDLLMEngineOutput(text="", token_ids=[], n_diff_steps=0)
        if max_new_tokens > self.block_length:
            raise NotImplementedError(
                "The first FastDLLMDreamEngine version supports one generation block only. "
                f"Got max_new_tokens={max_new_tokens}, block_length={self.block_length}."
            )

        prompt_ids = list(int(x) for x in prompt_ids)
        if prompt_positions is None:
            prompt_positions = list(range(len(prompt_ids)))
        else:
            prompt_positions = [int(x) for x in prompt_positions]
        if len(prompt_ids) != len(prompt_positions):
            raise ValueError(
                f"prompt_ids/prompt_positions length mismatch: {len(prompt_ids)} vs {len(prompt_positions)}"
            )
        decode_len = self.block_length
        suffix_pos_start = (max(prompt_positions) + 1) if prompt_positions else 0
        suffix_positions = list(range(suffix_pos_start, suffix_pos_start + decode_len))
        prompt_len = len(prompt_ids)
        full_len = prompt_len + decode_len
        if full_len > self.config.max_model_len:
            raise ValueError(
                f"full_prompt_mask length {full_len} exceeds max_model_len={self.config.max_model_len}"
            )

        num_prompt_pages = math.ceil(prompt_len / self.page_size)
        num_block_pages = math.ceil(decode_len / self.page_size)

        cached = self._prefix_cache.lookup_prefix(prompt_ids)
        if cached is not None:
            prompt_page_ids = cached.page_ids
            cached.ref_count += 1
            last_context_logit = cached.last_context_logit
            while self._prefix_cache.num_free_pages < num_block_pages:
                if not self._prefix_cache.evict_one():
                    break
            block_page_ids = self._prefix_cache.allocate_pages(num_block_pages)
            init_logits = self._forward_replace_block_for_init(
                prompt_len=prompt_len,
                prompt_page_ids=prompt_page_ids,
                block_page_ids=block_page_ids,
                suffix_positions=suffix_positions,
            )
            shifted_init = self._shift_logits(init_logits, last_context_logit)
            _, first_token = self._sample_tokens(shifted_init[:1, :])
        else:
            total_pages_needed = num_prompt_pages + num_block_pages
            while self._prefix_cache.num_free_pages < total_pages_needed:
                if not self._prefix_cache.evict_one():
                    break
            prompt_page_ids = self._prefix_cache.allocate_pages(num_prompt_pages)
            block_page_ids = self._prefix_cache.allocate_pages(num_block_pages)

            full_ids = prompt_ids + [self.mask_token_id] * decode_len
            full_positions = list(prompt_positions) + suffix_positions
            prefill_logits = self._forward_prefill_paged(
                full_ids, full_positions, prompt_page_ids, block_page_ids, prompt_len
            )
            shifted_prefill = self._shift_logits(prefill_logits)
            first_logits = shifted_prefill[prompt_len:prompt_len + 1, :]
            _, first_token = self._sample_tokens(first_logits)
            last_context_logit = prefill_logits[prompt_len - 1, :].detach() if prompt_len > 0 else None
            self._prefix_cache.register_prefix(prompt_ids, prompt_page_ids, prompt_len, last_context_logit)

        block_ids = torch.full((decode_len,), self.mask_token_id, dtype=torch.long, device=torch.cuda.current_device())
        block_ids[0] = first_token[0]
        all_page_ids_for_decode = prompt_page_ids + block_page_ids
        n_steps = 0
        while bool((block_ids == self.mask_token_id).any()):
            n_steps += 1
            mask_index = block_ids == self.mask_token_id
            logits = self._forward_replace_block_paged(
                block_ids,
                prompt_len=prompt_len,
                block_page_ids=block_page_ids,
                all_page_ids=all_page_ids_for_decode,
                block_positions=suffix_positions,
            )
            shifted_logits = self._shift_logits(logits, last_context_logit)
            confidence, sampled = self._sample_tokens(shifted_logits[mask_index])

            candidate = torch.full_like(block_ids, self.mask_token_id)
            candidate[mask_index] = sampled
            full_confidence = torch.full_like(block_ids, -torch.inf, dtype=confidence.dtype)
            full_confidence[mask_index] = confidence
            transfer_count = int(mask_index.sum().item())
            selected_confidence, select_index = torch.topk(full_confidence, transfer_count)
            transfer_index = torch.zeros_like(block_ids, dtype=torch.bool)
            transfer_index[select_index[0]] = True
            for idx in range(1, transfer_count):
                if selected_confidence[idx] >= self.threshold:
                    transfer_index[select_index[idx]] = True
            block_ids[transfer_index] = candidate[transfer_index]

        self._prefix_cache.release_pages(block_page_ids)
        if cached is not None:
            cached.ref_count -= 1

        generated = block_ids[:max_new_tokens].tolist()
        if stop_token_ids:
            stop_set = set(int(x) for x in stop_token_ids)
            for idx, token_id in enumerate(generated):
                if token_id in stop_set:
                    generated = generated[:idx]
                    break
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and eos in text:
            text = text.split(eos)[0]
        return FastDLLMEngineOutput(text=text, token_ids=generated, n_diff_steps=n_steps)

    def generate(
        self,
        prompts: Sequence[str | Sequence[int]],
        sampling_params: SamplingParams,
    ) -> List[FastDLLMEngineOutput]:
        outputs: List[FastDLLMEngineOutput] = []
        stop_token_ids = None
        if sampling_params.stop_token_ids:
            stop_token_ids = [item for group in sampling_params.stop_token_ids for item in group]
        for prompt in prompts:
            if isinstance(prompt, str):
                prompt_ids = self.tokenizer.encode(prompt)
            else:
                prompt_ids = list(prompt)
            outputs.append(
            self.generate_token_ids(
                prompt_ids,
                max_new_tokens=int(sampling_params.max_tokens),
                prompt_positions=None,
                stop_token_ids=stop_token_ids,
            )
            )
        return outputs
