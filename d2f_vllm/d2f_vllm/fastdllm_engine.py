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
try:
    from d2f_vllm.layers.attention.ops.triton_eviction_scorer import (
        query_to_chunk_attention_scores as _triton_query_to_chunk_attention_scores,
    )
except Exception:
    _triton_query_to_chunk_attention_scores = None
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
        for layer_idx, layer in enumerate(self.model.model.layers):
            layer.self_attn.attn.layer_idx = layer_idx
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        self.page_size = self.runner.block_size
        self._prefix_cache = _PrefixPageAllocator(
            num_pages=cfg.num_kvcache_blocks, page_size=self.page_size
        )
        self._active_decode_delta_state = None

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

    def _set_full_prefill_context(
        self,
        seq_len: int,
        slot_mapping: torch.Tensor,
        *,
        need_kv_cache_store: bool = True,
    ) -> None:
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
            need_kv_cache_store=need_kv_cache_store,
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
            decode_delta_state=self._active_decode_delta_state,
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
            decode_delta_state=self._active_decode_delta_state,
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

    def _range_slot_mapping(self, page_ids: List[int], start_token: int, num_tokens: int) -> torch.Tensor:
        slots = []
        for local_idx in range(int(num_tokens)):
            token_idx = int(start_token) + local_idx
            page_idx = token_idx // self.page_size
            offset = token_idx % self.page_size
            slots.append(page_ids[page_idx] * self.page_size + offset)
        return torch.tensor(slots, dtype=torch.int32, device=torch.cuda.current_device())

    def _forward_prefill_into_pages(
        self,
        ids: Sequence[int],
        positions: Sequence[int],
        page_ids: List[int],
        *,
        start_token: int = 0,
        need_kv_cache_store: bool = True,
        attention_mask: str = "full",
        prefix_len: int = 0,
        chunk_len: int = 0,
        query_len: Optional[int] = None,
    ) -> torch.Tensor:
        if len(ids) != len(positions):
            raise ValueError(f"ids/positions length mismatch: {len(ids)} vs {len(positions)}")
        if not ids:
            return torch.empty(0, int(self.config.hf_config.vocab_size), device=torch.cuda.current_device())
        input_ids = self._ids_tensor(ids)
        slot_mapping = self._range_slot_mapping(page_ids, start_token, len(ids))
        if (attention_mask or "full").lower() in {"full", "none"}:
            self._set_full_prefill_context(len(ids), slot_mapping, need_kv_cache_store=need_kv_cache_store)
        else:
            if query_len is None:
                query_len = max(0, len(ids) - int(prefix_len) - int(chunk_len))
            mask = self._selection_prefill_mask(
                attention_mask,
                seq_len=len(ids),
                prefix_len=int(prefix_len),
                chunk_len=int(chunk_len),
                query_len=int(query_len),
            )
            self._set_prefill_context_from_mask(mask, slot_mapping, need_kv_cache_store=need_kv_cache_store)
        try:
            hidden = self.model(input_ids, self._positions_tensor(positions))
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    def _forward_append_tokens_paged(
        self,
        ids: Sequence[int],
        positions: Sequence[int],
        *,
        context_len: int,
        all_page_ids: List[int],
        start_token: int,
        valid_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if len(ids) != len(positions):
            raise ValueError(f"ids/positions length mismatch: {len(ids)} vs {len(positions)}")
        if not ids:
            return torch.empty(0, int(self.config.hf_config.vocab_size), device=torch.cuda.current_device())
        input_ids = self._ids_tensor(ids)
        active_len = len(ids)
        slot_mapping = self._range_slot_mapping(all_page_ids, start_token, active_len)
        block_tables = torch.tensor(all_page_ids, dtype=torch.int32, device=input_ids.device).view(1, -1)
        mask = self._full_mask(active_len, int(context_len) + active_len)
        if valid_tokens is not None and int(valid_tokens) < active_len:
            valid = max(0, int(valid_tokens))
            mask[:valid, int(context_len) + valid:] = False
        seq = _StaticMaskSeq(mask, self.block_length)
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor([0, active_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, int(context_len) + active_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=active_len,
            max_seqlen_k=int(context_len) + active_len,
            slot_mapping=slot_mapping,
            context_lens=torch.tensor([int(context_len)], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=block_tables,
            seqs=[seq],
            seq_lens=[active_len],
            seq_lens_ts=torch.tensor([active_len], dtype=torch.int32, device=torch.cuda.current_device()),
            kv_cache_layout=self.config.kv_cache_layout,
            need_kv_cache_store=True,
        )
        try:
            hidden = self.model(input_ids, self._positions_tensor(positions))
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    def _copy_chunk_local_kv_to_prompt_pages(
        self,
        *,
        src_page_ids: List[int],
        dst_page_ids: List[int],
        src_chunk_start: int,
        dst_start: int,
        keep_indices_per_layer_per_head: Sequence[torch.Tensor],
    ) -> int:
        if self.config.kv_cache_layout != "unified":
            raise ValueError("chunk-local KV copy currently supports kv_cache_layout='unified' only.")
        if not keep_indices_per_layer_per_head:
            return 0
        kv_cache = self.runner.kv_cache
        src_pages = torch.tensor(src_page_ids, dtype=torch.long, device=torch.cuda.current_device())
        dst_pages_all = torch.tensor(dst_page_ids, dtype=torch.long, device=torch.cuda.current_device())
        keep_count = int(keep_indices_per_layer_per_head[0].shape[1])
        if keep_count <= 0:
            return 0
        dst_token_idx = torch.arange(int(dst_start), int(dst_start) + keep_count, dtype=torch.long, device=torch.cuda.current_device())
        dst_pages = dst_pages_all.index_select(0, dst_token_idx // self.page_size)
        dst_offsets = dst_token_idx % self.page_size
        for layer_idx, keep in enumerate(keep_indices_per_layer_per_head):
            keep = keep.to(device=torch.cuda.current_device(), dtype=torch.long)
            layer_cache = kv_cache[:, layer_idx]
            if int(keep.shape[1]) != keep_count:
                raise ValueError(f"Layer {layer_idx} keep_count mismatch in chunk-local KV copy")
            for head_idx in range(int(keep.shape[0])):
                src_token_idx = keep[head_idx] + int(src_chunk_start)
                src_pages_for_head = src_pages.index_select(0, src_token_idx // self.page_size)
                src_offsets = src_token_idx % self.page_size
                gathered = layer_cache[:, src_pages_for_head, src_offsets, head_idx, :].clone()
                layer_cache[:, dst_pages, dst_offsets, head_idx, :] = gathered
        return keep_count

    def _normalize_per_head_keep_indices(
        self,
        keep_indices_per_layer_per_head: Sequence[Sequence[Sequence[int]]],
        *,
        full_prompt_len: int,
    ) -> Tuple[List[torch.Tensor], int]:
        if self.config.kv_cache_layout != "unified":
            raise ValueError(
                "FastDLLMDreamEngine per-head KV eviction currently supports "
                "kv_cache_layout='unified' only."
            )
        kv_cache = getattr(self.runner, "kv_cache", None)
        if kv_cache is None:
            raise RuntimeError("Unified KV cache is not allocated on the model runner.")
        num_layers = int(kv_cache.shape[1])
        num_heads = int(kv_cache.shape[4])
        if len(keep_indices_per_layer_per_head) != num_layers:
            raise ValueError(
                "per-head keep index layer count mismatch: "
                f"got {len(keep_indices_per_layer_per_head)}, expected {num_layers}"
            )

        normalized: List[torch.Tensor] = []
        active_len: Optional[int] = None
        for layer_idx, layer_keep in enumerate(keep_indices_per_layer_per_head):
            keep = torch.as_tensor(layer_keep, dtype=torch.long, device=torch.cuda.current_device())
            if keep.ndim != 2:
                raise ValueError(f"Layer {layer_idx} keep indices must be [num_heads, active_len], got {tuple(keep.shape)}")
            if int(keep.shape[0]) != num_heads:
                raise ValueError(
                    f"Layer {layer_idx} keep head count mismatch: got {keep.shape[0]}, expected {num_heads}"
                )
            if active_len is None:
                active_len = int(keep.shape[1])
            elif int(keep.shape[1]) != active_len:
                raise ValueError(
                    f"Layer {layer_idx} active length mismatch: got {keep.shape[1]}, expected {active_len}"
                )
            if keep.numel() > 0:
                if int(keep.min().item()) < 0 or int(keep.max().item()) >= full_prompt_len:
                    raise ValueError(
                        f"Layer {layer_idx} keep indices are outside full prompt length {full_prompt_len}."
                    )
            normalized.append(keep.contiguous())

        return normalized, int(active_len or 0)

    def _compact_prompt_cache_per_layer_per_head(
        self,
        prompt_page_ids: List[int],
        keep_indices_per_layer_per_head: Sequence[torch.Tensor],
    ) -> int:
        """Compact prompt KV pages in-place using per-layer/per-head token indices."""
        if self.config.kv_cache_layout != "unified":
            raise ValueError(
                "FastDLLMDreamEngine per-head KV compaction currently supports "
                "kv_cache_layout='unified' only."
            )
        kv_cache = self.runner.kv_cache
        page_ids = torch.tensor(prompt_page_ids, dtype=torch.long, device=torch.cuda.current_device())
        if not keep_indices_per_layer_per_head:
            return 0
        active_len = int(keep_indices_per_layer_per_head[0].shape[1])
        if active_len <= 0:
            return 0

        dst_token_idx = torch.arange(active_len, dtype=torch.long, device=torch.cuda.current_device())
        dst_pages = page_ids.index_select(0, dst_token_idx // self.page_size)
        dst_offsets = dst_token_idx % self.page_size

        for layer_idx, keep in enumerate(keep_indices_per_layer_per_head):
            layer_cache = kv_cache[:, layer_idx]
            for head_idx in range(int(keep.shape[0])):
                src_token_idx = keep[head_idx]
                src_pages = page_ids.index_select(0, src_token_idx // self.page_size)
                src_offsets = src_token_idx % self.page_size
                compacted = layer_cache[:, src_pages, src_offsets, head_idx, :].clone()
                layer_cache[:, dst_pages, dst_offsets, head_idx, :] = compacted
        return active_len

    def _snapshot_prompt_cache_per_layer(
        self,
        prompt_page_ids: List[int],
        prompt_len: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Copy full prompt KV before compaction for decode-time delta attention."""
        if self.config.kv_cache_layout != "unified":
            raise ValueError("decode delta reservoir currently supports kv_cache_layout='unified' only.")
        if prompt_len <= 0:
            return [], []
        kv_cache = self.runner.kv_cache
        page_ids = torch.tensor(prompt_page_ids, dtype=torch.long, device=torch.cuda.current_device())
        token_idx = torch.arange(int(prompt_len), dtype=torch.long, device=torch.cuda.current_device())
        pages = page_ids.index_select(0, token_idx // self.page_size)
        offsets = token_idx % self.page_size

        full_k: List[torch.Tensor] = []
        full_v: List[torch.Tensor] = []
        for layer_idx in range(int(kv_cache.shape[1])):
            layer_cache = kv_cache[:, layer_idx]
            full_k.append(layer_cache[0, pages, offsets, :, :].clone().contiguous())
            full_v.append(layer_cache[1, pages, offsets, :, :].clone().contiguous())
        return full_k, full_v

    @staticmethod
    def _select_attention_layer_indices(total_layers: int, layer_window: int, layer_mode: str) -> List[int]:
        if total_layers <= 0:
            return []
        if layer_mode == "all" or layer_window <= 0:
            return list(range(total_layers))
        window = min(max(1, int(layer_window)), total_layers)
        if layer_mode == "first":
            return list(range(window))
        if layer_mode == "last":
            return list(range(total_layers - window, total_layers))
        raise ValueError(f"Unsupported attention layer mode: {layer_mode}")

    @staticmethod
    def _window_query(query_ids: Sequence[int], window: int) -> List[int]:
        ids = list(int(x) for x in query_ids)
        if window and window > 0:
            return ids[-int(window):]
        return ids

    @staticmethod
    def _select_positions_from_token_scores(
        token_scores: torch.Tensor,
        *,
        capacity: int,
        chunk_len: int,
        keep_high: bool = True,
        force_keep_first: bool = True,
    ) -> torch.Tensor:
        keep_count = min(max(1, int(capacity)), int(chunk_len))
        if token_scores.numel() != chunk_len:
            head = keep_count // 2
            tail = keep_count - head
            keep = sorted(set(list(range(head)) + list(range(chunk_len - tail, chunk_len))))
            return torch.tensor(keep[:keep_count], device=token_scores.device, dtype=torch.long)
        if force_keep_first and chunk_len > 0:
            if keep_count == 1:
                return torch.zeros(1, device=token_scores.device, dtype=torch.long)
            candidate_indices = torch.arange(1, chunk_len, device=token_scores.device, dtype=torch.long)
            if candidate_indices.numel() <= keep_count - 1:
                return torch.arange(chunk_len, device=token_scores.device, dtype=torch.long)
            candidate_scores = token_scores.index_select(0, candidate_indices)
            selected = candidate_indices[torch.topk(candidate_scores, k=keep_count - 1, largest=keep_high).indices]
            return torch.sort(torch.cat([torch.zeros(1, device=token_scores.device, dtype=torch.long), selected], dim=0)).values
        return torch.topk(token_scores, k=keep_count, largest=keep_high).indices.sort().values


    def _selection_prefill_mask(
        self,
        mode: str,
        *,
        seq_len: int,
        prefix_len: int,
        chunk_len: int,
        query_len: int,
    ) -> torch.Tensor:
        mode = (mode or "full").lower()
        device = torch.cuda.current_device()
        if mode in {"full", "none"}:
            return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        if mode == "causal":
            return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        if mode == "query_to_chunk":
            mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
            query_start = int(prefix_len + chunk_len)
            if query_len > 0 and query_start < seq_len:
                chunk_rows = torch.arange(prefix_len, query_start, device=device)
                if chunk_rows.numel() > 0:
                    mask[chunk_rows[:, None], torch.arange(query_start, seq_len, device=device)] = False
            return mask
        raise ValueError(f"Unsupported selection attention mask: {mode}")

    def _set_prefill_context_from_mask(
        self,
        attention_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        need_kv_cache_store: bool = False,
    ) -> None:
        seq_len = int(attention_mask.shape[0])
        seq = _StaticMaskSeq(attention_mask, self.block_length)
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
            need_kv_cache_store=need_kv_cache_store,
        )

    def _forward_prefill_for_selection(
        self,
        ids: Sequence[int],
        *,
        attention_mask: str = "full",
        prefix_len: int = 0,
        chunk_len: int = 0,
        query_len: Optional[int] = None,
    ) -> torch.Tensor:
        ids = [int(x) for x in ids]
        if not ids:
            return torch.empty(0, int(self.config.hf_config.vocab_size), device=torch.cuda.current_device())
        seq_len = len(ids)
        if query_len is None:
            query_len = max(0, seq_len - int(prefix_len) - int(chunk_len))
        input_ids = self._ids_tensor(ids)
        positions = self._positions(seq_len, 0)
        slot_mapping = torch.arange(seq_len, dtype=torch.int32, device=torch.cuda.current_device())
        mask = self._selection_prefill_mask(
            attention_mask,
            seq_len=seq_len,
            prefix_len=int(prefix_len),
            chunk_len=int(chunk_len),
            query_len=int(query_len),
        )
        self._set_prefill_context_from_mask(mask, slot_mapping, need_kv_cache_store=False)
        try:
            hidden = self.model(input_ids, positions)
            return self.model.compute_logits(hidden)
        finally:
            reset_context_diffusion_lm()

    @torch.inference_mode()
    def generate_partial_draft_rounds_for_selection(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        partial_rounds: int,
    ) -> Tuple[List[int], List[bool]]:
        draft_len = int(max_new_tokens or 0)
        if draft_len <= 0:
            return [], []
        prompt_ids = [int(x) for x in prompt_ids]
        block_ids = [self.mask_token_id] * draft_len
        confirmed = [False] * draft_len
        prompt_len = len(prompt_ids)

        logits = self._forward_prefill_for_selection(
            prompt_ids + block_ids,
            attention_mask="full",
            prefix_len=prompt_len,
            chunk_len=0,
            query_len=draft_len,
        )
        shifted = self._shift_logits(logits)
        _, first_token = self._sample_tokens(shifted[prompt_len:prompt_len + 1, :])
        block_ids[0] = int(first_token[0].item())
        confirmed[0] = True

        for _ in range(max(0, int(partial_rounds or 0))):
            mask_positions = [idx for idx, token_id in enumerate(block_ids) if int(token_id) == self.mask_token_id]
            if not mask_positions:
                break
            logits = self._forward_prefill_for_selection(
                prompt_ids + block_ids,
                attention_mask="full",
                prefix_len=prompt_len,
                chunk_len=0,
                query_len=draft_len,
            )
            shifted = self._shift_logits(logits)
            block_logits = shifted[prompt_len:prompt_len + draft_len, :]
            mask_tensor = torch.tensor(mask_positions, dtype=torch.long, device=torch.cuda.current_device())
            confidence, sampled = self._sample_tokens(block_logits.index_select(0, mask_tensor))
            best = int(torch.argmax(confidence).item())
            selected_pos = int(mask_positions[best])
            block_ids[selected_pos] = int(sampled[best].item())
            confirmed[selected_pos] = True
        return block_ids, confirmed

    def _selection_query_ids_with_score_target_engine(
        self,
        prefix_ids: Sequence[int],
        scoring_query_ids: Sequence[int],
        *,
        score_mode: str,
        score_draft_tokens: int = 0,
        score_draft_partial_rounds: Optional[int] = None,
        score_draft_score_all_slots: bool = False,
    ) -> Tuple[List[int], Optional[int], Optional[List[bool]]]:
        selection_query_ids = [int(x) for x in scoring_query_ids]
        score_token_count: Optional[int] = None
        score_token_mask: Optional[List[bool]] = None
        draft_ids: List[int] = []
        if (score_mode or "").lower() == "draft_self_information" and int(score_draft_tokens or 0) > 0:
            if score_draft_partial_rounds is not None:
                draft_ids, draft_mask = self.generate_partial_draft_rounds_for_selection(
                    list(prefix_ids) + list(scoring_query_ids),
                    max_new_tokens=int(score_draft_tokens),
                    partial_rounds=int(score_draft_partial_rounds),
                )
                if score_draft_score_all_slots:
                    score_token_mask = [True] * len(scoring_query_ids) + [True] * len(draft_ids)
                else:
                    score_token_mask = [True] * len(scoring_query_ids) + list(draft_mask)
            else:
                draft = self.generate_token_ids(
                    list(prefix_ids) + list(scoring_query_ids),
                    max_new_tokens=int(score_draft_tokens),
                    stop_token_ids=None,
                )
                draft_ids = list(draft.token_ids)
            selection_query_ids = list(scoring_query_ids) + list(draft_ids)
        return selection_query_ids, score_token_count, score_token_mask

    @torch.inference_mode()
    def score_chunk_self_information_engine(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
        *,
        score_token_count: Optional[int] = None,
        score_token_mask: Optional[Sequence[bool]] = None,
        score_attention_mask: str = "causal",
    ) -> float:
        prefix_ids = [int(x) for x in prefix_ids]
        chunk_ids = [int(x) for x in chunk_ids]
        query_ids = [int(x) for x in query_ids]
        if not chunk_ids or not query_ids:
            return float("-inf")
        prefix_len = len(prefix_ids)
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        joint_ids = prefix_ids + chunk_ids + query_ids
        if score_token_mask is not None:
            if len(score_token_mask) != query_len:
                return float("-inf")
            local_indices = [idx for idx, keep in enumerate(score_token_mask) if keep]
            if not local_indices:
                return float("-inf")
            label_positions = [prefix_len + chunk_len + idx for idx in local_indices]
            label_ids = [query_ids[idx] for idx in local_indices]
        elif score_token_count is None:
            start = prefix_len + chunk_len
            end = prefix_len + chunk_len + query_len
            label_positions = list(range(start, end))
            label_ids = joint_ids[start:end]
        else:
            target_len = min(query_len, max(0, int(score_token_count)))
            if target_len <= 0:
                return float("-inf")
            start = prefix_len + chunk_len
            end = start + target_len
            label_positions = list(range(start, end))
            label_ids = joint_ids[start:end]
        if not label_positions or min(label_positions) <= 0:
            return float("-inf")
        logits = self._forward_prefill_for_selection(
            joint_ids,
            attention_mask=score_attention_mask,
            prefix_len=prefix_len,
            chunk_len=chunk_len,
            query_len=query_len,
        )
        positions = torch.tensor(label_positions, device=torch.cuda.current_device(), dtype=torch.long)
        query_logits = logits.index_select(0, positions - 1)
        labels = torch.tensor(label_ids, device=torch.cuda.current_device(), dtype=torch.long)
        log_probs = F.log_softmax(query_logits.float(), dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return float(-token_nll.mean().item())

    def select_chunks_by_engine(
        self,
        prefix_ids: Sequence[int],
        candidate_chunks: Sequence[Sequence[int]],
        scoring_query_ids: Sequence[int],
        *,
        topk_chunks: int,
        score_mode: str = "draft_self_information",
        score_draft_tokens: int = 4,
        score_draft_partial_rounds: Optional[int] = 1,
        score_draft_score_all_slots: bool = False,
        score_attention_mask: str = "causal",
        score_context_mode: str = "single_chunk",
        keep_first_chunk: bool = False,
    ) -> Tuple[List[int], Dict[int, float], List[int], Optional[List[bool]]]:
        if not candidate_chunks:
            return [], {}, list(scoring_query_ids), None
        selection_query_ids, score_token_count, score_token_mask = self._selection_query_ids_with_score_target_engine(
            prefix_ids,
            scoring_query_ids,
            score_mode=score_mode,
            score_draft_tokens=score_draft_tokens,
            score_draft_partial_rounds=score_draft_partial_rounds,
            score_draft_score_all_slots=score_draft_score_all_slots,
        )
        if (score_mode or "none").lower() == "none" or not selection_query_ids:
            selected = list(range(len(candidate_chunks)))
            if int(topk_chunks or 0) > 0:
                selected = selected[:int(topk_chunks)]
            return selected, {}, selection_query_ids, score_token_mask
        if score_context_mode != "single_chunk":
            raise ValueError(f"Engine chunk selection currently supports score_context_mode='single_chunk', got {score_context_mode!r}.")
        mode = (score_mode or "").lower()
        if mode not in {"self_information", "draft_self_information"}:
            raise ValueError(f"Engine chunk selection currently supports self_information/draft_self_information, got {score_mode!r}.")
        scores: Dict[int, float] = {}
        for idx, chunk_ids in enumerate(candidate_chunks):
            score = self.score_chunk_self_information_engine(
                prefix_ids,
                chunk_ids,
                selection_query_ids,
                score_token_count=score_token_count,
                score_token_mask=score_token_mask,
                score_attention_mask=score_attention_mask,
            )
            scores[idx] = score if math.isfinite(score) else float("-inf")
        forced = [0] if keep_first_chunk and candidate_chunks else []
        remaining = [idx for idx in range(len(candidate_chunks)) if idx not in forced]
        ranked = sorted(remaining, key=lambda idx: scores.get(idx, float("-inf")), reverse=True)
        topk = len(ranked) if int(topk_chunks or 0) <= 0 else min(int(topk_chunks), len(ranked))
        return sorted(set(forced + ranked[:topk])), scores, selection_query_ids, score_token_mask

    @staticmethod
    def _row_allowed_keys(
        *,
        row: int,
        key_len: int,
        prefix_len: int,
        chunk_len: int,
        query_len: int,
        attention_mask: str,
        device: torch.device,
    ) -> torch.Tensor:
        mode = (attention_mask or "full").lower()
        allowed = torch.ones(key_len, dtype=torch.bool, device=device)
        if mode in {"full", "none"}:
            return allowed
        if mode == "causal":
            allowed &= torch.arange(key_len, device=device) <= int(row)
            return allowed
        if mode == "query_to_chunk":
            if prefix_len <= row < prefix_len + chunk_len and query_len > 0:
                allowed[prefix_len + chunk_len:] = False
            return allowed
        raise ValueError(f"Unsupported token attention mask: {attention_mask}")

    def _engine_chunk_attention_head_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        prefix_len: int,
        chunk_len: int,
        query_len: int,
        direction: str,
        reduce_mode: str,
        attention_mask: str,
        backend: str = "torch",
    ) -> torch.Tensor:
        if chunk_len <= 0 or query_len <= 0:
            return torch.empty(0, device=q.device)
        num_query_heads = int(q.shape[1])
        num_kv_heads = int(k.shape[1])
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads})"
            )
        group_size = num_query_heads // num_kv_heads
        key_len = int(k.shape[0])
        chunk_start = int(prefix_len)
        query_start = int(prefix_len + chunk_len)
        direction = (direction or "query_to_chunk").lower()
        backend = (backend or "torch").lower()
        parts: List[torch.Tensor] = []

        if direction == "query_to_chunk" and backend in {"triton", "auto"}:
            if _triton_query_to_chunk_attention_scores is None:
                if backend == "triton":
                    raise RuntimeError("TOKEN_SCORE_BACKEND=triton requested, but Triton eviction scorer is unavailable.")
            else:
                try:
                    return _triton_query_to_chunk_attention_scores(
                        q,
                        k,
                        prefix_len=prefix_len,
                        chunk_len=chunk_len,
                        query_len=query_len,
                        scale=float(self.model.model.layers[0].self_attn.scaling),
                        reduce_mode=reduce_mode,
                        attention_mask=attention_mask,
                    )
                except Exception:
                    if backend == "triton":
                        raise

        def probs_for_rows(rows: torch.Tensor) -> torch.Tensor:
            per_query_head_probs = []
            mask_mode = (attention_mask or "full").lower()
            for kv_head in range(num_kv_heads):
                q_group = q.index_select(0, rows)[:, kv_head * group_size:(kv_head + 1) * group_size, :]
                k_head = k[:, kv_head, :]
                logits = torch.einsum("rgd,sd->rgs", q_group.float(), k_head.float()) * float(self.model.model.layers[0].self_attn.scaling)
                if mask_mode not in {"full", "none"}:
                    for local_idx, row in enumerate(rows.tolist()):
                        allowed = self._row_allowed_keys(
                            row=int(row),
                            key_len=key_len,
                            prefix_len=prefix_len,
                            chunk_len=chunk_len,
                            query_len=query_len,
                            attention_mask=attention_mask,
                            device=q.device,
                        )
                        logits[local_idx, :, ~allowed] = torch.finfo(logits.dtype).min
                probs = torch.softmax(logits, dim=-1)
                per_query_head_probs.append(probs.permute(1, 0, 2))
            return torch.cat(per_query_head_probs, dim=0)

        if direction in {"query_to_chunk", "bidirectional"}:
            rows = torch.arange(query_start, query_start + query_len, device=q.device, dtype=torch.long)
            query_probs = probs_for_rows(rows)
            query_to_chunk = query_probs[:, :, chunk_start:chunk_start + chunk_len]
            if query_to_chunk.numel() > 0:
                if reduce_mode == "mean":
                    parts.append(query_to_chunk.mean(dim=1))
                else:
                    parts.append(query_to_chunk.sum(dim=1))
        if direction in {"chunk_to_query", "bidirectional"}:
            rows = torch.arange(chunk_start, chunk_start + chunk_len, device=q.device, dtype=torch.long)
            chunk_probs = probs_for_rows(rows)
            chunk_to_query = chunk_probs[:, :, query_start:query_start + query_len]
            if chunk_to_query.numel() > 0:
                if reduce_mode == "mean":
                    parts.append(chunk_to_query.mean(dim=2))
                else:
                    parts.append(chunk_to_query.sum(dim=2))
        if not parts:
            return torch.empty(0, device=q.device)
        return torch.stack(parts, dim=0).sum(dim=0)

    @torch.inference_mode()
    def compute_prompt_keep_indices_per_layer_per_head(
        self,
        *,
        full_prompt_len: int,
        prefix_ids: Sequence[int],
        chunk_spans: Sequence[Dict[str, object]],
        query_ids: Sequence[int],
        token_capacity: int,
        token_score_query_window: int = 8,
        token_score_layers: int = 1,
        token_score_layer_mode: str = "first",
        token_score_reduce: str = "sum",
        token_score_pooling: str = "maxpool",
        token_score_pool_kernel: int = 7,
        token_score_direction: str = "query_to_chunk",
        token_score_keep: str = "high",
        token_score_include_prefix: bool = True,
        token_attention_mask: str = "causal",
        token_score_backend: str = "torch",
    ) -> Tuple[List[List[List[int]]], List[Dict[str, int]]]:
        capacity = int(token_capacity or 0)
        query_ids = self._window_query(query_ids, token_score_query_window)
        num_layers = len(self.model.model.layers)
        selected_layers = self._select_attention_layer_indices(num_layers, token_score_layers, token_score_layer_mode)
        if not selected_layers:
            selected_layers = [0]

        all_layer_keeps: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
        chunk_meta: List[Dict[str, int]] = []
        for span in chunk_spans:
            chunk_ids = [int(x) for x in span["chunk_ids"]]
            start = int(span["start"])
            end = int(span["end"])
            chunk_len = len(chunk_ids)
            keep_count = min(max(1, capacity), chunk_len) if capacity > 0 else chunk_len
            if chunk_len <= 0:
                chunk_meta.append({"start": start, "end": end, "kept_tokens": 0, "union_kept_tokens": 0})
                continue
            if capacity <= 0 or chunk_len <= capacity or not query_ids:
                base = torch.arange(chunk_len, device=torch.cuda.current_device(), dtype=torch.long)
                per_layer_keep = [base.unsqueeze(0).expand(int(self.config.hf_config.num_key_value_heads), -1).clone() for _ in range(num_layers)]
            else:
                score_prefix_ids = list(int(x) for x in prefix_ids) if token_score_include_prefix else []
                prefix_len = len(score_prefix_ids)
                joint_ids = score_prefix_ids + chunk_ids + list(query_ids)
                positions = self._positions(len(joint_ids), 0)
                input_ids = self._ids_tensor(joint_ids)
                slot_mapping = torch.arange(len(joint_ids), dtype=torch.int32, device=torch.cuda.current_device())
                self._set_full_prefill_context(len(joint_ids), slot_mapping, need_kv_cache_store=False)
                scores_by_layer: Dict[int, torch.Tensor] = {}
                try:
                    hidden_states = self.model.model.embed_tokens(input_ids)
                    residual = None
                    for layer_idx, layer in enumerate(self.model.model.layers):
                        if residual is None:
                            residual = hidden_states
                            normed = layer.input_layernorm(hidden_states)
                        else:
                            normed, residual = layer.input_layernorm(hidden_states, residual)
                        if layer_idx in selected_layers:
                            attn = layer.self_attn
                            q = attn.q_proj(normed)
                            k = attn.k_proj(normed)
                            q, k = attn.rotary_emb(positions, q, k)
                            q = q.view(len(joint_ids), attn.num_heads, attn.head_dim)
                            k = k.view(len(joint_ids), attn.num_kv_heads, attn.head_dim)
                            scores = self._engine_chunk_attention_head_scores(
                                q,
                                k,
                                prefix_len=prefix_len,
                                chunk_len=chunk_len,
                                query_len=len(query_ids),
                                direction=token_score_direction,
                                reduce_mode=token_score_reduce,
                                attention_mask=token_attention_mask,
                                backend=token_score_backend,
                            )
                            if scores.numel() > 0 and scores.shape[-1] == chunk_len:
                                pooling = (token_score_pooling or "none").lower()
                                if pooling != "none" and scores.shape[-1] > 0:
                                    kernel = max(1, min(int(token_score_pool_kernel or 1), scores.shape[-1]))
                                    padding = kernel // 2
                                    pooled = scores.unsqueeze(1)
                                    if pooling == "avgpool":
                                        pooled = F.avg_pool1d(pooled, kernel_size=kernel, padding=padding, stride=1)
                                    elif pooling == "maxpool":
                                        pooled = F.max_pool1d(pooled, kernel_size=kernel, padding=padding, stride=1)
                                    else:
                                        raise ValueError(f"Unsupported token_score_pooling: {token_score_pooling}")
                                    scores = pooled.squeeze(1)[..., :chunk_len]
                                if scores.shape[0] != int(attn.num_kv_heads):
                                    if scores.shape[0] % int(attn.num_kv_heads) != 0:
                                        raise ValueError(
                                            f"Cannot group {scores.shape[0]} query-head scores into "
                                            f"{attn.num_kv_heads} KV heads."
                                        )
                                    grouped_scores = []
                                    for head_group in torch.tensor_split(scores, int(attn.num_kv_heads), dim=0):
                                        if head_group.shape[0] == 0:
                                            grouped_scores.append(
                                                torch.zeros(chunk_len, device=scores.device, dtype=scores.dtype)
                                            )
                                        else:
                                            grouped_scores.append(head_group.mean(dim=0))
                                    scores = torch.stack(grouped_scores, dim=0)
                                scores_by_layer[layer_idx] = scores
                        hidden_states = layer.self_attn(positions, normed)
                        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
                        hidden_states = layer.mlp(hidden_states)
                finally:
                    reset_context_diffusion_lm()

                selected_keeps: List[torch.Tensor] = []
                for layer_idx in selected_layers:
                    scores = scores_by_layer.get(layer_idx)
                    if scores is None:
                        fallback = torch.arange(chunk_len, device=torch.cuda.current_device(), dtype=torch.long)
                        if keep_count < chunk_len:
                            fallback = self._select_positions_from_token_scores(
                                torch.empty(0, device=torch.cuda.current_device()),
                                capacity=keep_count,
                                chunk_len=chunk_len,
                                keep_high=token_score_keep == "high",
                            )
                        selected_keeps.append(
                            fallback.unsqueeze(0)
                            .expand(int(self.model.model.layers[0].self_attn.num_kv_heads), -1)
                            .clone()
                        )
                        continue
                    head_keeps = []
                    for head_idx in range(scores.shape[0]):
                        head_keeps.append(
                            self._select_positions_from_token_scores(
                                scores[head_idx],
                                capacity=keep_count,
                                chunk_len=chunk_len,
                                keep_high=token_score_keep == "high",
                            )
                        )
                    selected_keeps.append(torch.stack(head_keeps, dim=0))
                if not selected_keeps:
                    fallback = torch.arange(chunk_len, device=torch.cuda.current_device(), dtype=torch.long)
                    if keep_count < chunk_len:
                        fallback = self._select_positions_from_token_scores(
                            torch.empty(0, device=torch.cuda.current_device()),
                            capacity=keep_count,
                            chunk_len=chunk_len,
                            keep_high=token_score_keep == "high",
                        )
                    selected_keeps.append(
                        fallback.unsqueeze(0)
                        .expand(int(self.model.model.layers[0].self_attn.num_kv_heads), -1)
                        .clone()
                    )
                per_layer_keep = [
                    selected_keeps[min(layer_idx, len(selected_keeps) - 1)].clone()
                    for layer_idx in range(num_layers)
                ]

            union_parts = [keep.reshape(-1) for keep in per_layer_keep if keep.numel() > 0]
            union_keep = torch.sort(torch.unique(torch.cat(union_parts, dim=0))).values if union_parts else torch.empty(0, device=torch.cuda.current_device(), dtype=torch.long)
            for layer_idx, layer_keep in enumerate(per_layer_keep):
                all_layer_keeps[layer_idx].append(layer_keep + start)
            chunk_meta.append(
                {
                    "start": start,
                    "end": end,
                    "kept_tokens": keep_count,
                    "union_kept_tokens": int(union_keep.numel()),
                }
            )

        prompt_keep: List[List[List[int]]] = []
        for layer_idx, layer_parts in enumerate(all_layer_keeps):
            if not layer_parts:
                base = torch.arange(full_prompt_len, device=torch.cuda.current_device(), dtype=torch.long)
                layer_keep = base.unsqueeze(0).expand(int(self.model.model.layers[0].self_attn.num_kv_heads), -1).clone()
                prompt_keep.append(layer_keep.tolist())
                continue
            num_heads = int(layer_parts[0].shape[0])
            head_indices = []
            for head_idx in range(num_heads):
                pieces = []
                cursor = 0
                for span, layer_keep in zip(chunk_spans, layer_parts):
                    start = int(span["start"])
                    end = int(span["end"])
                    if start > cursor:
                        pieces.append(torch.arange(cursor, start, device=torch.cuda.current_device(), dtype=torch.long))
                    pieces.append(layer_keep[head_idx].to(device=torch.cuda.current_device(), dtype=torch.long))
                    cursor = end
                if cursor < full_prompt_len:
                    pieces.append(torch.arange(cursor, full_prompt_len, device=torch.cuda.current_device(), dtype=torch.long))
                head_indices.append(torch.cat(pieces, dim=0))
            prompt_keep.append(torch.stack(head_indices, dim=0).tolist())
        return prompt_keep, chunk_meta

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
    def generate_token_ids_from_chunk_local_kv(
        self,
        *,
        prefix_ids: Sequence[int],
        prefix_positions: Sequence[int],
        chunk_ids_list: Sequence[Sequence[int]],
        chunk_positions_list: Sequence[Sequence[int]],
        chunk_keep_indices_per_layer_per_head: Sequence[Sequence[Sequence[Sequence[int]]]],
        query_ids: Sequence[int],
        query_positions: Sequence[int],
        local_query_ids: Optional[Sequence[int]] = None,
        local_attention_mask: str = "causal",
        max_new_tokens: int,
        stop_token_ids: Optional[Iterable[int]] = None,
    ) -> FastDLLMEngineOutput:
        """Generate from ParallelComp-style chunk-local per-head compressed KV.

        Each selected chunk is locally prefetched with ``prefix + chunk + query_window``.
        The chunk K/V is gathered per layer/KV-head according to the provided local
        keep indices, then appended into the engine KV cache before the final query
        and Fast-DLLM generation block are processed.
        """
        if max_new_tokens <= 0:
            return FastDLLMEngineOutput(text="", token_ids=[], n_diff_steps=0)
        if max_new_tokens > self.block_length:
            raise NotImplementedError(
                "FastDLLMDreamEngine chunk-local KV path supports one generation block only. "
                f"Got max_new_tokens={max_new_tokens}, block_length={self.block_length}."
            )
        if self.config.kv_cache_layout != "unified":
            raise ValueError("chunk-local KV generation currently supports kv_cache_layout='unified' only.")

        prefix_ids = [int(x) for x in prefix_ids]
        prefix_positions = [int(x) for x in prefix_positions]
        query_ids = [int(x) for x in query_ids]
        query_positions = [int(x) for x in query_positions]
        if len(prefix_ids) != len(prefix_positions):
            raise ValueError("prefix_ids/prefix_positions length mismatch")
        if len(query_ids) != len(query_positions):
            raise ValueError("query_ids/query_positions length mismatch")
        if len(chunk_ids_list) != len(chunk_positions_list):
            raise ValueError("chunk ids/positions list length mismatch")
        if len(chunk_ids_list) != len(chunk_keep_indices_per_layer_per_head):
            raise ValueError("chunk ids/keep-indices list length mismatch")

        decode_len = self.block_length
        local_query_ids = list(int(x) for x in (local_query_ids if local_query_ids is not None else query_ids))
        local_query_ids = self._window_query(local_query_ids, len(local_query_ids))

        normalized_chunk_keeps: List[List[torch.Tensor]] = []
        kept_counts: List[int] = []
        for chunk_ids, chunk_keep in zip(chunk_ids_list, chunk_keep_indices_per_layer_per_head):
            tensors, active_len = self._normalize_per_head_keep_indices(
                chunk_keep,
                full_prompt_len=len(chunk_ids),
            )
            normalized_chunk_keeps.append(tensors)
            kept_counts.append(active_len)

        base_len = len(prefix_ids) + sum(kept_counts)
        prompt_len = base_len + len(query_ids)
        suffix_pos_start = (max(query_positions) + 1) if query_positions else (max(prefix_positions) + 1 if prefix_positions else 0)
        block_positions = list(range(suffix_pos_start, suffix_pos_start + decode_len))
        suffix_ids = list(query_ids) + [self.mask_token_id] * decode_len
        suffix_positions = list(query_positions) + list(block_positions)
        pad_len = (-len(suffix_ids)) % self.block_length
        if pad_len:
            pad_start = suffix_positions[-1] + 1 if suffix_positions else suffix_pos_start
            suffix_ids.extend([self.mask_token_id] * pad_len)
            suffix_positions.extend(range(pad_start, pad_start + pad_len))

        total_slots = base_len + len(suffix_ids)
        if total_slots > self.config.max_model_len:
            raise ValueError(
                f"chunk-local KV prompt length {total_slots} exceeds max_model_len={self.config.max_model_len}"
            )
        total_pages = math.ceil(total_slots / self.page_size)
        while self._prefix_cache.num_free_pages < total_pages:
            if not self._prefix_cache.evict_one():
                break
        all_page_ids = self._prefix_cache.allocate_pages(total_pages)
        temp_page_ids_list: List[List[int]] = []

        try:
            if prefix_ids:
                self._forward_prefill_into_pages(prefix_ids, prefix_positions, all_page_ids, start_token=0)

            dst_start = len(prefix_ids)
            for chunk_ids_raw, chunk_positions_raw, chunk_keep in zip(
                chunk_ids_list,
                chunk_positions_list,
                normalized_chunk_keeps,
            ):
                chunk_ids = [int(x) for x in chunk_ids_raw]
                chunk_positions = [int(x) for x in chunk_positions_raw]
                if len(chunk_ids) != len(chunk_positions):
                    raise ValueError("chunk ids/positions length mismatch")
                if not chunk_ids:
                    continue
                local_query_start = (max(chunk_positions) + 1) if chunk_positions else len(prefix_ids)
                local_query_positions = list(range(local_query_start, local_query_start + len(local_query_ids)))
                local_ids = list(prefix_ids) + list(chunk_ids) + list(local_query_ids)
                local_positions = list(prefix_positions) + list(chunk_positions) + local_query_positions
                temp_pages = math.ceil(len(local_ids) / self.page_size)
                while self._prefix_cache.num_free_pages < temp_pages:
                    if not self._prefix_cache.evict_one():
                        break
                temp_page_ids = self._prefix_cache.allocate_pages(temp_pages)
                temp_page_ids_list.append(temp_page_ids)
                self._forward_prefill_into_pages(
                    local_ids,
                    local_positions,
                    temp_page_ids,
                    start_token=0,
                    attention_mask=local_attention_mask,
                    prefix_len=len(prefix_ids),
                    chunk_len=len(chunk_ids),
                    query_len=len(local_query_ids),
                )
                copied = self._copy_chunk_local_kv_to_prompt_pages(
                    src_page_ids=temp_page_ids,
                    dst_page_ids=all_page_ids,
                    src_chunk_start=len(prefix_ids),
                    dst_start=dst_start,
                    keep_indices_per_layer_per_head=chunk_keep,
                )
                dst_start += copied
                self._prefix_cache.release_pages(temp_page_ids)
                temp_page_ids_list.pop()
            if dst_start != base_len:
                raise ValueError(f"chunk-local KV base length mismatch: {dst_start} vs {base_len}")

            suffix_logits = self._forward_append_tokens_paged(
                suffix_ids,
                suffix_positions,
                context_len=base_len,
                all_page_ids=all_page_ids,
                start_token=base_len,
                valid_tokens=len(query_ids) + decode_len,
            )
            query_len = len(query_ids)
            shifted_suffix = self._shift_logits(suffix_logits)
            first_logits = shifted_suffix[query_len:query_len + 1, :]
            _, first_token = self._sample_tokens(first_logits)
            last_context_logit = suffix_logits[query_len - 1, :].detach() if query_len > 0 else None

            block_ids = torch.full((decode_len,), self.mask_token_id, dtype=torch.long, device=torch.cuda.current_device())
            block_ids[0] = first_token[0]
            decode_pages = all_page_ids[: math.ceil((prompt_len + decode_len) / self.page_size)]
            n_steps = 0
            while bool((block_ids == self.mask_token_id).any()):
                n_steps += 1
                mask_index = block_ids == self.mask_token_id
                logits = self._forward_append_tokens_paged(
                    block_ids.tolist(),
                    block_positions,
                    context_len=prompt_len,
                    all_page_ids=decode_pages,
                    start_token=prompt_len,
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

            generated = block_ids[:max_new_tokens].tolist()
        finally:
            for temp_page_ids in temp_page_ids_list:
                self._prefix_cache.release_pages(temp_page_ids)
            self._prefix_cache.release_pages(all_page_ids)

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

    @torch.inference_mode()
    def generate_token_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int,
        prompt_positions: Optional[Sequence[int]] = None,
        active_prompt_positions: Optional[Sequence[int]] = None,
        prompt_keep_indices_per_layer_per_head: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        decode_delta_mode: str = "none",
        decode_delta_stride: int = 4,
        decode_delta_left: int = 3,
        decode_delta_scale: float = 1.0,
        decode_delta_debug: bool = False,
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

        per_head_keep_indices: Optional[List[torch.Tensor]] = None
        per_head_active_len: Optional[int] = None
        if prompt_keep_indices_per_layer_per_head is not None:
            per_head_keep_indices, per_head_active_len = self._normalize_per_head_keep_indices(
                prompt_keep_indices_per_layer_per_head,
                full_prompt_len=len(prompt_ids),
            )
            if active_prompt_positions is None:
                raise ValueError("active_prompt_positions is required for per-head KV eviction.")
            active_prompt_positions = [int(x) for x in active_prompt_positions]
            if len(active_prompt_positions) != per_head_active_len:
                raise ValueError(
                    "active_prompt_positions length mismatch: "
                    f"{len(active_prompt_positions)} vs per-head active length {per_head_active_len}"
                )
        elif active_prompt_positions is not None:
            raise ValueError("active_prompt_positions requires prompt_keep_indices_per_layer_per_head.")

        prefill_suffix_pos_start = (max(prompt_positions) + 1) if prompt_positions else 0
        prefill_suffix_positions = list(range(prefill_suffix_pos_start, prefill_suffix_pos_start + decode_len))
        if per_head_keep_indices is not None:
            suffix_pos_start = (max(active_prompt_positions) + 1) if active_prompt_positions else 0
            suffix_positions = list(range(suffix_pos_start, suffix_pos_start + decode_len))
            prompt_len = int(per_head_active_len or 0)
            full_prompt_len = len(prompt_ids)
        else:
            suffix_positions = prefill_suffix_positions
            prompt_len = len(prompt_ids)
            full_prompt_len = prompt_len

        full_len = full_prompt_len + decode_len
        if full_len > self.config.max_model_len:
            raise ValueError(
                f"full_prompt_mask length {full_len} exceeds max_model_len={self.config.max_model_len}"
            )

        num_prompt_pages = math.ceil(full_prompt_len / self.page_size)
        num_block_pages = math.ceil(decode_len / self.page_size)
        use_prefix_cache = per_head_keep_indices is None
        decode_delta_mode = (decode_delta_mode or "none").lower()
        decode_delta_state = None

        cached = self._prefix_cache.lookup_prefix(prompt_ids) if use_prefix_cache else None
        owns_prompt_pages = False
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
            owns_prompt_pages = not use_prefix_cache
            block_page_ids = self._prefix_cache.allocate_pages(num_block_pages)

            full_ids = prompt_ids + [self.mask_token_id] * decode_len
            full_positions = list(prompt_positions) + prefill_suffix_positions
            prefill_logits = self._forward_prefill_paged(
                full_ids, full_positions, prompt_page_ids, block_page_ids, full_prompt_len
            )
            if per_head_keep_indices is not None:
                if decode_delta_mode in {"s4_left3", "delta_s4_left3", "sampled_delta_s4_left3"}:
                    full_k, full_v = self._snapshot_prompt_cache_per_layer(prompt_page_ids, full_prompt_len)
                    stride = max(1, int(decode_delta_stride or 4))
                    decode_delta_state = {
                        "enabled": True,
                        "mode": "s4_left3",
                        "full_k": full_k,
                        "full_v": full_v,
                        "full_prompt_len": full_prompt_len,
                        "active_prompt_len": int(per_head_active_len or 0),
                        "stride": stride,
                        "left": max(0, int(decode_delta_left)),
                        "anchor_offset": stride - 1,
                        "scale": float(decode_delta_scale),
                        "debug": bool(decode_delta_debug),
                    }
                elif decode_delta_mode not in {"none", "off", "0", ""}:
                    raise ValueError(f"Unsupported decode_delta_mode: {decode_delta_mode}")
                self._compact_prompt_cache_per_layer_per_head(prompt_page_ids, per_head_keep_indices)
                active_prompt_pages = math.ceil(prompt_len / self.page_size)
                if active_prompt_pages < len(prompt_page_ids):
                    self._prefix_cache.release_pages(prompt_page_ids[active_prompt_pages:])
                    prompt_page_ids = prompt_page_ids[:active_prompt_pages]
            shifted_prefill = self._shift_logits(prefill_logits)
            first_logits = shifted_prefill[full_prompt_len:full_prompt_len + 1, :]
            _, first_token = self._sample_tokens(first_logits)
            last_context_logit = prefill_logits[full_prompt_len - 1, :].detach() if full_prompt_len > 0 else None
            if use_prefix_cache:
                self._prefix_cache.register_prefix(prompt_ids, prompt_page_ids, prompt_len, last_context_logit)

        block_ids = torch.full((decode_len,), self.mask_token_id, dtype=torch.long, device=torch.cuda.current_device())
        block_ids[0] = first_token[0]
        all_page_ids_for_decode = prompt_page_ids + block_page_ids
        n_steps = 0
        previous_delta_state = self._active_decode_delta_state
        self._active_decode_delta_state = decode_delta_state
        try:
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
        finally:
            self._active_decode_delta_state = previous_delta_state
        if decode_delta_state is not None and decode_delta_state.get("debug", False):
            print(
                "[decode-delta] "
                f"mode={decode_delta_state.get('mode')} "
                f"calls={decode_delta_state.get('calls', 0)} "
                f"nonzero_calls={decode_delta_state.get('nonzero_calls', 0)} "
                f"anchor_count={decode_delta_state.get('anchor_count', 0)} "
                f"max_abs={float(decode_delta_state.get('max_abs', 0.0)):.6e}",
                flush=True,
            )

        self._prefix_cache.release_pages(block_page_ids)
        if owns_prompt_pages:
            self._prefix_cache.release_pages(prompt_page_ids)
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
