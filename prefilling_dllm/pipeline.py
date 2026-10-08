"""Prefilling-dLLM as one call: document and question in, answer out.

Issues the draft, chunk-scoring, and generation requests of one answer against
a PrefillingDream SGLang server, or against a PD router plus a scoring server.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional, Protocol

import msgspec

from prefilling_dllm.chunk_selection import (
    add_chunk_bos,
    requires_chunk_scoring,
    score_chunks,
    select_chunk_indices,
    split_token_chunks,
)
from prefilling_dllm.client import PARTIAL_DRAFT_ROUNDS, SGLangClient

# LongBench MultiFieldQA-en prompt.
DEFAULT_TEMPLATE = (
    "Read the following text and answer briefly.\n\n{context}\n\n"
    "Now, answer the following question based on the above text, only give me "
    "the answer and do not output any other words.\n\n"
    "Question: {question}\nAnswer:"
)


class Tokenizer(Protocol):
    bos_token_id: Optional[int]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class PipelineConfig(msgspec.Struct, frozen=True, kw_only=True):
    chunk_size: int = 1024
    top_k: int = 4
    chunk_bos: bool = True
    # Draft slots appended to the scoring query; 0 scores the query alone.
    draft_tokens: int = 4
    # Chunk rows per scoring request.
    score_batch_size: int = 8
    # Above the server's block size the answer is generated block by block.
    max_new_tokens: int = 32
    # Query RoPE positions start after one full chunk slot per selected chunk,
    # so a short last chunk leaves a gap instead of shifting the query.
    query_after_chunk_slots: bool = True
    # Per-chunk KV budget of per-head token eviction; 0 keeps every token.
    token_capacity: int = 0
    token_score_query_window: int = 8
    token_score_pool_kernel: int = 7
    token_score_bidirectional: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or self.score_batch_size <= 0:
            raise ValueError("chunk_size and score_batch_size must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        # The server's draft stage only supports a four-slot canvas.
        if self.draft_tokens not in (0, 4):
            raise ValueError("draft_tokens must be 0 (disabled) or 4")
        if self.token_capacity < 0:
            raise ValueError("token_capacity must be non-negative")
        if self.token_score_query_window <= 0 or self.token_score_pool_kernel <= 0:
            raise ValueError(
                "token_score_query_window and token_score_pool_kernel must be positive"
            )


class TokenizedPrompt(msgspec.Struct, frozen=True, kw_only=True):
    prefix_ids: list[int]
    chunks: list[list[int]]
    query_ids: list[int]
    context_tokens: int


class ChunkSelection(msgspec.Struct, frozen=True, kw_only=True):
    # Ascending chunk indices.
    indices: list[int]
    # None when every chunk is kept without scoring.
    scores: Optional[list[float]]
    draft_ids: list[int]
    draft_seconds: float
    score_seconds: float


class GenerationRequest(msgspec.Struct, frozen=True, kw_only=True):
    input_ids: list[int]
    position_start: int
    position_offset: int
    custom_params: Optional[dict[str, Any]]


class PipelineResult(msgspec.Struct, frozen=True, kw_only=True):
    answer: str
    num_chunks: int
    selected_chunk_indices: list[int]
    chunk_scores: Optional[list[float]]
    draft_ids: list[int]
    context_tokens: int
    prompt_tokens: int
    draft_seconds: float
    score_seconds: float
    generation_seconds: float
    generation_meta: Optional[dict[str, Any]]


def split_template(template: str, *, question: str) -> tuple[str, str]:
    """Split a prompt template at its document slot into (prefix, query)."""
    if "{question}" not in template:
        raise ValueError("prompt template is missing a {question} slot")
    prefix, slot, query = template.partition("{context}")
    if not slot:
        raise ValueError("prompt template is missing a {context} slot")
    return (
        prefix.replace("{question}", question),
        query.replace("{question}", question),
    )


def tokenize_prompt(
    *,
    tokenizer: Tokenizer,
    prefix: str,
    context: str,
    query: str,
    config: PipelineConfig,
) -> TokenizedPrompt:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if tokenizer.bos_token_id is not None:
        prefix_ids = [tokenizer.bos_token_id] + prefix_ids
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    if not query_ids:
        raise ValueError("the query part of the prompt is empty")
    chunks = split_token_chunks(context_ids, config.chunk_size)
    if config.chunk_bos:
        chunks = add_chunk_bos(chunks, tokenizer.bos_token_id, config.chunk_size)
    return TokenizedPrompt(
        prefix_ids=prefix_ids,
        chunks=chunks,
        query_ids=query_ids,
        context_tokens=len(context_ids),
    )


def select_chunks(
    *,
    client: SGLangClient,
    score_client: SGLangClient,
    prompt: TokenizedPrompt,
    config: PipelineConfig,
) -> ChunkSelection:
    """Draft from prefix + query, score every chunk, and keep the top k."""
    chunk_count = len(prompt.chunks)
    if not requires_chunk_scoring("query_logprob", chunk_count, config.top_k):
        return ChunkSelection(
            indices=list(range(chunk_count)),
            scores=None,
            draft_ids=[],
            draft_seconds=0.0,
            score_seconds=0.0,
        )

    draft_start = time.perf_counter()
    draft = client.partial_draft(
        prompt.prefix_ids + prompt.query_ids,
        config.draft_tokens,
        rounds=PARTIAL_DRAFT_ROUNDS,
    )
    score_start = time.perf_counter()
    scores = score_chunks(
        score_client,
        prompt.prefix_ids,
        prompt.chunks,
        prompt.query_ids + draft.token_ids,
        config.score_batch_size,
        # Unconfirmed draft slots still condition the row but are not scored.
        score_token_mask=[True] * len(prompt.query_ids) + draft.confirmed_mask,
        draft_len=len(draft.token_ids),
    )
    score_end = time.perf_counter()
    return ChunkSelection(
        indices=select_chunk_indices(
            "query_logprob", chunk_count, config.top_k, scores
        ),
        scores=scores,
        draft_ids=draft.token_ids,
        draft_seconds=score_start - draft_start,
        score_seconds=score_end - score_start,
    )


def build_generation_request(
    *, prompt: TokenizedPrompt, selected: list[int], config: PipelineConfig
) -> GenerationRequest:
    """Assemble prefix + selected chunks + query and its dLLM custom params."""
    chunk_lens = [len(prompt.chunks[chunk_index]) for chunk_index in selected]
    context_ids = [
        token_id for chunk_index in selected for token_id in prompt.chunks[chunk_index]
    ]
    query_start = len(prompt.prefix_ids) + len(context_ids)
    position_offset = 0
    if config.query_after_chunk_slots:
        query_rope_start = len(prompt.prefix_ids) + len(selected) * config.chunk_size
        position_offset = query_rope_start - query_start

    custom_params = None
    if config.token_capacity > 0 and selected:
        custom_params = {
            "dllm_token_eviction": {
                "capacity": config.token_capacity,
                "prefix_len": len(prompt.prefix_ids),
                "chunk_lens": chunk_lens,
                "query_len": len(prompt.query_ids),
                "score_query_window": config.token_score_query_window,
                "pool_kernel": config.token_score_pool_kernel,
                "force_keep_first": config.chunk_bos,
                "bidirectional": config.token_score_bidirectional,
            }
        }
    return GenerationRequest(
        input_ids=prompt.prefix_ids + context_ids + prompt.query_ids,
        position_start=query_start,
        position_offset=position_offset,
        custom_params=custom_params,
    )


class PrefillingDreamPipeline:
    """Answers questions over long documents; safe to share across threads."""

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        client: SGLangClient,
        score_client: Optional[SGLangClient] = None,
        config: PipelineConfig = PipelineConfig(),
    ):
        self.tokenizer = tokenizer
        # Drafts and generation; a PD router in a disaggregated deployment.
        self.client = client
        self.score_client = client if score_client is None else score_client
        self.config = config
        # Hugging Face fast tokenizers can raise "Already borrowed" when
        # several threads encode at once.
        self._tokenizer_lock = threading.Lock()

    def answer(
        self, *, context: str, question: str, template: str = DEFAULT_TEMPLATE
    ) -> PipelineResult:
        prefix, query = split_template(template, question=question)
        return self.answer_parts(prefix=prefix, context=context, query=query)

    def answer_parts(self, *, prefix: str, context: str, query: str) -> PipelineResult:
        """Answer a prompt given as the text before and after the document."""
        with self._tokenizer_lock:
            prompt = tokenize_prompt(
                tokenizer=self.tokenizer,
                prefix=prefix,
                context=context,
                query=query,
                config=self.config,
            )
        selection = select_chunks(
            client=self.client,
            score_client=self.score_client,
            prompt=prompt,
            config=self.config,
        )
        request = build_generation_request(
            prompt=prompt, selected=selection.indices, config=self.config
        )
        generation_start = time.perf_counter()
        generation = self.client.generate(
            request.input_ids,
            self.config.max_new_tokens,
            position_start=request.position_start,
            position_offset=request.position_offset,
            custom_params=request.custom_params,
        )
        generation_seconds = time.perf_counter() - generation_start
        return PipelineResult(
            answer=generation["text"],
            num_chunks=len(prompt.chunks),
            selected_chunk_indices=selection.indices,
            chunk_scores=selection.scores,
            draft_ids=selection.draft_ids,
            context_tokens=prompt.context_tokens,
            prompt_tokens=len(request.input_ids),
            draft_seconds=selection.draft_seconds,
            score_seconds=selection.score_seconds,
            generation_seconds=generation_seconds,
            generation_meta=generation.get("meta_info"),
        )
