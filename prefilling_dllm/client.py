"""Blocking HTTP client for the Dream requests of a PrefillingDream SGLang server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, NamedTuple, Sequence

PARTIAL_DRAFT_ROUNDS = 1


class PartialDraft(NamedTuple):
    token_ids: list[int]
    confirmed_mask: list[bool]


class SGLangClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        causal_prompt_logprobs: bool = False,
        score_attention_mask: str = "full",
        score_on_pd_server: bool = False,
    ):
        self.score_on_pd_server = score_on_pd_server
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.generate_url = f"{base_url}/generate"
        self.timeout = timeout
        self.causal_prompt_logprobs = causal_prompt_logprobs
        mask = (score_attention_mask or "full").lower()
        if mask not in {"causal", "full"}:
            raise ValueError(
                "score_attention_mask must be 'causal' or 'full', "
                f"got {score_attention_mask!r}"
            )
        if causal_prompt_logprobs and mask == "full":
            mask = "causal"
        self.score_attention_mask = mask

    def post(self, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang returned HTTP {error.code}: {body}") from error

    def prompt_logprobs(
        self,
        input_ids: Sequence[Sequence[int]],
        logprob_start_lens: Sequence[int],
        *,
        prefix_lens: Sequence[int] | None = None,
        chunk_lens: Sequence[int] | None = None,
        query_lens: Sequence[int] | None = None,
        draft_lens: Sequence[int] | None = None,
    ) -> list[list[Any]]:
        rows = [list(ids) for ids in input_ids]
        if self.score_attention_mask == "full":
            if prefix_lens is None or chunk_lens is None or query_lens is None:
                raise ValueError(
                    "full Dream scoring requires prefix_lens, chunk_lens, and query_lens"
                )
            if draft_lens is None:
                draft_lens = [0] * len(rows)
            if not (
                len(rows)
                == len(prefix_lens)
                == len(chunk_lens)
                == len(query_lens)
                == len(draft_lens)
            ):
                raise ValueError(
                    "full Dream scoring spans must match the number of scoring rows"
                )
            sampling_params = [
                {
                    "temperature": 0,
                    "max_new_tokens": 0,
                    "custom_params": {
                        "dream_score_attention_mask": "full",
                        "dream_score_prefix_len": int(prefix_len),
                        "dream_score_chunk_len": int(chunk_len),
                        "dream_score_query_len": int(query_len),
                        "dream_score_draft_len": int(draft_len),
                    },
                }
                for prefix_len, chunk_len, query_len, draft_len in zip(
                    prefix_lens, chunk_lens, query_lens, draft_lens
                )
            ]
        else:
            sampling_params = {
                "temperature": 0,
                "max_new_tokens": 0,
                "custom_params": {
                    "dream_causal_prompt_logprob": True,
                    "dream_score_attention_mask": "causal",
                },
            }
        payload = {
            "input_ids": rows,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": list(logprob_start_lens),
        }
        if self.score_on_pd_server:
            # A PD server only accepts requests that carry a bootstrap room;
            # scoring requests finish on the server they are sent to.
            payload["bootstrap_room"] = [
                int.from_bytes(os.urandom(7), "big") for _ in rows
            ]
        result = self.post(payload)
        payloads = result if isinstance(result, list) else [result]
        return [row["meta_info"]["input_token_logprobs"] for row in payloads]

    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        position_start: int | None = None,
        position_offset: int = 0,
        custom_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sampling_params = self._generation_sampling_params(
            max_new_tokens,
            position_start,
            position_offset,
            custom_params,
        )
        return self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": sampling_params,
            }
        )

    @staticmethod
    def _generation_sampling_params(
        max_new_tokens: int,
        position_start: int | None,
        position_offset: int,
        custom_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        }
        request_custom_params = dict(custom_params or {})
        if position_start is not None and position_offset:
            request_custom_params.update(
                {
                    "dllm_position_start": position_start,
                    "dllm_position_offset": position_offset,
                }
            )
        if request_custom_params:
            sampling_params["custom_params"] = request_custom_params
        return sampling_params

    def generate_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        *,
        position_starts: Sequence[int | None] | None = None,
        position_offsets: Sequence[int] | None = None,
        custom_params: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        batch_size = len(input_ids)
        if batch_size == 0:
            return []
        if position_starts is None:
            position_starts = [None] * batch_size
        if position_offsets is None:
            position_offsets = [0] * batch_size
        if custom_params is None:
            custom_params = [None] * batch_size
        for name, values in (
            ("position_starts", position_starts),
            ("position_offsets", position_offsets),
            ("custom_params", custom_params),
        ):
            if len(values) != batch_size:
                raise ValueError(
                    f"{name} has {len(values)} rows, expected {batch_size}"
                )
        if batch_size == 1:
            return [
                self.generate(
                    input_ids[0],
                    max_new_tokens,
                    position_start=position_starts[0],
                    position_offset=position_offsets[0],
                    custom_params=custom_params[0],
                )
            ]

        sampling_params = [
            self._generation_sampling_params(
                max_new_tokens,
                position_start,
                position_offset,
                request_custom_params,
            )
            for position_start, position_offset, request_custom_params in zip(
                position_starts,
                position_offsets,
                custom_params,
                strict=True,
            )
        ]
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": sampling_params,
            }
        )
        rows = result if isinstance(result, list) else [result]
        if len(rows) != batch_size:
            raise RuntimeError(
                "Dream final generation returned "
                f"{len(rows)} rows, expected {batch_size}"
            )
        invalid_rows = [
            index for index, row in enumerate(rows) if not isinstance(row, dict)
        ]
        if invalid_rows:
            raise RuntimeError(
                "Dream final generation returned non-object rows at indices "
                f"{invalid_rows}"
            )
        return rows

    def partial_draft(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        *,
        rounds: int = PARTIAL_DRAFT_ROUNDS,
    ) -> PartialDraft:
        if rounds < 0:
            raise ValueError("partial draft rounds must be non-negative")
        if max_new_tokens <= 0:
            return PartialDraft([], [])
        result = self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "custom_params": {
                        "dllm_partial_draft": {"rounds": rounds},
                    },
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        return self._partial_draft_from_result(result, max_new_tokens, rounds)

    def partial_draft_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        *,
        rounds: int = PARTIAL_DRAFT_ROUNDS,
    ) -> list[PartialDraft]:
        if rounds < 0:
            raise ValueError("partial draft rounds must be non-negative")
        if max_new_tokens <= 0:
            return [PartialDraft([], []) for _ in input_ids]
        if not input_ids:
            return []
        if len(input_ids) == 1:
            return [
                self.partial_draft(
                    input_ids[0],
                    max_new_tokens,
                    rounds=rounds,
                )
            ]
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "custom_params": {
                        "dllm_partial_draft": {"rounds": rounds},
                    },
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        rows = result if isinstance(result, list) else [result]
        if len(rows) != len(input_ids):
            raise RuntimeError(
                "Dream partial draft generation returned "
                f"{len(rows)} rows, expected {len(input_ids)}"
            )
        return [
            self._partial_draft_from_result(row, max_new_tokens, rounds) for row in rows
        ]

    @staticmethod
    def _partial_draft_from_result(
        result: dict[str, Any], max_new_tokens: int, rounds: int
    ) -> PartialDraft:
        output_ids = result.get("output_ids", [])
        if output_ids and isinstance(output_ids[0], list):
            if len(output_ids) != 1:
                raise RuntimeError(
                    "Dream partial draft response contains multiple output-id rows"
                )
            output_ids = output_ids[0]
        token_ids = [int(token_id) for token_id in output_ids]
        if len(token_ids) != max_new_tokens:
            raise RuntimeError(
                "Dream partial draft generation returned "
                f"{len(token_ids)} slots, expected {max_new_tokens}"
            )

        meta_info = result.get("meta_info")
        if not isinstance(meta_info, dict) or "dllm_confirmed_mask" not in meta_info:
            raise RuntimeError(
                "Dream partial draft response is missing "
                "meta_info.dllm_confirmed_mask"
            )
        confirmed_mask = meta_info["dllm_confirmed_mask"]
        if not isinstance(confirmed_mask, list):
            raise RuntimeError("Dream partial draft confirmed mask must be a list")
        if len(confirmed_mask) != max_new_tokens:
            raise RuntimeError(
                "Dream partial draft confirmed mask has "
                f"{len(confirmed_mask)} slots, expected {max_new_tokens}"
            )
        if any(type(value) is not bool for value in confirmed_mask):
            raise RuntimeError(
                "Dream partial draft confirmed mask must contain only booleans"
            )
        expected_confirmed = min(max_new_tokens, 1 + rounds)
        if sum(confirmed_mask) != expected_confirmed:
            raise RuntimeError(
                "Dream partial draft confirmed "
                f"{sum(confirmed_mask)} slots, expected {expected_confirmed}"
            )
        if max_new_tokens and not confirmed_mask[0]:
            raise RuntimeError("Dream partial draft must confirm slot zero")
        return PartialDraft(token_ids, list(confirmed_mask))

    def draft_ids(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        generation_block_size: int,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []
        request_tokens = max(max_new_tokens, generation_block_size)
        result = self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": request_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        return self._draft_ids_from_result(result, max_new_tokens)

    def draft_ids_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        generation_block_size: int,
    ) -> list[list[int]]:
        if max_new_tokens <= 0:
            return [[] for _ in input_ids]
        if not input_ids:
            return []
        if len(input_ids) == 1:
            return [
                self.draft_ids(
                    input_ids[0],
                    max_new_tokens,
                    generation_block_size,
                )
            ]
        request_tokens = max(max_new_tokens, generation_block_size)
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": request_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        rows = result if isinstance(result, list) else [result]
        if len(rows) != len(input_ids):
            raise RuntimeError(
                "Dream draft generation returned "
                f"{len(rows)} rows, expected {len(input_ids)}"
            )
        return [self._draft_ids_from_result(row, max_new_tokens) for row in rows]

    @staticmethod
    def _draft_ids_from_result(
        result: dict[str, Any], max_new_tokens: int
    ) -> list[int]:
        values = result["meta_info"].get("output_token_logprobs", [])
        token_ids = [
            int(value[1])
            for value in values[:max_new_tokens]
            if len(value) > 1 and value[1] is not None
        ]
        if not token_ids:
            # Diffusion generation returns the sampled ids at the top level but
            # currently leaves output_token_logprobs empty.  Falling back to
            # output_ids keeps draft-self-information scoring equivalent to
            # the reference engine instead of silently scoring without drafts.
            output_ids = result.get("output_ids", [])
            if output_ids and isinstance(output_ids[0], list):
                output_ids = output_ids[0]
            token_ids = [int(token_id) for token_id in output_ids[:max_new_tokens]]
        if len(token_ids) != max_new_tokens:
            raise RuntimeError(
                "Dream draft generation returned "
                f"{len(token_ids)} tokens, expected {max_new_tokens}"
            )
        return token_ids
