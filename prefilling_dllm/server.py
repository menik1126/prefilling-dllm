#!/usr/bin/env python3
"""HTTP front end of the Prefilling-dLLM pipeline.

``POST /answer`` takes ``{"context": ..., "question": ...}`` (plus an optional
``template`` with ``{context}`` and ``{question}`` slots) and returns the
answer with the selected chunks and per-stage timings. The service holds no
model: it tokenizes, then calls a PrefillingDream SGLang server.
"""

from __future__ import annotations

import argparse
import logging
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgspec

from prefilling_dllm.client import SGLangClient
from prefilling_dllm.pipeline import (
    DEFAULT_TEMPLATE,
    PipelineConfig,
    PrefillingDreamPipeline,
)

logger = logging.getLogger(__name__)


class AnswerRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    context: str
    question: str
    template: str = DEFAULT_TEMPLATE


class PipelineHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # Arbitrary; the socketserver default of 5 refuses bursts of clients.
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], pipeline: PrefillingDreamPipeline):
        super().__init__(address, PipelineRequestHandler)
        self.pipeline = pipeline


class PipelineRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: PipelineHTTPServer

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/answer":
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        try:
            request = msgspec.json.decode(body, type=AnswerRequest)
            result = self.server.pipeline.answer(
                context=request.context,
                question=request.question,
                template=request.template,
            )
        except (msgspec.DecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
        except (RuntimeError, urllib.error.URLError, TimeoutError) as error:
            # SGLangClient raises RuntimeError for SGLang HTTP errors and
            # malformed responses.
            logger.exception("SGLang request failed")
            self._send_json(502, {"error": str(error)})
        except Exception as error:
            logger.exception("pipeline failed")
            self._send_json(500, {"error": str(error)})
        else:
            self._send_json(200, result)

    def _send_json(self, status: int, payload: object) -> None:
        body = msgspec.json.encode(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve Prefilling-dLLM answers on top of a PrefillingDream SGLang server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--model-path", required=True, help="Tokenizer of the served Dream model."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="SGLang endpoint for drafts and generation: the server, or a PD router.",
    )
    parser.add_argument(
        "--score-base-url",
        help="SGLang endpoint for chunk-scoring requests; defaults to --base-url.",
    )
    parser.add_argument(
        "--score-on-pd-server",
        action="store_true",
        help=(
            "--score-base-url is a PD prefill or decode server: send scoring "
            "requests with a bootstrap room so they finish on that server."
        ),
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--chunk-bos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--token-capacity",
        type=int,
        default=0,
        help="Per-chunk KV budget of per-head token eviction; 0 keeps every token.",
    )
    parser.add_argument(
        "--token-score-direction",
        choices=("bidirectional", "query_to_chunk"),
        default="bidirectional",
    )
    return parser


def build_pipeline(args: argparse.Namespace) -> PrefillingDreamPipeline:
    from transformers import AutoTokenizer

    client = SGLangClient(args.base_url, args.timeout)
    score_client = SGLangClient(
        args.score_base_url or args.base_url,
        args.timeout,
        score_on_pd_server=args.score_on_pd_server,
    )
    return PrefillingDreamPipeline(
        tokenizer=AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        ),
        client=client,
        score_client=score_client,
        config=PipelineConfig(
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            chunk_bos=args.chunk_bos,
            draft_tokens=args.draft_tokens,
            score_batch_size=args.score_batch_size,
            max_new_tokens=args.max_new_tokens,
            token_capacity=args.token_capacity,
            token_score_bidirectional=args.token_score_direction == "bidirectional",
        ),
    )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    server = PipelineHTTPServer((args.host, args.port), build_pipeline(args))
    logger.info("Prefilling-dLLM pipeline listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
