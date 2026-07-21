import queue
import sys
import threading
import time
import unittest
import importlib.util
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OrderedPrefetchPipelineTests(unittest.TestCase):
    def test_prepare_next_item_runs_while_current_item_is_consumed(self):
        ordered_prefetch_map = load_module_from_path(
            "pd_pipeline_under_test",
            ROOT / "d2f_vllm" / "d2f_vllm" / "pd_pipeline.py",
        ).ordered_prefetch_map

        prepare_events: "queue.Queue[tuple[str, int]]" = queue.Queue()
        release_events: list[str] = []
        consume_started = threading.Event()
        item_two_prepared = threading.Event()

        def prepare(item: int) -> str:
            prepare_events.put(("prepare_start", item))
            if item == 2:
                item_two_prepared.set()
            return f"record-{item}"

        def consume(item: int, record: str) -> str:
            if item == 1:
                consume_started.set()
                self.assertTrue(
                    item_two_prepared.wait(timeout=1.0),
                    "item 2 should be prepared while item 1 is still consuming",
                )
            return f"{item}:{record}"

        results = list(
            ordered_prefetch_map(
                [1, 2],
                prepare=prepare,
                consume=consume,
                release_prepared=release_events.append,
            )
        )

        self.assertTrue(consume_started.is_set())
        self.assertEqual(results, ["1:record-1", "2:record-2"])
        self.assertEqual(release_events, [])
        self.assertEqual(
            [prepare_events.get_nowait() for _ in range(2)],
            [("prepare_start", 1), ("prepare_start", 2)],
        )

    def test_releases_prefetched_item_when_consume_fails(self):
        ordered_prefetch_map = load_module_from_path(
            "pd_pipeline_under_test",
            ROOT / "d2f_vllm" / "d2f_vllm" / "pd_pipeline.py",
        ).ordered_prefetch_map

        prepared_second = threading.Event()
        releases: list[str] = []

        def prepare(item: int) -> str:
            if item == 2:
                prepared_second.set()
            return f"record-{item}"

        def consume(item: int, record: str) -> str:
            if item == 1:
                self.assertTrue(prepared_second.wait(timeout=1.0))
                raise RuntimeError("decode failed")
            return record

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            list(
                ordered_prefetch_map(
                    [1, 2],
                    prepare=prepare,
                    consume=consume,
                    release_prepared=releases.append,
                )
            )

        self.assertEqual(releases, ["record-2"])


class ThreadLocalContextTests(unittest.TestCase):
    def test_diffusion_context_is_isolated_between_threads(self):
        sequence_module = types.ModuleType("d2f_vllm.engine.sequence")
        sequence_module.SequenceForDiffusionLM = object
        sys.modules.setdefault("d2f_vllm", types.ModuleType("d2f_vllm"))
        sys.modules.setdefault("d2f_vllm.engine", types.ModuleType("d2f_vllm.engine"))
        sys.modules["d2f_vllm.engine.sequence"] = sequence_module
        context_module = load_module_from_path(
            "context_under_test",
            ROOT / "d2f_vllm" / "d2f_vllm" / "utils" / "context.py",
        )
        get_context_diffusion_lm = context_module.get_context_diffusion_lm
        reset_context_diffusion_lm = context_module.reset_context_diffusion_lm
        set_context_diffusion_lm = context_module.set_context_diffusion_lm

        reset_context_diffusion_lm()
        set_context_diffusion_lm(True, max_seqlen_q=11)
        worker_ready = threading.Event()
        worker_done = threading.Event()
        seen: dict[str, int] = {}

        def worker() -> None:
            reset_context_diffusion_lm()
            set_context_diffusion_lm(False, max_seqlen_q=22)
            seen["worker"] = get_context_diffusion_lm().max_seqlen_q
            worker_ready.set()
            worker_done.wait(timeout=1.0)

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=1.0))
        self.assertEqual(get_context_diffusion_lm().max_seqlen_q, 11)
        worker_done.set()
        thread.join(timeout=1.0)
        self.assertEqual(seen, {"worker": 22})


if __name__ == "__main__":
    unittest.main()
