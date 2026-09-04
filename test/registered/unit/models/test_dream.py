import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.dream import Dream, sample_tokens
from sglang.srt.dllm.algorithm.prefilling_dream import PrefillingDream
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.models.dream import DreamModel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _config(**kwargs):
    values = dict(
        algorithm="Dream",
        algorithm_config={"steps": 1},
        block_size=None,
        mask_id=99,
        max_running_requests=4,
        needs_full_prefill=True,
    )
    values.update(kwargs)
    return DllmConfig(**values)


class TestDreamRequestCanvas(CustomTestCase):
    def _make_req(self, config, *, max_new_tokens=3, custom_params=None):
        return Req(
            rid="req",
            origin_input_text="prompt",
            origin_input_ids=array("q", [10, 11]),
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens, custom_params=custom_params
            ),
            dllm_config=config,
        )

    def test_dream_initializes_one_full_generation_canvas(self):
        req = self._make_req(_config(mask_id=151666))

        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_DECODE)
        self.assertEqual(req.dllm_algo_state, {"prompt_len": 2, "step": 0})

        req.init_next_round_input()

        self.assertEqual(
            list(req.full_untruncated_fill_ids), [10, 11, 151666, 151666, 151666]
        )
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)
        self.assertEqual(req.dllm_block_offset, 0)

        req.output_ids = array("q", [20])
        req.init_next_round_input()
        self.assertEqual(
            list(req.full_untruncated_fill_ids), [10, 11, 20, 151666, 151666]
        )

    def test_block_dllm_still_uses_fixed_block_canvas(self):
        config = DllmConfig(
            algorithm="LLaDA",
            algorithm_config={},
            block_size=4,
            mask_id=156895,
            max_running_requests=1,
        )
        req = self._make_req(config)
        req.init_next_round_input()

        self.assertEqual(
            list(req.full_untruncated_fill_ids),
            [10, 11, 156895, 156895, 156895, 156895],
        )
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)

    def test_partial_draft_initializes_request_specific_canvas(self):
        config = _config(
            algorithm="PrefillingDream",
            algorithm_config={"threshold": 0.9, "dual_cache": True},
            block_size=32,
            first_done_first_out_mode=True,
        )
        req = self._make_req(
            config,
            max_new_tokens=4,
            custom_params={"dllm_partial_draft": {"rounds": 1}},
        )

        self.assertEqual(req.dllm_algo_state["canvas_len"], 4)
        self.assertEqual(req.dllm_algo_state["partial_draft_stage"], "prompt")
        self.assertIsNone(req.dllm_algo_state["partial_draft_first_token"])
        self.assertEqual(req.dllm_algo_state["partial_draft_round_limit"], 1)
        self.assertEqual(req.dllm_algo_state["partial_draft_rounds_done"], 0)
        self.assertEqual(
            req.dllm_algo_state["partial_draft_confirmed_mask"],
            [False, False, False, False],
        )
        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_PREFILL)

        req.init_next_round_input()
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 11])

        req.dllm_algo_state["partial_draft_stage"] = "suffix_init"
        req.dllm_algo_state["partial_draft_first_token"] = 7
        req.prefix_indices = torch.tensor([101, 102])
        req.dllm_initialized = False
        req.init_next_round_input()
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 11, 99, 99, 99, 99])
        self.assertEqual(req.prefix_indices.tolist(), [101, 102])

    def test_partial_draft_reset_discards_staged_kv_state(self):
        config = _config(
            algorithm="PrefillingDream",
            algorithm_config={"threshold": 0.9, "dual_cache": True},
            block_size=32,
            first_done_first_out_mode=True,
        )
        req = self._make_req(
            config,
            max_new_tokens=4,
            custom_params={"dllm_partial_draft": {"rounds": 1}},
        )
        req.dllm_algo_state.update(
            {
                "partial_draft_stage": "partial_round",
                "partial_draft_first_token": 7,
                "partial_draft_rounds_done": 1,
                "partial_draft_confirmed_mask": [True, False, True, False],
                "dual_cache_ready": True,
                "is_prefill": False,
            }
        )
        req.dllm_incomplete_ids = array("q", [7, 99, 8, 99])
        req.dllm_kv_indices = torch.tensor([201, 202, 203, 204])

        req.reset_partial_draft_state()

        self.assertEqual(req.dllm_algo_state["partial_draft_stage"], "prompt")
        self.assertIsNone(req.dllm_algo_state["partial_draft_first_token"])
        self.assertEqual(req.dllm_algo_state["partial_draft_rounds_done"], 0)
        self.assertEqual(
            req.dllm_algo_state["partial_draft_confirmed_mask"],
            [False, False, False, False],
        )
        self.assertFalse(req.dllm_incomplete_ids)
        self.assertIsNone(req.dllm_kv_indices)
        self.assertNotIn("dual_cache_ready", req.dllm_algo_state)

    def test_partial_draft_rejects_non_reference_shape(self):
        config = _config(
            algorithm="PrefillingDream",
            algorithm_config={"threshold": 0.9, "dual_cache": True},
            block_size=32,
            first_done_first_out_mode=True,
        )
        with self.assertRaisesRegex(ValueError, "max_new_tokens=4"):
            self._make_req(
                config,
                max_new_tokens=3,
                custom_params={"dllm_partial_draft": {"rounds": 1}},
            )
        with self.assertRaisesRegex(ValueError, "rounds=1"):
            self._make_req(
                config,
                max_new_tokens=4,
                custom_params={"dllm_partial_draft": {"rounds": 2}},
            )


class TestDreamAlgorithm(CustomTestCase):
    def _forward_batch(
        self,
        input_ids,
        lengths,
        *,
        top_k=None,
        block_size=None,
        canvas_lens=None,
    ):
        batch_size = len(lengths)
        return SimpleNamespace(
            input_ids=input_ids,
            extend_seq_lens_cpu=lengths,
            batch_size=batch_size,
            dllm_block_size=block_size,
            dllm_canvas_lens_cpu=canvas_lens,
            sampling_info=SimpleNamespace(
                original_temperatures=torch.ones(batch_size),
                original_top_ks=torch.tensor(
                    [-1 if top_k is None else top_k] * batch_size, dtype=torch.int32
                ),
                top_ps=torch.ones(batch_size),
            ),
        )

    def test_sample_tokens_applies_greedy_top_k(self):
        logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 1.0, 2.0]])
        confidence, tokens = sample_tokens(logits, temperature=0.0, top_k=1)

        self.assertEqual(tokens.tolist(), [1, 0])
        self.assertTrue(torch.all(confidence > 0))

    def test_dream_batches_requests_with_different_sequence_lengths(self):
        dream = Dream(_config())
        input_ids = torch.tensor([10, 11, 99, 99, 20, 99, 99])
        forward_batch = self._forward_batch(input_ids, [4, 3], top_k=1)
        logits = torch.zeros((7, 5))
        logits[:, 2] = 5.0
        states = [
            {"prompt_len": 2, "step": 0},
            {"prompt_len": 1, "step": 0},
        ]

        done = dream.step(forward_batch, logits, states)

        self.assertEqual(done, [True, True])
        self.assertEqual(forward_batch.input_ids.tolist(), [10, 11, 2, 2, 20, 2, 2])
        self.assertEqual([state["step"] for state in states], [1, 1])

    def test_dream_accepts_logits_pruned_to_generation_blocks(self):
        dream = Dream(_config())
        input_ids = torch.tensor([10, 11, 99, 99, 20, 99, 99])
        forward_batch = self._forward_batch(input_ids, [4, 3], top_k=1, block_size=2)
        logits = torch.zeros((4, 5))
        logits[:, 3] = 5.0
        states = [
            {"prompt_len": 2, "step": 0},
            {"prompt_len": 1, "step": 0},
        ]

        done = dream.step(forward_batch, logits, states)

        self.assertEqual(done, [True, True])
        self.assertEqual(forward_batch.input_ids.tolist(), [10, 11, 3, 3, 20, 3, 3])

    def test_prefilling_dream_accepts_pruned_initial_logits(self):
        algorithm = PrefillingDream(
            _config(
                algorithm="PrefillingDream",
                algorithm_config={"threshold": 0.9, "dual_cache": True},
                block_size=2,
            )
        )
        input_ids = torch.tensor([10, 11, 99, 99])
        forward_batch = self._forward_batch(input_ids, [4], block_size=2)
        logits = torch.zeros((2, 5))
        logits[:, 4] = 5.0
        states = [{"prompt_len": 2, "is_prefill": True}]

        done = algorithm.step(forward_batch, logits, states)

        self.assertEqual(done, [False])
        self.assertEqual(forward_batch.input_ids.tolist(), [10, 11, 4, 99])
        self.assertTrue(states[0]["dual_cache_ready"])

    def test_prefilling_dream_partial_draft_uses_explicit_confirmation_mask(self):
        algorithm = PrefillingDream(
            _config(
                algorithm="PrefillingDream",
                algorithm_config={"threshold": 0.9, "dual_cache": True},
                block_size=32,
                mask_id=4,
            )
        )
        state = {
            "prompt_len": 2,
            "partial_draft": True,
            "partial_draft_stage": "prompt",
            "partial_draft_first_token": None,
            "canvas_len": 4,
            "partial_draft_round_limit": 1,
            "partial_draft_rounds_done": 0,
            "partial_draft_confirmed_mask": [False, False, False, False],
        }

        first_batch = self._forward_batch(
            torch.tensor([10, 11]),
            [2],
            block_size=32,
            canvas_lens=[1],
        )
        first_logits = torch.zeros((1, 5))
        first_logits[0, 4] = 5.0

        first_done = algorithm.step(first_batch, first_logits, [state])

        self.assertEqual(first_done, [False])
        self.assertEqual(first_batch.input_ids.tolist(), [10, 11])
        self.assertEqual(state["partial_draft_stage"], "suffix_init")
        self.assertEqual(state["partial_draft_first_token"], 4)
        self.assertEqual(
            state["partial_draft_confirmed_mask"],
            [False, False, False, False],
        )
        self.assertFalse(state["last_round_was_dual_cache"])

        second_batch = self._forward_batch(
            torch.tensor([4, 4, 4, 4]),
            [4],
            block_size=32,
            canvas_lens=[4],
        )
        second_done = algorithm.step(second_batch, torch.zeros((4, 5)), [state])

        self.assertEqual(second_done, [False])
        # The first confirmed token deliberately equals mask_id. Its explicit
        # confirmed bit must keep it out of the next round's candidates.
        self.assertEqual(second_batch.input_ids.tolist(), [4, 4, 4, 4])
        self.assertEqual(state["partial_draft_stage"], "partial_round")
        self.assertEqual(
            state["partial_draft_confirmed_mask"],
            [True, False, False, False],
        )
        self.assertTrue(state["dual_cache_ready"])
        self.assertFalse(state["last_round_was_dual_cache"])

        third_batch = self._forward_batch(
            torch.tensor([4, 4, 4, 4]),
            [4],
            block_size=32,
            canvas_lens=[4],
        )
        # All three candidates exceed the normal threshold. Partial drafting
        # must nevertheless accept only the single globally most-confident row.
        third_logits = torch.zeros((4, 5))
        third_logits[0, 0] = 20.0
        third_logits[1, 1] = 5.0
        third_logits[2, 2] = 8.0
        third_logits[3, 3] = 6.0

        third_done = algorithm.step(third_batch, third_logits, [state])

        self.assertEqual(third_done, [True])
        self.assertEqual(third_batch.input_ids.tolist(), [4, 4, 2, 4])
        self.assertEqual(state["partial_draft_rounds_done"], 1)
        self.assertEqual(
            state["partial_draft_confirmed_mask"],
            [True, False, True, False],
        )
        self.assertTrue(state["last_round_was_dual_cache"])

    def test_prefilling_dream_default_threshold_path_is_unchanged(self):
        algorithm = PrefillingDream(
            _config(
                algorithm="PrefillingDream",
                algorithm_config={"threshold": 0.9, "dual_cache": True},
                block_size=3,
            )
        )
        batch = self._forward_batch(torch.tensor([99, 99, 99]), [3], block_size=3)
        logits = torch.zeros((3, 5))
        logits[0, 1] = 8.0
        logits[1, 2] = 8.0
        logits[2, 3] = 8.0
        state = {"prompt_len": 0, "is_prefill": False, "dual_cache_ready": True}

        done = algorithm.step(batch, logits, [state])

        self.assertEqual(done, [False])
        self.assertEqual(batch.input_ids.tolist(), [1, 2, 3])

    def test_dream_fdfo_carries_each_request_until_done(self):
        dream = Dream(_config(algorithm_config={"steps": 2, "alg": "origin"}))
        model_runner = MagicMock()
        model_runner.forward.side_effect = [
            SimpleNamespace(
                logits_output=SimpleNamespace(
                    full_logits=torch.zeros((4, 5)),
                ),
                can_run_graph=False,
            ),
            SimpleNamespace(
                logits_output=SimpleNamespace(
                    full_logits=torch.zeros((2, 5)),
                ),
                can_run_graph=False,
            ),
        ]

        first_batch = self._forward_batch(
            torch.tensor([10, 99, 20, 99]), [2, 2], top_k=1
        )
        first_batch.sampling_info.original_temperatures[:] = 0
        first_batch.sampling_info.top_ps[:] = 1
        states = [
            {"prompt_len": 1, "step": 1},
            {"prompt_len": 1, "step": 0},
        ]
        with patch(
            "sglang.srt.dllm.algorithm.dream.torch.rand",
            side_effect=[torch.tensor([0.0]), torch.tensor([1.0])],
        ):
            first = dream._run_fdfo_full_prefill(model_runner, first_batch, states)

        self.assertEqual(first[5], [True, False])
        self.assertEqual(first[3][0], None)
        self.assertEqual(first[3][1]["step"], 1)
        self.assertEqual(first[1][1], [20, 99])

        second_batch = self._forward_batch(torch.tensor([20, 99]), [2], top_k=1)
        second_batch.sampling_info.original_temperatures[:] = 0
        second_batch.sampling_info.top_ps[:] = 1
        second = dream._run_fdfo_full_prefill(model_runner, second_batch, [first[3][1]])

        self.assertEqual(second[5], [True])
        self.assertEqual(second[3], [None])
        self.assertEqual(second[1], [[20, 0]])

    def test_dream_fdfo_preserves_cuda_graph_capability(self):
        dream = Dream(_config(algorithm_config={"steps": 1}))
        forward_batch = self._forward_batch(torch.tensor([10, 99]), [2])
        logits_output = SimpleNamespace(full_logits=torch.zeros((2, 5)))
        model_runner = MagicMock()
        model_runner.forward.return_value = SimpleNamespace(
            logits_output=logits_output,
            can_run_graph=True,
        )

        result = dream._run_fdfo_full_prefill(
            model_runner,
            forward_batch,
            [{"prompt_len": 1, "step": 0}],
        )

        self.assertTrue(result[4])

    def test_dream_confidence_algorithms_resolve_final_canvas(self):
        for alg in ("maskgit_plus", "topk_margin", "entropy"):
            with self.subTest(alg=alg):
                dream = Dream(_config(algorithm_config={"steps": 1, "alg": alg}))
                forward_batch = self._forward_batch(
                    torch.tensor([10, 99, 99]), [3], top_k=1
                )
                logits = torch.zeros((3, 5))
                logits[:, 4] = 5.0
                states = [{"prompt_len": 1, "step": 0}]

                done = dream.step(forward_batch, logits, states)

                self.assertEqual(done, [True])
                self.assertEqual(forward_batch.input_ids.tolist(), [10, 4, 4])

    def test_dream_requires_cpu_sequence_lengths(self):
        dream = Dream(_config())
        forward_batch = self._forward_batch(torch.tensor([10, 99]), [2])
        forward_batch.extend_seq_lens_cpu = None

        with self.assertRaisesRegex(RuntimeError, "sequence lengths"):
            dream.step(
                forward_batch, torch.zeros((2, 5)), [{"prompt_len": 1, "step": 0}]
            )


class TestDreamCudaGraphPath(CustomTestCase):
    def _dream_forward_with_hidden_states(
        self,
        hidden_states,
        seq_lens,
        block_size=None,
        canvas_lens=None,
        raw_last_logits=None,
    ):
        def model(*args, **kwargs):
            del args, kwargs
            return hidden_states

        def logits_processor(input_ids, states, lm_head, forward_batch):
            del input_ids, lm_head, forward_batch
            return LogitsProcessorOutput(next_token_logits=None, full_logits=states)

        dream_model = SimpleNamespace(
            capture_aux_hidden_states=False,
            model=model,
            logits_processor=logits_processor,
            lm_head=None,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DLLM_EXTEND,
            extend_seq_lens_cpu=seq_lens,
            dllm_block_size=block_size,
            dllm_canvas_lens_cpu=canvas_lens,
            dllm_raw_last_logits_cpu=raw_last_logits,
        )
        return DreamModel.forward(
            dream_model,
            torch.zeros(hidden_states.shape[0], dtype=torch.long),
            torch.zeros(hidden_states.shape[0], dtype=torch.long),
            forward_batch,
        )

    def test_dream_forward_trims_prefill_bcg_padding_before_split(self):
        hidden_states = torch.arange(6, dtype=torch.float32).view(6, 1)

        output = self._dream_forward_with_hidden_states(hidden_states, [3, 2])

        # The graph bucket has 6 rows, while the live Dream canvas has 5.
        self.assertEqual(output.full_logits.shape, (5, 1))
        self.assertTrue(
            torch.equal(
                output.full_logits[:, 0],
                torch.tensor([0.0, 0.0, 1.0, 3.0, 3.0]),
            )
        )

    def test_dream_forward_rejects_lengths_larger_than_hidden_states(self):
        with self.assertRaisesRegex(RuntimeError, "sequence lengths exceed"):
            self._dream_forward_with_hidden_states(torch.zeros((4, 1)), [5])

    def test_dream_forward_projects_only_generation_blocks(self):
        hidden_states = torch.arange(7, dtype=torch.float32).view(7, 1)

        output = self._dream_forward_with_hidden_states(
            hidden_states, [4, 3], block_size=2
        )

        self.assertEqual(output.full_logits.shape, (4, 1))
        self.assertTrue(
            torch.equal(
                output.full_logits[:, 0],
                torch.tensor([1.0, 2.0, 4.0, 5.0]),
            )
        )

    def test_dream_forward_keeps_short_internal_batches(self):
        hidden_states = torch.arange(2, dtype=torch.float32).view(2, 1)

        output = self._dream_forward_with_hidden_states(
            hidden_states, [2], block_size=32
        )

        self.assertEqual(output.full_logits.shape, (2, 1))
        self.assertTrue(torch.equal(output.full_logits[:, 0], torch.tensor([0.0, 0.0])))

    def test_dream_forward_projects_mixed_request_canvas_lengths(self):
        hidden_states = torch.arange(40, dtype=torch.float32).view(40, 1)

        output = self._dream_forward_with_hidden_states(
            hidden_states,
            [6, 34],
            block_size=32,
            canvas_lens=[4, 32],
        )

        self.assertEqual(output.full_logits.shape, (36, 1))
        self.assertTrue(
            torch.equal(output.full_logits[:4, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )
        self.assertTrue(
            torch.equal(
                output.full_logits[4:, 0], torch.arange(7, 39, dtype=torch.float32)
            )
        )

    def test_dream_forward_partial_prompt_uses_raw_final_hidden_state(self):
        hidden_states = torch.tensor([[10.0], [20.0], [30.0]])

        output = self._dream_forward_with_hidden_states(
            hidden_states,
            [3],
            block_size=32,
            canvas_lens=[1],
            raw_last_logits=[True],
        )

        # Normal Dream shifting would expose 20.0 at the final row. The partial
        # prompt stage must instead project the raw final prompt hidden state.
        self.assertEqual(output.full_logits.shape, (1, 1))
        self.assertEqual(output.full_logits.item(), 30.0)

    def test_dream_forward_rejects_oversized_request_canvas(self):
        with self.assertRaisesRegex(RuntimeError, "fit their request token spans"):
            self._dream_forward_with_hidden_states(
                torch.zeros((3, 1)),
                [3],
                block_size=32,
                canvas_lens=[4],
            )

    def test_prefill_graph_rejects_partial_dream_dynamic_metadata(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)

        self.assertFalse(
            runner.can_run_graph(
                SimpleNamespace(
                    dllm_canvas_lens_cpu=[4],
                    dllm_raw_last_logits_cpu=None,
                )
            )
        )
        self.assertFalse(
            runner.can_run_graph(
                SimpleNamespace(
                    dllm_canvas_lens_cpu=None,
                    dllm_raw_last_logits_cpu=[True],
                )
            )
        )

    def test_prefill_bcg_trims_full_logits_to_raw_tokens(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.raw_bs = 1
        runner.raw_num_tokens = 4
        runner.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_speculative=lambda: False)
        )
        full_logits = torch.arange(12, dtype=torch.float32).view(6, 2)

        output = runner._trim_logits_output(
            LogitsProcessorOutput(next_token_logits=None, full_logits=full_logits)
        )

        self.assertEqual(output.full_logits.shape, (4, 2))
        self.assertTrue(torch.equal(output.full_logits, full_logits[:4]))

    def _make_server_args_for_graph_test(self, disable_cuda_graph):
        args = ServerArgs.__new__(ServerArgs)
        args.dllm_algorithm = "Dream"
        args.dllm_fdfo = True
        args.tp_size = 1
        args.pp_size = 1
        args.disable_radix_cache = False
        args.disable_cuda_graph = disable_cuda_graph
        args.cuda_graph_config = SimpleNamespace(
            decode=SimpleNamespace(
                backend=Backend.DISABLED if disable_cuda_graph else Backend.FULL
            ),
            prefill=SimpleNamespace(
                backend=Backend.DISABLED if disable_cuda_graph else Backend.BREAKABLE
            ),
        )
        args.enable_hierarchical_cache = False
        args.enable_lmcache = False
        args.enable_flexkv = False
        args.enable_lora = False
        args.disaggregation_mode = "null"
        args.enable_mixed_chunk = False
        args.get_model_config = MagicMock(
            return_value=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DreamModel"])
            )
        )
        return args

    @patch("sglang.srt.arg_groups.overrides.run_post_process_pass")
    def test_server_guard_selects_dream_graph_backend(self, run_post_process_pass):
        for disable_cuda_graph in (False, True):
            with self.subTest(disable_cuda_graph=disable_cuda_graph):
                args = self._make_server_args_for_graph_test(disable_cuda_graph)
                with patch("sglang.srt.server_args.is_hip", return_value=False):
                    args._handle_dllm_inference()

                self.assertTrue(args.disable_radix_cache)
                self.assertTrue(args.dllm_fdfo)
                self.assertEqual(
                    args.cuda_graph_config.decode.backend,
                    Backend.DISABLED,
                )
                self.assertEqual(
                    args.cuda_graph_config.prefill.backend,
                    Backend.DISABLED if disable_cuda_graph else Backend.BREAKABLE,
                )
                self.assertEqual(run_post_process_pass.call_count, 3)
                run_post_process_pass.reset_mock()


class _OneStepAlgorithm(DllmAlgorithm):
    def __init__(self):
        super().__init__(
            DllmConfig(
                algorithm="stub",
                algorithm_config={},
                block_size=4,
                mask_id=99,
                max_running_requests=2,
            )
        )
        self.seen_states = None

    def init_step_state(self, forward_batch):
        return [{"prompt_len": 1} for _ in range(forward_batch.batch_size)]

    def max_steps(self, block_size):
        return 1

    def step(self, forward_batch, full_logits, states):
        self.seen_states = states
        return [True] * forward_batch.batch_size


class TestDllmAlgorithmSyncRun(CustomTestCase):
    def test_sync_run_splits_variable_length_requests(self):
        algorithm = _OneStepAlgorithm()
        forward_batch = SimpleNamespace(
            batch_size=2,
            input_ids=torch.tensor([10, 99, 99, 20, 99]),
            extend_seq_lens_cpu=[3, 2],
        )
        logits_output = SimpleNamespace(full_logits=torch.zeros((5, 4)))
        model_runner = MagicMock()
        model_runner.forward.return_value = SimpleNamespace(
            logits_output=logits_output,
            can_run_graph=False,
        )

        result = algorithm.run(
            model_runner,
            forward_batch,
            algo_states=[{"prompt_len": 1, "round": 7}, None],
        )

        self.assertEqual([tokens.tolist() for tokens in result[1]], [[99, 99], [99]])
        self.assertEqual(algorithm.seen_states[0]["round"], 7)
        model_runner.forward.assert_called_once()

    def test_sync_run_requires_sequence_lengths(self):
        algorithm = _OneStepAlgorithm()
        forward_batch = SimpleNamespace(
            batch_size=1,
            input_ids=torch.tensor([10, 99, 99, 99]),
            extend_seq_lens_cpu=None,
        )

        with self.assertRaisesRegex(RuntimeError, "sequence lengths"):
            algorithm.run(MagicMock(), forward_batch)


class TestDreamConfig(CustomTestCase):
    def _server_args(self, *, max_running_requests=None, fdfo=True):
        return SimpleNamespace(
            dllm_algorithm="Dream",
            dllm_algorithm_config=None,
            dllm_fdfo=fdfo,
            max_running_requests=max_running_requests,
            model_path="dream",
            revision=None,
        )

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_dream_uses_full_prefill_and_scheduler_capacity(self, from_server_args):
        from_server_args.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DreamModel"])
        )

        config = DllmConfig.from_server_args(
            self._server_args(max_running_requests=4, fdfo=True)
        )

        self.assertTrue(config.needs_full_prefill)
        self.assertIsNone(config.block_size)
        self.assertEqual(config.mask_id, 151666)
        self.assertEqual(config.max_running_requests, 4)
        self.assertTrue(config.first_done_first_out_mode)

        no_fdfo_config = DllmConfig.from_server_args(
            self._server_args(max_running_requests=4, fdfo=False)
        )
        self.assertFalse(no_fdfo_config.first_done_first_out_mode)

    @patch("sglang.srt.dllm.config.ModelConfig.from_server_args")
    def test_block_dllm_defaults_remain_unchanged(self, from_server_args):
        from_server_args.return_value = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LLaDA2MoeModelLM"])
        )

        config = DllmConfig.from_server_args(self._server_args(fdfo=True))

        self.assertFalse(config.needs_full_prefill)
        self.assertEqual(config.block_size, 32)
        self.assertEqual(config.mask_id, 156895)
        self.assertEqual(config.max_running_requests, 1)
        self.assertTrue(config.first_done_first_out_mode)


class TestDreamSchedulerAdmission(CustomTestCase):
    def _scheduler(self, config, max_running_requests=4):
        scheduler = SimpleNamespace(
            dllm_config=config,
            dllm_manager=DllmManager(config),
            max_running_requests=max_running_requests,
            waiting_queue=[],
        )
        scheduler._fetch_waiting_reqs = SchedulerDllmMixin._fetch_waiting_reqs.__get__(
            scheduler
        )
        return scheduler

    def test_dream_fetch_uses_resolved_scheduler_capacity(self):
        scheduler = self._scheduler(_config(max_running_requests=1))
        requests = [SimpleNamespace(rid=f"req-{i}") for i in range(4)]
        scheduler.waiting_queue = requests

        scheduler._fetch_waiting_reqs()

        self.assertEqual(scheduler.dllm_manager.waiting_queue, requests)
        self.assertEqual(scheduler.waiting_queue, [])

    def test_block_dllm_keeps_configured_dllm_capacity(self):
        config = DllmConfig(
            algorithm="LLaDA",
            algorithm_config={},
            block_size=4,
            mask_id=156895,
            max_running_requests=1,
        )
        scheduler = self._scheduler(config)
        scheduler.waiting_queue = [SimpleNamespace(rid=f"req-{i}") for i in range(4)]

        scheduler._fetch_waiting_reqs()

        self.assertEqual(len(scheduler.dllm_manager.waiting_queue), 1)
        self.assertEqual(len(scheduler.waiting_queue), 3)

    def test_partial_prompt_incoming_admission_preserves_prompt_boundary(self):
        config = _config(
            algorithm="PrefillingDream",
            algorithm_config={"threshold": 0.9, "dual_cache": True},
            block_size=32,
            first_done_first_out_mode=True,
        )
        req = Req(
            rid="partial-admission",
            origin_input_text="prompt",
            origin_input_ids=array("q", [10, 11]),
            sampling_params=SamplingParams(
                max_new_tokens=4,
                custom_params={"dllm_partial_draft": {"rounds": 1}},
            ),
            dllm_config=config,
        )
        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_PREFILL)

        adder = SimpleNamespace(
            can_run_list=[],
            add_one_req=MagicMock(return_value=AddReqResult.CONTINUE),
        )
        running_batch = SimpleNamespace(reqs=[], batch_is_full=False)
        scheduler = SimpleNamespace(
            dllm_config=config,
            tree_cache=MagicMock(),
            truncation_align_size=1,
            enable_priority_preemption=False,
            server_args=SimpleNamespace(),
            get_num_allocatable_reqs=lambda _running_bs: 4,
        )

        result = SchedulerDllmMixin.process_dllm_incoming_reqs(
            scheduler, adder, [req], running_batch
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 11])
        self.assertEqual(req.dllm_algo_state["prompt_len"], 2)
        adder.add_one_req.assert_called_once_with(
            req, has_chunked_req=True, truncation_align_size=1
        )


class TestDreamFDFOResultProcessing(CustomTestCase):
    def test_unresolved_canvas_and_state_survive_scheduler_round(self):
        config = _config(
            algorithm_config={"steps": 3},
            first_done_first_out_mode=True,
        )
        sampling_params = SamplingParams(max_new_tokens=3)
        sampling_params.normalize(None)
        req = Req(
            rid="req",
            origin_input_text="prompt",
            origin_input_ids=array("q", [10]),
            sampling_params=sampling_params,
            dllm_config=config,
        )
        req.init_next_round_input()

        scheduler = SimpleNamespace(
            dllm_config=config,
            metrics_reporter=SimpleNamespace(
                num_generated_tokens=0,
                report_prefill_stats=MagicMock(),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=MagicMock(),
                free_group_end=MagicMock(),
            ),
            tree_cache=MagicMock(),
            output_streamer=SimpleNamespace(stream_output=MagicMock()),
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )

        unresolved = SimpleNamespace(
            copy_done=None,
            accept_length_per_req_cpu=None,
            dllm_done_per_req_cpu=[False],
            dllm_algo_state=[{"prompt_len": 1, "step": 1}],
            next_token_ids=[[10, 20, 99, 99]],
            can_run_cuda_graph=False,
        )
        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release:
            SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, unresolved)
        release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)

        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 20, 99, 99])
        self.assertEqual(req.dllm_algo_state, {"prompt_len": 1, "step": 1})
        self.assertEqual(list(req.output_ids), [])

        req.init_next_round_input()
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 20, 99, 99])

        done = SimpleNamespace(
            copy_done=None,
            accept_length_per_req_cpu=None,
            dllm_done_per_req_cpu=[True],
            dllm_algo_state=[{"prompt_len": 1, "step": 3}],
            next_token_ids=[[10, 20, 30, 40]],
            can_run_cuda_graph=False,
        )
        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache"):
            SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, done)

        self.assertEqual(list(req.output_ids), [20, 30, 40])
        self.assertTrue(req.finished())
        self.assertIsNone(req.dllm_algo_state)

    def test_partial_draft_stages_clean_prompt_then_mask_suffix_kv(self):
        config = _config(
            algorithm="PrefillingDream",
            algorithm_config={"threshold": 0.9, "dual_cache": True},
            block_size=32,
            first_done_first_out_mode=True,
        )
        sampling_params = SamplingParams(
            max_new_tokens=4,
            ignore_eos=True,
            custom_params={"dllm_partial_draft": {"rounds": 1}},
        )
        sampling_params.normalize(None)
        req = Req(
            rid="staged-partial",
            origin_input_text="prompt",
            origin_input_ids=array("q", [10, 11]),
            sampling_params=sampling_params,
            dllm_config=config,
        )
        req.init_next_round_input()
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 11])

        scheduler = SimpleNamespace(
            dllm_config=config,
            metrics_reporter=SimpleNamespace(
                num_generated_tokens=0,
                report_prefill_stats=MagicMock(),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=MagicMock(),
                free_group_end=MagicMock(),
            ),
            tree_cache=MagicMock(),
            output_streamer=SimpleNamespace(stream_output=MagicMock()),
        )
        batch_base = dict(
            batch_size=lambda: 1,
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )

        prompt_state = req.dllm_algo_state
        prompt_state.update(
            {
                "partial_draft_first_token": 7,
                "partial_draft_stage": "suffix_init",
                "last_partial_draft_stage": "prompt",
                "last_round_was_dual_cache": False,
            }
        )
        prompt_batch = SimpleNamespace(
            **batch_base,
            extend_lens=[2],
            out_cache_loc=torch.tensor([101, 102]),
        )
        prompt_result = SimpleNamespace(
            copy_done=None,
            accept_length_per_req_cpu=None,
            dllm_done_per_req_cpu=[False],
            dllm_algo_state=[prompt_state],
            next_token_ids=[[10, 11]],
            can_run_cuda_graph=False,
        )

        SchedulerDllmMixin.process_batch_result_dllm(
            scheduler, prompt_batch, prompt_result
        )

        self.assertEqual(req.prefix_indices.tolist(), [101, 102])
        self.assertEqual(req.dllm_algo_state["partial_draft_stage"], "suffix_init")
        self.assertFalse(req.dllm_incomplete_ids)
        self.assertEqual(list(req.output_ids), [])

        # DllmManager.init_next_round() rebuilds only the suffix while retaining
        # the clean prompt KV prefix.
        req.init_next_round_input()
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 11, 99, 99, 99, 99])
        self.assertEqual(req.prefix_indices.tolist(), [101, 102])

        suffix_state = req.dllm_algo_state
        suffix_state.update(
            {
                "partial_draft_stage": "partial_round",
                "last_partial_draft_stage": "suffix_init",
                "dual_cache_ready": True,
                "is_prefill": False,
                "partial_draft_confirmed_mask": [True, False, False, False],
            }
        )
        suffix_batch = SimpleNamespace(
            **batch_base,
            extend_lens=[4],
            out_cache_loc=torch.tensor([201, 202, 203, 204]),
        )
        suffix_result = SimpleNamespace(
            copy_done=None,
            accept_length_per_req_cpu=None,
            dllm_done_per_req_cpu=[False],
            dllm_algo_state=[suffix_state],
            next_token_ids=[[7, 99, 99, 99]],
            can_run_cuda_graph=False,
        )

        SchedulerDllmMixin.process_batch_result_dllm(
            scheduler, suffix_batch, suffix_result
        )

        self.assertEqual(req.prefix_indices.tolist(), [101, 102])
        self.assertEqual(req.dllm_kv_indices.tolist(), [201, 202, 203, 204])
        self.assertEqual(list(req.dllm_incomplete_ids), [7, 99, 99, 99])
        self.assertEqual(list(req.full_untruncated_fill_ids), [10, 11, 7, 99, 99, 99])
        self.assertEqual(list(req.output_ids), [])

    def test_partial_draft_returns_confirmed_bitmap_after_dual_cache_round(self):
        config = _config(
            algorithm="PrefillingDream",
            algorithm_config={"threshold": 0.9, "dual_cache": True},
            block_size=32,
            first_done_first_out_mode=True,
        )
        sampling_params = SamplingParams(
            max_new_tokens=4,
            ignore_eos=True,
            custom_params={"dllm_partial_draft": {"rounds": 1}},
        )
        sampling_params.normalize(None)
        req = Req(
            rid="partial",
            origin_input_text="prompt",
            origin_input_ids=array("q", [10]),
            sampling_params=sampling_params,
            dllm_config=config,
        )
        req.init_next_round_input()
        req.dllm_algo_state["dual_cache_ready"] = True
        req.dllm_algo_state["last_round_was_dual_cache"] = True
        req.dllm_algo_state["partial_draft_confirmed_mask"] = [
            True,
            False,
            True,
            False,
        ]

        scheduler = SimpleNamespace(
            dllm_config=config,
            metrics_reporter=SimpleNamespace(
                num_generated_tokens=0,
                report_prefill_stats=MagicMock(),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=MagicMock(),
                free_group_end=MagicMock(),
            ),
            tree_cache=MagicMock(),
            output_streamer=SimpleNamespace(stream_output=MagicMock()),
        )
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            reqs=[req],
            return_logprob=False,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        done = SimpleNamespace(
            copy_done=None,
            accept_length_per_req_cpu=None,
            dllm_done_per_req_cpu=[True],
            dllm_algo_state=[None],
            # Slot 0 deliberately equals mask_id even though the algorithm
            # confirmed it. The explicit bitmap, not token equality, is the
            # source of truth.
            next_token_ids=[[99, 99, 30, 99]],
            can_run_cuda_graph=False,
        )

        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release:
            SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, done)

        self.assertEqual(list(req.output_ids), [99, 99, 30, 99])
        self.assertEqual(
            req.customized_info["dllm_confirmed_mask"],
            [True, False, True, False],
        )
        self.assertTrue(req.finished())
        self.assertIsNone(req.dllm_algo_state)
        release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)


if __name__ == "__main__":
    unittest.main()
