from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from recipe.waste_sd.collect_offline_rollout import (
    DEFAULT_ROLLOUT_INSTRUCTION,
    _load_records,
    _prepend_rollout_instruction,
    _normalize_prompt_messages,
    _prompt_text_for_storage,
)
from recipe.waste_sd.distill_debug import DistillDebugRecorder
from recipe.waste_sd.distill_losses import get_distill_loss_fn
from recipe.waste_sd.agent_loop import build_waste_sd_agent_loop_config
from recipe.waste_sd.fsdp_workers import WasteSDAsyncActorRolloutRefWorker
from recipe.waste_sd.main_waste_sd import _normalize_rollout_agent_num_workers
from recipe.waste_sd.main_teacher_off_policy import apply_teacher_off_policy_defaults
from recipe.waste_sd.offline_rollout_dataset import OfflineTeacherRolloutDataset
from recipe.waste_sd.dp_actor import DataParallelWasteSDDistillActor, validate_distill_objective_config
from recipe.waste_sd.ray_trainer import WasteSDRayTrainer
from recipe.waste_sd.waste_weighting import build_strict_weights
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

try:
    from verl.experimental.agent_loop.single_turn_agent_loop import _trim_accept_lens

    _HAS_AGENT_LOOP_HELPERS = True
except Exception:
    _HAS_AGENT_LOOP_HELPERS = False


class TestWasteWeighting(unittest.TestCase):
    def test_strict_alignment_and_full_block(self):
        weights, stats = build_strict_weights([2, 3], response_valid_len=5, gamma=4, strict=True)
        self.assertEqual(weights.tolist(), [4.0, 3.0, 4.0, 3.0, 2.0])
        self.assertEqual(stats["distill/strict_alignment_ok"], 1.0)

        # Timeout/full block case (length gamma+1): last forced teacher token has zero weight.
        full_block_weights, _ = build_strict_weights([5], response_valid_len=5, gamma=4, strict=True)
        self.assertEqual(full_block_weights.tolist(), [4.0, 3.0, 2.0, 1.0, 0.0])

    def test_full_block_flag_is_noop(self):
        w_true, _ = build_strict_weights([5], response_valid_len=5, gamma=4, strict=True, full_block_participate=True)
        w_false, _ = build_strict_weights([5], response_valid_len=5, gamma=4, strict=True, full_block_participate=False)
        self.assertEqual(w_true.tolist(), w_false.tolist())

    def test_strict_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_strict_weights([2], response_valid_len=5, gamma=4, strict=True)


class TestDistillLosses(unittest.TestCase):
    def test_fkl_rkl_tvd_manual(self):
        # token x vocab
        p = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]], dtype=torch.float32)
        q = torch.tensor([[0.4, 0.4, 0.2], [0.2, 0.2, 0.6]], dtype=torch.float32)
        log_p = torch.log(p)
        log_q = torch.log(q)
        teacher = {"log_probs": log_q}
        weights = torch.tensor([2.0, 1.0], dtype=torch.float32)

        fkl_fn = get_distill_loss_fn("fkl")
        rkl_fn = get_distill_loss_fn("rkl")
        tvd_fn = get_distill_loss_fn("tvd")

        fkl, _, _, _ = fkl_fn(log_p, teacher, weights)
        rkl, _, _, _ = rkl_fn(log_p, teacher, weights)
        tvd, _, _, _ = tvd_fn(log_p, teacher, weights)

        token_fkl = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
        token_rkl = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
        token_tvd = 0.5 * torch.sum(torch.abs(p - q), dim=-1)
        denom = weights.sum()

        expected_fkl = torch.sum(token_fkl * weights) / denom
        expected_rkl = torch.sum(token_rkl * weights) / denom
        expected_tvd = torch.sum(token_tvd * weights) / denom

        self.assertTrue(torch.allclose(fkl, expected_fkl, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(rkl, expected_rkl, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(tvd, expected_tvd, atol=1e-6, rtol=0))

    def test_distill_loss_zero_when_student_equals_teacher(self):
        probs = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
        logits = torch.log(probs)
        teacher = {"log_probs": torch.log(probs)}
        weights = torch.tensor([1.0], dtype=torch.float32)

        for loss_type in ("fkl", "rkl", "tvd"):
            loss_fn = get_distill_loss_fn(loss_type)
            loss, _, _, _ = loss_fn(logits, teacher, weights)
            self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-6, rtol=0))

    def test_teacher_shape_mismatch_raises(self):
        student_logits = torch.randn(2, 5)
        teacher = {"logits": torch.randn(3, 5)}
        weights = torch.tensor([1.0, 1.0], dtype=torch.float32)
        fkl_fn = get_distill_loss_fn("fkl")
        with self.assertRaises(ValueError):
            fkl_fn(student_logits, teacher, weights)

    def test_batched_fkl_matches_single_sample_path(self):
        torch.manual_seed(0)
        student_logits = torch.randn(2, 4, 5)
        teacher_logits = torch.randn(2, 4, 5)
        teacher = {"logits": teacher_logits}
        response_mask = torch.tensor([[True, True, False, True], [True, False, True, True]])
        token_weights = torch.tensor([[1.0, 2.0, 0.0, 3.0], [4.0, 0.0, 5.0, 6.0]], dtype=torch.float32)

        fkl_fn = get_distill_loss_fn("fkl")
        batched_loss, batched_weighted_sum, batched_weight_sum, _ = fkl_fn(
            student_logits,
            teacher,
            token_weights,
            response_mask=response_mask,
        )

        sample_weighted_sum = torch.zeros((), dtype=torch.float32)
        sample_weight_sum = torch.zeros((), dtype=torch.float32)
        for i in range(student_logits.shape[0]):
            mask = response_mask[i]
            if int(mask.sum().item()) == 0:
                continue
            _, weighted_sum, weight_sum, _ = fkl_fn(
                student_logits[i][mask],
                {"logits": teacher_logits[i][mask]},
                token_weights[i][mask],
            )
            sample_weighted_sum = sample_weighted_sum + weighted_sum.detach().cpu()
            sample_weight_sum = sample_weight_sum + weight_sum.detach().cpu()

        expected_loss = sample_weighted_sum / sample_weight_sum
        self.assertTrue(torch.allclose(batched_weighted_sum.detach().cpu(), sample_weighted_sum, atol=1e-5, rtol=0))
        self.assertTrue(torch.allclose(batched_weight_sum.detach().cpu(), sample_weight_sum, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(batched_loss.detach().cpu(), expected_loss, atol=1e-6, rtol=0))

    def test_uniform_mean_micro_aggregation_matches_global_token_mean(self):
        torch.manual_seed(123)
        student_logits = torch.randn(3, 4, 6)
        teacher_logits = torch.randn(3, 4, 6)
        response_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, False],
            ]
        )
        token_weights = response_mask.to(dtype=torch.float32)
        fkl_fn = get_distill_loss_fn("fkl")

        full_loss, _, _, _ = fkl_fn(
            student_logits,
            {"logits": teacher_logits},
            token_weights,
            response_mask=response_mask,
        )

        total_weighted_sum = 0.0
        total_weight_sum = 0.0
        # Simulate two micro-batches with different valid-token counts.
        for idx in (torch.tensor([0, 1]), torch.tensor([2])):
            _, weighted_sum, weight_sum, _ = fkl_fn(
                student_logits[idx],
                {"logits": teacher_logits[idx]},
                token_weights[idx],
                response_mask=response_mask[idx],
            )
            total_weighted_sum += float(weighted_sum.detach().cpu().item())
            total_weight_sum += float(weight_sum.detach().cpu().item())

        micro_aggregated = torch.tensor(total_weighted_sum / total_weight_sum, dtype=torch.float32)
        self.assertTrue(torch.allclose(micro_aggregated, full_loss.detach().cpu(), atol=1e-6, rtol=0))

    def test_teacher_greedy_nll_matches_manual(self):
        student_logits = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0],
                    [0.2, 1.5, -0.3],
                ]
            ],
            dtype=torch.float32,
        )
        teacher_logits = torch.tensor(
            [
                [
                    [-1.0, 3.0, 0.5],
                    [1.4, 0.3, -2.0],
                ]
            ],
            dtype=torch.float32,
        )
        response_mask = torch.tensor([[True, True]])
        token_weights = torch.tensor([[2.0, 1.0]], dtype=torch.float32)

        loss_fn = get_distill_loss_fn("teacher_greedy_nll")
        loss, weighted_sum, weight_sum, metrics = loss_fn(
            student_logits,
            {"logits": teacher_logits},
            token_weights,
            response_mask=response_mask,
        )

        teacher_targets = teacher_logits.argmax(dim=-1)
        student_logp = torch.log_softmax(student_logits, dim=-1)
        token_nll = -torch.gather(student_logp, dim=-1, index=teacher_targets.unsqueeze(-1)).squeeze(-1)
        expected_weighted_sum = torch.sum(token_nll * token_weights)
        expected_weight_sum = token_weights.sum()
        expected_loss = expected_weighted_sum / expected_weight_sum
        expected_match = student_logits.argmax(dim=-1).eq(teacher_targets).to(dtype=torch.float32).mean()

        self.assertTrue(torch.allclose(weighted_sum, expected_weighted_sum, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(weight_sum, expected_weight_sum, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-6, rtol=0))
        self.assertAlmostEqual(metrics["distill/teacher_greedy_nll_token_div_mean"], token_nll.mean().item(), places=6)
        self.assertAlmostEqual(metrics["distill/teacher_greedy_match_rate"], expected_match.item(), places=6)

    def test_teacher_greedy_nll_micro_aggregation_matches_global_token_mean(self):
        torch.manual_seed(321)
        student_logits = torch.randn(3, 4, 6)
        teacher_logits = torch.randn(3, 4, 6)
        response_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
                [True, True, False, False],
            ]
        )
        token_weights = response_mask.to(dtype=torch.float32)
        loss_fn = get_distill_loss_fn("teacher_greedy_nll")

        full_loss, _, _, _ = loss_fn(
            student_logits,
            {"logits": teacher_logits},
            token_weights,
            response_mask=response_mask,
        )

        total_weighted_sum = 0.0
        total_weight_sum = 0.0
        for idx in (torch.tensor([0, 1]), torch.tensor([2])):
            _, weighted_sum, weight_sum, _ = loss_fn(
                student_logits[idx],
                {"logits": teacher_logits[idx]},
                token_weights[idx],
                response_mask=response_mask[idx],
            )
            total_weighted_sum += float(weighted_sum.detach().cpu().item())
            total_weight_sum += float(weight_sum.detach().cpu().item())

        micro_aggregated = torch.tensor(total_weighted_sum / total_weight_sum, dtype=torch.float32)
        self.assertTrue(torch.allclose(micro_aggregated, full_loss.detach().cpu(), atol=1e-6, rtol=0))


class TestDistillObjectiveValidation(unittest.TestCase):
    def test_teacher_greedy_nll_requires_uniform_mean(self):
        with self.assertRaisesRegex(ValueError, "uniform_mean"):
            validate_distill_objective_config(
                loss_type="teacher_greedy_nll",
                weighting_mode="waste",
                kl_floor_coef=0.0,
                rembudget_unweighted_kl_coef=0.0,
                exact_unweighted_kl_coef=0.0,
            )

    def test_teacher_greedy_nll_accepts_uniform_mean(self):
        validate_distill_objective_config(
            loss_type="teacher_greedy_nll",
            weighting_mode="uniform_mean",
            kl_floor_coef=0.0,
            rembudget_unweighted_kl_coef=0.0,
            exact_unweighted_kl_coef=0.0,
        )

    def test_exact_block_count_wnll_requires_uniform_mean(self):
        with self.assertRaisesRegex(ValueError, "uniform_mean"):
            validate_distill_objective_config(
                loss_type="exact_block_count_wnll",
                weighting_mode="remaining_budget_forward",
                kl_floor_coef=0.0,
                rembudget_unweighted_kl_coef=0.0,
                exact_unweighted_kl_coef=0.0,
            )

    def test_exact_block_count_wnll_accepts_uniform_mean(self):
        validate_distill_objective_config(
            loss_type="exact_block_count_wnll",
            weighting_mode="uniform_mean",
            kl_floor_coef=0.0,
            rembudget_unweighted_kl_coef=0.0,
            exact_unweighted_kl_coef=0.0,
        )

    def test_exact_unweighted_kl_coef_requires_exact_loss(self):
        with self.assertRaisesRegex(ValueError, "only applies to"):
            validate_distill_objective_config(
                loss_type="fkl",
                weighting_mode="uniform_mean",
                kl_floor_coef=0.0,
                rembudget_unweighted_kl_coef=0.0,
                exact_unweighted_kl_coef=0.2,
            )

    def test_exact_unweighted_kl_coef_accepts_exact_loss(self):
        validate_distill_objective_config(
            loss_type="exact_block_count_wnll",
            weighting_mode="uniform_mean",
            kl_floor_coef=0.0,
            rembudget_unweighted_kl_coef=0.0,
            exact_unweighted_kl_coef=0.2,
        )

    def test_rembudget_unweighted_kl_coef_requires_remaining_budget_mode(self):
        with self.assertRaisesRegex(ValueError, "only applies to"):
            validate_distill_objective_config(
                loss_type="fkl",
                weighting_mode="uniform_mean",
                kl_floor_coef=0.0,
                rembudget_unweighted_kl_coef=0.2,
                exact_unweighted_kl_coef=0.0,
            )

    def test_rembudget_forward_accepts_tvd(self):
        validate_distill_objective_config(
            loss_type="tvd",
            weighting_mode="remaining_budget_forward",
            kl_floor_coef=0.0,
            rembudget_unweighted_kl_coef=0.0,
            exact_unweighted_kl_coef=0.0,
        )

    def test_rembudget_unweighted_kl_coef_accepts_forward_fkl(self):
        validate_distill_objective_config(
            loss_type="fkl",
            weighting_mode="remaining_budget_forward",
            kl_floor_coef=0.0,
            rembudget_unweighted_kl_coef=0.2,
            exact_unweighted_kl_coef=0.0,
        )

    def test_rembudget_unweighted_kl_coef_accepts_forward_tvd(self):
        validate_distill_objective_config(
            loss_type="tvd",
            weighting_mode="remaining_budget_forward",
            kl_floor_coef=0.0,
            rembudget_unweighted_kl_coef=0.2,
            exact_unweighted_kl_coef=0.0,
        )


class TestPatchDataPathHelpers(unittest.TestCase):
    def test_trim_accept_lens(self):
        if not _HAS_AGENT_LOOP_HELPERS:
            self.skipTest("agent loop helper import unavailable in this environment")
        trimmed = _trim_accept_lens([3, 3, 3], max_tokens=7)
        self.assertEqual(trimmed, [3, 3, 1])


class TestWasteSDAgentLoopConfigRewrite(unittest.TestCase):
    def test_local_ref_rewrites_manager_local_model_identity(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {
                        "path": "actor/model",
                        "tokenizer_path": "actor/tokenizer",
                        "custom_chat_template": "actor-template",
                        "hf_config_path": "actor/model",
                    },
                    "ref": {
                        "model": {
                            "path": "ref/model",
                        }
                    },
                },
                "distill": {
                    "rollout_target": "local_ref",
                },
            }
        )

        rewritten = build_waste_sd_agent_loop_config(config)

        self.assertEqual(rewritten.actor_rollout_ref.model.path, "ref/model")
        self.assertEqual(rewritten.actor_rollout_ref.model.tokenizer_path, "ref/model")
        self.assertEqual(rewritten.actor_rollout_ref.model.hf_config_path, "ref/model")
        self.assertIsNone(rewritten.actor_rollout_ref.model.custom_chat_template)

        # Original training config must remain untouched.
        self.assertEqual(config.actor_rollout_ref.model.path, "actor/model")
        self.assertEqual(config.actor_rollout_ref.model.tokenizer_path, "actor/tokenizer")
        self.assertEqual(config.actor_rollout_ref.model.custom_chat_template, "actor-template")

    def test_actor_target_keeps_actor_model_identity(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {
                        "path": "actor/model",
                        "tokenizer_path": "actor/tokenizer",
                    },
                    "ref": {
                        "model": {
                            "path": "ref/model",
                        }
                    },
                },
                "distill": {
                    "rollout_target": "actor",
                },
            }
        )

        rewritten = build_waste_sd_agent_loop_config(config)

        self.assertEqual(rewritten.actor_rollout_ref.model.path, "actor/model")
        self.assertEqual(rewritten.actor_rollout_ref.model.tokenizer_path, "actor/tokenizer")


class TestTeacherOffPolicyDefaults(unittest.TestCase):
    def test_defaults_preserve_explicit_weighting_override(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"mtp": {"enable": True, "enable_train": True, "enable_rollout": True}},
                    "rollout": {"engine_kwargs": {"sglang": {"speculative_algorithm": "STANDALONE"}}},
                },
                "distill": {
                    "weighting_mode": "remaining_budget_forward",
                    "gamma": 5,
                    "kl_floor_coef": 0.3,
                },
            }
        )


class TestCollectOfflineRolloutHelpers(unittest.TestCase):
    def test_normalize_prompt_messages_handles_gsm8k_object_array(self):
        prompt_value = np.array([{"role": "user", "content": "Solve 1+1."}], dtype=object)
        messages = _normalize_prompt_messages(prompt_value)
        self.assertEqual(messages, [{"role": "user", "content": "Solve 1+1."}])

    def test_prompt_text_for_storage_prefers_user_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Sure"},
            {"role": "user", "content": "Question 2"},
        ]
        self.assertEqual(_prompt_text_for_storage(messages), "Question 1\n\nQuestion 2")

    def test_prepend_rollout_instruction_updates_first_user_message_once(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Solve 2+2."},
            {"role": "assistant", "content": "Let me think."},
            {"role": "user", "content": "Now solve 3+3."},
        ]
        updated = _prepend_rollout_instruction(messages)
        self.assertTrue(updated[1]["content"].startswith(DEFAULT_ROLLOUT_INSTRUCTION))
        self.assertEqual(updated[-1]["content"], "Now solve 3+3.")
        updated_again = _prepend_rollout_instruction(updated)
        self.assertEqual(updated_again[1]["content"].count(DEFAULT_ROLLOUT_INSTRUCTION), 1)

        apply_teacher_off_policy_defaults(config)

        self.assertEqual(config.distill.weighting_mode, "remaining_budget_forward")
        self.assertEqual(config.distill.gamma, 5)
        self.assertEqual(config.distill.kl_floor_coef, 0.3)
        self.assertTrue(config.distill.off_policy_require_ref)
        self.assertEqual(config.distill.rollout_target, "local_ref")
        self.assertNotIn("speculative_algorithm", config.actor_rollout_ref.rollout.engine_kwargs.sglang)

    def test_defaults_set_uniform_mean_when_weighting_mode_absent(self):
        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"mtp": {"enable": True, "enable_train": True, "enable_rollout": True}},
                    "rollout": {"engine_kwargs": {"sglang": {}}},
                },
                "distill": {},
            }
        )

        apply_teacher_off_policy_defaults(config)

        self.assertEqual(config.distill.weighting_mode, "uniform_mean")
        self.assertEqual(config.distill.gamma, 1)
        self.assertEqual(config.distill.kl_floor_coef, 0.0)

    def test_offline_teacher_rollout_sets_default_custom_dataset(self):
        config = OmegaConf.create(
            {
                "data": {},
                "actor_rollout_ref": {
                    "model": {"mtp": {"enable": True, "enable_train": True, "enable_rollout": True}},
                    "rollout": {"engine_kwargs": {"sglang": {}}},
                },
                "distill": {
                    "data_mode": "offline_teacher_rollout",
                },
            }
        )

        apply_teacher_off_policy_defaults(config)

        self.assertEqual(config.data.custom_cls.path, "pkg://recipe.waste_sd.offline_rollout_dataset")
        self.assertEqual(config.data.custom_cls.name, "OfflineTeacherRolloutDataset")

    def test_offline_teacher_rollout_preserves_explicit_custom_dataset(self):
        config = OmegaConf.create(
            {
                "data": {
                    "custom_cls": {
                        "path": "pkg://custom.module",
                        "name": "CustomDataset",
                    }
                },
                "actor_rollout_ref": {
                    "model": {"mtp": {"enable": True, "enable_train": True, "enable_rollout": True}},
                    "rollout": {"engine_kwargs": {"sglang": {}}},
                },
                "distill": {
                    "data_mode": "offline_teacher_rollout",
                },
            }
        )

        apply_teacher_off_policy_defaults(config)

        self.assertEqual(config.data.custom_cls.path, "pkg://custom.module")
        self.assertEqual(config.data.custom_cls.name, "CustomDataset")


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class TestOfflineTeacherRolloutDataset(unittest.TestCase):
    def test_offline_rollout_dataset_uses_stored_token_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "offline_rollout.jsonl"
            sample = {
                "uid": "sample-1",
                "prompt": "Prompt text",
                "response": "Response text",
                "prompt_ids": [11, 12, 13],
                "response_ids": [21, 22],
            }
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            config = OmegaConf.create(
                {
                    "cache_dir": tmp_dir,
                    "shuffle": False,
                    "seed": 123,
                    "max_prompt_length": 6,
                    "max_response_length": 5,
                    "truncation": "error",
                    "use_shm": False,
                }
            )

            dataset = OfflineTeacherRolloutDataset(
                data_files=str(dataset_path),
                tokenizer=_DummyTokenizer(),
                config=config,
            )
            item = dataset[0]

            self.assertEqual(item["uid"], "sample-1")
            self.assertEqual(item["prompts"].tolist(), [0, 0, 0, 11, 12, 13])
            self.assertEqual(item["responses"].tolist(), [21, 22, 0, 0, 0])
            self.assertEqual(item["response_mask"].tolist(), [True, True, False, False, False])
            self.assertEqual(item["attention_mask"].tolist(), [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
            self.assertEqual(item["input_ids"].tolist(), [0, 0, 0, 11, 12, 13, 21, 22, 0, 0, 0])
            self.assertEqual(item["position_ids"].tolist(), [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4])


class TestCollectOfflineRolloutHelpers(unittest.TestCase):
    def test_normalize_prompt_messages_handles_gsm8k_object_array(self):
        prompt_value = np.array(
            [
                {
                    "role": "user",
                    "content": "Janet's ducks lay 16 eggs per day.",
                }
            ],
            dtype=object,
        )

        messages = _normalize_prompt_messages(prompt_value)

        self.assertEqual(
            messages,
            [{"role": "user", "content": "Janet's ducks lay 16 eggs per day."}],
        )

    def test_prompt_text_for_storage_prefers_user_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Prior answer"},
            {"role": "user", "content": "Question 2"},
        ]

        prompt_text = _prompt_text_for_storage(messages)

        self.assertEqual(prompt_text, "Question 1\n\nQuestion 2")

    def test_load_records_respects_rollout_instruction_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "data.jsonl"
            input_path.write_text(json.dumps({"prompt": "What is 2 + 2?"}, ensure_ascii=True) + "\n", encoding="utf-8")

            with_instruction = _load_records(
                str(input_path),
                prompt_key="prompt",
                max_prompts=-1,
                add_rollout_instruction=True,
            )
            without_instruction = _load_records(
                str(input_path),
                prompt_key="prompt",
                max_prompts=-1,
                add_rollout_instruction=False,
            )

            self.assertIn(DEFAULT_ROLLOUT_INSTRUCTION, with_instruction[0]["messages"][0]["content"])
            self.assertNotIn(DEFAULT_ROLLOUT_INSTRUCTION, without_instruction[0]["messages"][0]["content"])


class TestOfflineTeacherRolloutTrainerHelpers(unittest.TestCase):
    def test_finalize_actor_checkpoint_storage_policy_keeps_latest_full(self):
        trainer = WasteSDRayTrainer.__new__(WasteSDRayTrainer)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config = OmegaConf.create({"trainer": {"default_local_dir": tmpdir}})
            older_actor_dir = Path(tmpdir) / "global_step_600" / "actor"
            latest_actor_dir = Path(tmpdir) / "global_step_1200" / "actor"
            older_actor_dir.mkdir(parents=True)
            latest_actor_dir.mkdir(parents=True)

            for actor_dir in (older_actor_dir, latest_actor_dir):
                (actor_dir / "model_world_size_4_rank_0.pt").write_text("model", encoding="utf-8")
                (actor_dir / "optim_world_size_4_rank_0.pt").write_text("optim", encoding="utf-8")
                (actor_dir / "extra_state_world_size_4_rank_0.pt").write_text("extra", encoding="utf-8")

            trainer._finalize_actor_checkpoint_storage_policy(latest_step=1200)

            self.assertTrue((older_actor_dir / "model_world_size_4_rank_0.pt").exists())
            self.assertFalse((older_actor_dir / "optim_world_size_4_rank_0.pt").exists())
            self.assertFalse((older_actor_dir / "extra_state_world_size_4_rank_0.pt").exists())

            self.assertTrue((latest_actor_dir / "model_world_size_4_rank_0.pt").exists())
            self.assertTrue((latest_actor_dir / "optim_world_size_4_rank_0.pt").exists())
            self.assertTrue((latest_actor_dir / "extra_state_world_size_4_rank_0.pt").exists())

            older_metadata = json.loads((older_actor_dir / "checkpoint_contents.json").read_text(encoding="utf-8"))
            latest_metadata = json.loads((latest_actor_dir / "checkpoint_contents.json").read_text(encoding="utf-8"))

            self.assertEqual(older_metadata["load_contents"], ["model"])
            self.assertEqual(older_metadata["save_contents"], ["model"])
            self.assertEqual(latest_metadata["load_contents"], ["model", "optimizer", "extra"])
            self.assertEqual(latest_metadata["save_contents"], ["model", "optimizer", "extra"])

    def test_prepare_offline_teacher_rollout_batch_and_extract_ids(self):
        trainer = WasteSDRayTrainer.__new__(WasteSDRayTrainer)
        trainer.data_mode = "offline_teacher_rollout"
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "n": 1,
                        "temperature": 0.7,
                    }
                }
            }
        )
        trainer.tokenizer = _DummyTokenizer()
        trainer._on_rollout_batch_ready = lambda batch: batch.meta_info.update({"hook_called": True})

        batch = DataProto.from_single_dict(
            {
                "prompts": torch.tensor([[0, 0, 11, 12]], dtype=torch.long),
                "responses": torch.tensor([[21, 22, 0]], dtype=torch.long),
                "input_ids": torch.tensor([[0, 0, 11, 12, 21, 22, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3, 3]], dtype=torch.long),
                "response_mask": torch.tensor([[True, True, False]]),
            }
        )

        prepared = trainer._prepare_offline_teacher_rollout_batch(batch)

        self.assertEqual(prepared.meta_info["temperature"], 0.7)
        self.assertEqual(prepared.meta_info["pad_token_id"], 0)
        self.assertEqual(prepared.meta_info["eos_token_id"], 2)
        self.assertTrue(prepared.meta_info["offline_teacher_rollout"])
        self.assertTrue(prepared.meta_info["hook_called"])

        prompt_ids, response_ids = trainer._extract_prompt_response_ids(prepared)
        self.assertEqual(prompt_ids, [[11, 12]])
        self.assertEqual(response_ids, [[21, 22]])

    def test_offline_mode_skips_rollout_weight_sync(self):
        trainer = WasteSDRayTrainer.__new__(WasteSDRayTrainer)
        trainer.data_mode = "offline_teacher_rollout"
        trainer._rollout_weight_version = 7
        trainer.checkpoint_manager = mock.Mock()

        trainer._initialize_rollout_state()
        self.assertEqual(trainer._rollout_weight_version, 0)
        trainer.checkpoint_manager.update_weights.assert_not_called()

        trainer._maybe_update_rollout_weights(did_actor_update=True, timing_raw={})
        trainer.checkpoint_manager.update_weights.assert_not_called()

    def test_compute_block_metrics_prefers_raw_verify_counters(self):
        trainer = WasteSDRayTrainer.__new__(WasteSDRayTrainer)
        trainer.strict = True

        batch = DataProto.from_single_dict(
            {
                "response_mask": torch.tensor(
                    [
                        [True, True, True],
                        [True, True, False],
                    ]
                )
            }
        )
        batch.non_tensor_batch["spec_verify_ct"] = np.array([3, 2], dtype=object)
        batch.non_tensor_batch["spec_accepted_tokens"] = np.array([7, 4], dtype=object)
        batch.non_tensor_batch["spec_accept_lens"] = np.array([[99], [99]], dtype=object)

        metrics = trainer._compute_block_metrics(batch)

        self.assertEqual(metrics["block_eval/step_traced_samples"], 2.0)
        self.assertEqual(metrics["block_eval/step_total_blocks"], 5.0)
        self.assertEqual(metrics["block_eval/step_total_accepted_tokens"], 11.0)
        self.assertEqual(metrics["block_eval/step_total_response_valid_len"], 5.0)

    def test_compute_block_metrics_falls_back_to_accept_lens(self):
        trainer = WasteSDRayTrainer.__new__(WasteSDRayTrainer)
        trainer.strict = True

        batch = DataProto.from_single_dict(
            {
                "response_mask": torch.tensor(
                    [
                        [True, True, True, True],
                    ]
                )
            }
        )
        batch.non_tensor_batch["spec_accept_lens"] = np.array([[4, 2, 1]], dtype=object)

        metrics = trainer._compute_block_metrics(batch)

        self.assertEqual(metrics["block_eval/step_traced_samples"], 1.0)
        self.assertEqual(metrics["block_eval/step_total_blocks"], 3.0)
        self.assertEqual(metrics["block_eval/step_total_accepted_tokens"], 4.0)

    def test_compute_block_metrics_strict_missing_metadata_raises(self):
        trainer = WasteSDRayTrainer.__new__(WasteSDRayTrainer)
        trainer.strict = True

        batch = DataProto.from_single_dict(
            {
                "response_mask": torch.tensor(
                    [
                        [True, False, False],
                    ]
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "requires per-sample speculative metadata"):
            trainer._compute_block_metrics(batch)


class TestOfflineTeacherRolloutWorker(unittest.TestCase):
    def test_load_checkpoint_uses_metadata_to_switch_to_model_only(self):
        worker = WasteSDAsyncActorRolloutRefWorker.__new__(WasteSDAsyncActorRolloutRefWorker)
        worker._is_actor = True
        worker._is_rollout = False
        worker._is_offload_param = False
        worker._is_offload_optimizer = False
        worker.checkpoint_manager = mock.Mock()
        worker.checkpoint_manager.checkpoint_load_contents = ["model", "optimizer", "extra"]

        with tempfile.TemporaryDirectory() as tmpdir:
            actor_dir = Path(tmpdir) / "actor"
            actor_dir.mkdir()
            (actor_dir / "checkpoint_contents.json").write_text(
                json.dumps({"load_contents": ["model"]}, ensure_ascii=True),
                encoding="utf-8",
            )

            WasteSDAsyncActorRolloutRefWorker.load_checkpoint(worker, str(actor_dir))

        self.assertEqual(worker.checkpoint_manager.checkpoint_load_contents, ["model"])
        worker.checkpoint_manager.load_checkpoint.assert_called_once()

    def test_explicit_checkpoint_load_override_wins_over_metadata(self):
        worker = WasteSDAsyncActorRolloutRefWorker.__new__(WasteSDAsyncActorRolloutRefWorker)
        worker._is_actor = True
        worker._is_rollout = False
        worker._is_offload_param = False
        worker._is_offload_optimizer = False
        worker.checkpoint_manager = mock.Mock()
        worker.checkpoint_manager.checkpoint_load_contents = ["model", "optimizer", "extra"]

        WasteSDAsyncActorRolloutRefWorker.set_checkpoint_load_contents(worker, ["model"])

        with tempfile.TemporaryDirectory() as tmpdir:
            actor_dir = Path(tmpdir) / "actor"
            actor_dir.mkdir()
            (actor_dir / "checkpoint_contents.json").write_text(
                json.dumps({"load_contents": ["model", "optimizer", "extra"]}, ensure_ascii=True),
                encoding="utf-8",
            )

            WasteSDAsyncActorRolloutRefWorker.load_checkpoint(worker, str(actor_dir))

        self.assertEqual(worker.checkpoint_manager.checkpoint_load_contents, ["model"])

    def test_offline_mode_disables_rollout_engine_before_super_init(self):
        worker = WasteSDAsyncActorRolloutRefWorker.__new__(WasteSDAsyncActorRolloutRefWorker)
        worker.config = OmegaConf.create(
            {
                "distill": {
                    "data_mode": "offline_teacher_rollout",
                }
            }
        )
        worker._is_actor = False
        worker._is_rollout = True
        worker._is_ref = False

        observed = {}

        def _fake_super_init(self):
            observed["is_rollout_during_super"] = self._is_rollout

        with mock.patch.object(AsyncActorRolloutRefWorker, "init_model", _fake_super_init):
            WasteSDAsyncActorRolloutRefWorker.init_model(worker)

        self.assertFalse(observed["is_rollout_during_super"])
        self.assertTrue(worker._waste_sd_rollout_disabled)


class TestWasteSDAgentWorkerNormalization(unittest.TestCase):
    def test_normalizes_worker_count_to_largest_compatible_divisor(self):
        config = OmegaConf.create(
            {
                "data": {
                    "train_batch_size": 2,
                    "val_batch_size": 6,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "agent": {
                            "num_workers": 8,
                        }
                    }
                },
            }
        )

        normalized = _normalize_rollout_agent_num_workers(config)

        self.assertEqual(normalized, 2)
        self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 2)

    def test_preserves_worker_count_when_already_compatible(self):
        config = OmegaConf.create(
            {
                "data": {
                    "train_batch_size": 8,
                    "val_batch_size": 4,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "agent": {
                            "num_workers": 4,
                        }
                    }
                },
            }
        )

        normalized = _normalize_rollout_agent_num_workers(config)

        self.assertEqual(normalized, 4)
        self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 4)


class TestDistillDebugRecorder(unittest.TestCase):
    def test_debug_dump_contains_block_trace_and_zero_divergence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = DistillDebugRecorder(
                {
                    "enable": True,
                    "dump_dir": tmp_dir,
                    "max_samples_per_step": 1,
                    "max_tokens_per_sample": 8,
                    "include_topk": False,
                }
            )

            probs = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]], dtype=torch.float32)
            logits = torch.log(probs)
            weights = torch.tensor([4.0, 3.0], dtype=torch.float32)
            response_ids = torch.tensor([101, 102], dtype=torch.long)

            recorder.maybe_record_sample(
                step=3,
                uid="sample-1",
                sample_index=0,
                loss_type="fkl",
                gamma=4,
                strict=True,
                temperature=1.0,
                spec_accept_lens=[2],
                response_ids=response_ids,
                token_weights=weights,
                student_logits=logits,
                teacher_logits=logits.clone(),
            )
            # Max samples per step = 1, second write should be dropped.
            recorder.maybe_record_sample(
                step=3,
                uid="sample-2",
                sample_index=1,
                loss_type="fkl",
                gamma=4,
                strict=True,
                temperature=1.0,
                spec_accept_lens=[2],
                response_ids=response_ids,
                token_weights=weights,
                student_logits=logits,
                teacher_logits=logits.clone(),
            )

            dump_files = list(Path(tmp_dir).glob("step_00000003_*.jsonl"))
            self.assertEqual(len(dump_files), 1)

            lines = dump_files[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])

            self.assertEqual(record["uid"], "sample-1")
            self.assertEqual(record["weights"], [4.0, 3.0])
            self.assertEqual(record["block_trace"]["alignment_ok"], True)
            self.assertEqual(record["block_trace"]["block_index"], [0, 0])
            self.assertEqual(record["block_trace"]["within_block_offset"], [0, 1])
            self.assertTrue(all(abs(x) < 1e-7 for x in record["token_divergence"]))
            self.assertAlmostEqual(record["weighted_sum"], 0.0, places=7)

    def test_debug_dump_supports_teacher_greedy_nll(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = DistillDebugRecorder(
                {
                    "enable": True,
                    "dump_dir": tmp_dir,
                    "max_samples_per_step": 1,
                    "max_tokens_per_sample": 8,
                    "include_topk": False,
                }
            )

            student_logits = torch.tensor(
                [
                    [2.0, 0.0, -1.0],
                    [0.2, 1.5, -0.3],
                ],
                dtype=torch.float32,
            )
            teacher_logits = torch.tensor(
                [
                    [-1.0, 3.0, 0.5],
                    [1.4, 0.3, -2.0],
                ],
                dtype=torch.float32,
            )
            weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

            recorder.maybe_record_sample(
                step=5,
                uid="sample-greedy",
                sample_index=0,
                loss_type="teacher_greedy_nll",
                gamma=1,
                strict=True,
                temperature=1.0,
                spec_accept_lens=[2],
                response_ids=torch.tensor([101, 102], dtype=torch.long),
                token_weights=weights,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
            )

            dump_files = list(Path(tmp_dir).glob("step_00000005_*.jsonl"))
            self.assertEqual(len(dump_files), 1)
            record = json.loads(dump_files[0].read_text(encoding="utf-8").strip())

            teacher_targets = teacher_logits.argmax(dim=-1)
            student_logp = torch.log_softmax(student_logits, dim=-1)
            expected_nll = -torch.gather(student_logp, dim=-1, index=teacher_targets.unsqueeze(-1)).squeeze(-1)

            self.assertEqual(record["loss_type"], "teacher_greedy_nll")
            self.assertEqual(record["block_trace"]["alignment_ok"], True)
            self.assertEqual(record["response_token_ids"], [101, 102])
            self.assertTrue(np.allclose(record["token_divergence"], expected_nll.tolist(), atol=1e-6, rtol=0))

    def test_debug_dump_supports_exact_block_count_wnll(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = DistillDebugRecorder(
                {
                    "enable": True,
                    "dump_dir": tmp_dir,
                    "max_samples_per_step": 1,
                    "max_tokens_per_sample": 8,
                    "include_topk": False,
                }
            )

            student_logits = torch.log(torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float32))
            teacher_logits = torch.log(torch.tensor([[0.8, 0.2], [0.8, 0.2]], dtype=torch.float32))
            response_ids = torch.tensor([0, 0], dtype=torch.long)
            weights = torch.tensor([0.375, 0.125], dtype=torch.float32)

            recorder.maybe_record_sample(
                step=6,
                uid="sample-exact",
                sample_index=0,
                loss_type="exact_block_count_wnll",
                gamma=1,
                strict=True,
                temperature=1.0,
                spec_accept_lens=[1, 1],
                response_ids=response_ids,
                token_weights=weights,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
            )

            dump_files = list(Path(tmp_dir).glob("step_00000006_*.jsonl"))
            self.assertEqual(len(dump_files), 1)
            record = json.loads(dump_files[0].read_text(encoding="utf-8").strip())
            expected_nll = (-torch.log(torch.tensor([0.6, 0.4], dtype=torch.float32))).tolist()

            self.assertEqual(record["loss_type"], "exact_block_count_wnll")
            self.assertEqual(record["response_token_ids"], [0, 0])
            self.assertTrue(np.allclose(record["token_divergence"], expected_nll, atol=1e-6, rtol=0))


class TestForwardRemainingBudgetMetrics(unittest.TestCase):
    def test_masked_sample_mean_ignores_padded_positions(self):
        values = torch.tensor(
            [
                [0.2, 0.4, 0.9, 0.9],
                [0.7, 0.1, 0.8, 0.8],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
            ]
        )

        means = DataParallelWasteSDDistillActor._masked_sample_mean(values, mask)
        expected = torch.tensor(
            [
                (0.2 + 0.4) / 2.0,
                (0.7 + 0.1 + 0.8) / 3.0,
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(means, expected, atol=1e-6, rtol=0))

    def test_weighted_token_mean_matches_weighted_sum_over_weight_sum(self):
        weighted_sum = torch.tensor(12.0, dtype=torch.float32)
        weight_sum = torch.tensor(3.0, dtype=torch.float32)

        objective = DataParallelWasteSDDistillActor._weighted_token_mean(weighted_sum, weight_sum)

        self.assertTrue(torch.allclose(objective, torch.tensor(4.0), atol=1e-6, rtol=0))


class TestRolloutTargetResolution(unittest.TestCase):
    def _resolve(self, cfg: dict):
        worker = WasteSDAsyncActorRolloutRefWorker.__new__(WasteSDAsyncActorRolloutRefWorker)
        worker.config = cfg
        return worker._resolve_rollout_target_model_path()

    def test_rollout_target_actor(self):
        cfg = {
            "model": {"path": "student-path"},
            "ref": {"model": {"path": "teacher-path"}},
            "distill": {"rollout_target": "actor"},
        }
        self.assertEqual(self._resolve(cfg), "student-path")

    def test_rollout_target_local_ref(self):
        cfg = {
            "model": {"path": "student-path"},
            "ref": {"model": {"path": "teacher-path"}},
            "distill": {"rollout_target": "local_ref"},
        }
        self.assertEqual(self._resolve(cfg), "teacher-path")

    def test_rollout_target_local_ref_fallback_actor(self):
        cfg = {
            "model": {"path": "student-path"},
            "distill": {"rollout_target": "local_ref"},
        }
        self.assertEqual(self._resolve(cfg), "student-path")

    def test_rollout_target_invalid(self):
        cfg = {
            "model": {"path": "student-path"},
            "ref": {"model": {"path": "teacher-path"}},
            "distill": {"rollout_target": "bad_value"},
        }
        with self.assertRaises(ValueError):
            self._resolve(cfg)

    def test_require_ref_teacher_raises_without_ref_path(self):
        cfg = {
            "model": {"path": "student-path"},
            "distill": {
                "rollout_target": "actor",
                "off_policy_require_ref": True,
            },
        }
        with self.assertRaises(ValueError):
            self._resolve(cfg)


if __name__ == "__main__":
    unittest.main()
