from __future__ import annotations

import functools
import unittest

import torch

from recipe.waste_sd.block_count_dp import (
    compute_block_count_omega,
    compute_forward_remaining_budget_weights,
)
from recipe.waste_sd.exact_block_count_loss import compute_exact_block_count_wnll_from_logits


def _make_logprobs_from_rejects(rejects: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    teacher = torch.zeros(len(rejects), dtype=torch.float64)
    student = torch.log(torch.tensor([1.0 - r for r in rejects], dtype=torch.float64))
    return teacher, student


def _bruteforce_future_blocks(rejects: list[float], gamma: int, token_idx: int, state: int) -> float:
    if token_idx >= len(rejects):
        return 0.0
    if state == gamma:
        return 1.0 + _bruteforce_future_blocks(rejects, gamma, token_idx + 1, 0)
    reject = rejects[token_idx]
    return reject * (1.0 + _bruteforce_future_blocks(rejects, gamma, token_idx + 1, 0)) + (1.0 - reject) * (
        _bruteforce_future_blocks(rejects, gamma, token_idx + 1, state + 1)
    )


def _bruteforce_state_distribution(rejects: list[float], gamma: int, token_idx: int) -> list[float]:
    @functools.lru_cache(maxsize=None)
    def rec(n: int) -> tuple[float, ...]:
        if n == 0:
            start = [0.0] * (gamma + 1)
            start[0] = 1.0
            return tuple(start)

        prev = rec(n - 1)
        reject = rejects[n - 1]
        cur = [0.0] * (gamma + 1)
        cur[0] = prev[gamma] + reject * sum(prev[:gamma])
        for s in range(gamma):
            cur[s + 1] = (1.0 - reject) * prev[s]
        return tuple(cur)

    return list(rec(token_idx))


class BlockCountDPTest(unittest.TestCase):
    def test_simple_gamma1_example(self):
        gamma = 1
        # r1 = 0.25, r2 = 0.5
        teacher_logp = torch.log(torch.tensor([0.8, 0.8], dtype=torch.float64))
        student_logp = torch.log(torch.tensor([0.6, 0.4], dtype=torch.float64))

        result = compute_block_count_omega(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=gamma,
        )

        self.assertTrue(torch.allclose(result.reject_prob, torch.tensor([0.25, 0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(result.omega, torch.tensor([0.5, 0.25], dtype=torch.float64)))
        expected_alpha = torch.tensor([0.5 * 0.75, 0.25 * 0.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(result.alpha, expected_alpha))

    def test_equal_models_yield_zero_reject_and_alpha(self):
        teacher_logp = torch.log(torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64))
        student_logp = teacher_logp.clone()

        result = compute_block_count_omega(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=3,
        )

        self.assertTrue(torch.allclose(result.reject_prob, torch.zeros_like(result.reject_prob)))
        self.assertTrue(torch.allclose(result.alpha, torch.zeros_like(result.alpha)))
        self.assertTrue(torch.all(result.omega >= 0.0))
        self.assertTrue(torch.all(result.omega <= 1.0))

    def test_forward_and_backward_match_bruteforce_gamma2(self):
        rejects = [0.2, 0.6, 0.3]
        teacher_logp, student_logp = _make_logprobs_from_rejects(rejects)

        result = compute_block_count_omega(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=2,
        )

        expected_u = torch.tensor(
            [_bruteforce_state_distribution(rejects, gamma=2, token_idx=n) for n in range(len(rejects))],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(result.occupancy_u, expected_u, atol=1e-12))

        expected_U = torch.tensor(
            [
                [_bruteforce_future_blocks(rejects, gamma=2, token_idx=n, state=s) for s in range(3)]
                for n in range(len(rejects) + 1)
            ],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(result.future_block_count_U, expected_U, atol=1e-12))

    def test_omega_matches_finite_difference_of_expected_block_count(self):
        rejects = [0.17, 0.41, 0.28, 0.63]
        gamma = 3
        teacher_logp, student_logp = _make_logprobs_from_rejects(rejects)

        result = compute_block_count_omega(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=gamma,
        )

        def total_blocks(cur_rejects: list[float]) -> float:
            return _bruteforce_future_blocks(cur_rejects, gamma=gamma, token_idx=0, state=0)

        finite_diff = []
        eps = 1e-6
        for idx in range(len(rejects)):
            plus = list(rejects)
            minus = list(rejects)
            plus[idx] += eps
            minus[idx] -= eps
            finite_diff.append((total_blocks(plus) - total_blocks(minus)) / (2.0 * eps))

        expected = torch.tensor(finite_diff, dtype=torch.float64)
        self.assertTrue(torch.allclose(result.omega, expected, atol=1e-6, rtol=1e-6))

    def test_batched_masked_inputs_match_separate_runs(self):
        teacher_logp = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        student_logp = torch.log(
            torch.tensor(
                [
                    [0.9, 0.5, 0.8, 0.7],
                    [0.6, 0.4, 0.9, 0.9],
                ],
                dtype=torch.float64,
            )
        )
        response_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )

        batched = compute_block_count_omega(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=2,
            response_mask=response_mask,
        )

        single0 = compute_block_count_omega(
            teacher_logprobs=teacher_logp[0],
            student_logprobs=student_logp[0],
            gamma=2,
        )
        single1 = compute_block_count_omega(
            teacher_logprobs=teacher_logp[1, :2],
            student_logprobs=student_logp[1, :2],
            gamma=2,
        )

        self.assertTrue(torch.allclose(batched.omega[0], single0.omega, atol=1e-12))
        self.assertTrue(torch.allclose(batched.alpha[0], single0.alpha, atol=1e-12))
        self.assertTrue(torch.allclose(batched.omega[1, :2], single1.omega, atol=1e-12))
        self.assertTrue(torch.allclose(batched.alpha[1, :2], single1.alpha, atol=1e-12))
        self.assertTrue(torch.allclose(batched.omega[1, 2:], torch.zeros(2, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(batched.alpha[1, 2:], torch.zeros(2, dtype=torch.float64), atol=1e-12))

    def test_advantage_and_omega_are_unit_bounded(self):
        rejects = [0.11, 0.37, 0.52, 0.23, 0.41]
        teacher_logp, student_logp = _make_logprobs_from_rejects(rejects)

        result = compute_block_count_omega(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=4,
        )

        eps = 1e-12
        self.assertTrue(torch.all(result.advantage_A >= -eps))
        self.assertTrue(torch.all(result.advantage_A <= 1.0 + eps))
        self.assertTrue(torch.all(result.omega >= -eps))
        self.assertTrue(torch.all(result.omega <= 1.0 + eps))
        self.assertTrue(torch.all(result.alpha >= -eps))
        self.assertTrue(torch.all(result.alpha <= 1.0 + eps))


class ForwardRemainingBudgetDPTest(unittest.TestCase):
    def test_gamma1_expected_position_and_weight(self):
        result = compute_forward_remaining_budget_weights(
            teacher_logprobs=torch.log(torch.tensor([0.8, 0.8], dtype=torch.float64)),
            student_logprobs=torch.log(torch.tensor([0.6, 0.4], dtype=torch.float64)),
            gamma=1,
            kl_floor_coef=0.0,
        )

        expected_reject = torch.tensor([0.25, 0.5], dtype=torch.float64)
        expected_u = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float64)
        expected_pos = torch.tensor([0.0, 0.75], dtype=torch.float64)
        expected_weight = torch.tensor([1.0, 0.25], dtype=torch.float64)

        self.assertTrue(torch.allclose(result.reject_prob, expected_reject))
        self.assertTrue(torch.allclose(result.occupancy_u, expected_u))
        self.assertTrue(torch.allclose(result.expected_position, expected_pos))
        self.assertTrue(torch.allclose(result.remaining_budget_weight, expected_weight))
        self.assertTrue(torch.allclose(result.mixed_weight, expected_weight))

    def test_kl_floor_mixes_weights(self):
        result = compute_forward_remaining_budget_weights(
            teacher_logprobs=torch.log(torch.tensor([0.8, 0.8], dtype=torch.float64)),
            student_logprobs=torch.log(torch.tensor([0.6, 0.4], dtype=torch.float64)),
            gamma=1,
            kl_floor_coef=0.2,
        )
        self.assertTrue(torch.allclose(result.mixed_weight, torch.tensor([1.0, 0.4], dtype=torch.float64)))

    def test_batched_masked_inputs_match_separate_runs(self):
        teacher_logp = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        student_logp = torch.log(
            torch.tensor(
                [
                    [0.9, 0.5, 0.8, 0.7],
                    [0.6, 0.4, 0.9, 0.9],
                ],
                dtype=torch.float64,
            )
        )
        response_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
            ]
        )

        batched = compute_forward_remaining_budget_weights(
            teacher_logprobs=teacher_logp,
            student_logprobs=student_logp,
            gamma=2,
            response_mask=response_mask,
            kl_floor_coef=0.0,
        )
        single0 = compute_forward_remaining_budget_weights(
            teacher_logprobs=teacher_logp[0],
            student_logprobs=student_logp[0],
            gamma=2,
            kl_floor_coef=0.0,
        )
        single1 = compute_forward_remaining_budget_weights(
            teacher_logprobs=teacher_logp[1, :2],
            student_logprobs=student_logp[1, :2],
            gamma=2,
            kl_floor_coef=0.0,
        )

        self.assertTrue(
            torch.allclose(batched.remaining_budget_weight[0], single0.remaining_budget_weight, atol=1e-12)
        )
        self.assertTrue(torch.allclose(batched.mixed_weight[0], single0.mixed_weight, atol=1e-12))
        self.assertTrue(
            torch.allclose(batched.remaining_budget_weight[1, :2], single1.remaining_budget_weight, atol=1e-12)
        )
        self.assertTrue(torch.allclose(batched.mixed_weight[1, :2], single1.mixed_weight, atol=1e-12))
        self.assertTrue(
            torch.allclose(
                batched.remaining_budget_weight[1, 2:], torch.zeros(2, dtype=torch.float64), atol=1e-12
            )
        )
        self.assertTrue(torch.allclose(batched.mixed_weight[1, 2:], torch.zeros(2, dtype=torch.float64), atol=1e-12))


class ExactBlockCountWNLLTest(unittest.TestCase):
    def test_matches_manual_gamma1_example(self):
        teacher_logits = torch.log(
            torch.tensor(
                [
                    [[0.8, 0.2], [0.8, 0.2]],
                ],
                dtype=torch.float32,
            )
        )
        student_logits = torch.log(
            torch.tensor(
                [
                    [[0.6, 0.4], [0.4, 0.6]],
                ],
                dtype=torch.float32,
            )
        )
        responses = torch.tensor([[0, 0]], dtype=torch.long)
        response_mask = torch.tensor([[True, True]])

        result = compute_exact_block_count_wnll_from_logits(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            responses=responses,
            gamma=1,
            response_mask=response_mask,
        )

        expected_alpha = torch.tensor([[0.5 * 0.75, 0.25 * 0.5]], dtype=torch.float32)
        expected_token_nll = -torch.log(torch.tensor([[0.6, 0.4]], dtype=torch.float32))
        expected_sample_loss = torch.sum(expected_alpha * expected_token_nll, dim=-1)
        expected_loss = expected_sample_loss / expected_alpha.sum()

        self.assertTrue(torch.allclose(result.alpha, expected_alpha, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(result.token_nll, expected_token_nll, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(result.sample_loss_sum, expected_sample_loss, atol=1e-6, rtol=0))
        self.assertTrue(torch.allclose(result.loss, expected_loss.squeeze(0), atol=1e-6, rtol=0))

    def test_equal_models_give_zero_exact_loss(self):
        probs = torch.tensor(
            [
                [[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]],
            ],
            dtype=torch.float32,
        )
        logits = torch.log(probs)
        responses = torch.tensor([[0, 1]], dtype=torch.long)
        response_mask = torch.tensor([[True, True]])

        result = compute_exact_block_count_wnll_from_logits(
            teacher_logits=logits,
            student_logits=logits.clone().requires_grad_(True),
            responses=responses,
            gamma=2,
            response_mask=response_mask,
        )

        self.assertTrue(torch.allclose(result.alpha, torch.zeros_like(result.alpha), atol=1e-7, rtol=0))
        self.assertTrue(torch.allclose(result.loss, torch.zeros_like(result.loss), atol=1e-7, rtol=0))

    def test_micro_aggregation_matches_full_batch_weighted_mean(self):
        teacher_logits = torch.log(
            torch.tensor(
                [
                    [[0.8, 0.2], [0.8, 0.2], [0.7, 0.3]],
                    [[0.9, 0.1], [0.6, 0.4], [0.5, 0.5]],
                    [[0.75, 0.25], [0.8, 0.2], [0.6, 0.4]],
                ],
                dtype=torch.float32,
            )
        )
        student_logits = torch.log(
            torch.tensor(
                [
                    [[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]],
                    [[0.7, 0.3], [0.55, 0.45], [0.5, 0.5]],
                    [[0.65, 0.35], [0.7, 0.3], [0.55, 0.45]],
                ],
                dtype=torch.float32,
            )
        )
        responses = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=torch.long,
        )
        response_mask = torch.tensor(
            [
                [True, True, True],
                [True, True, False],
                [True, False, False],
            ]
        )

        full = compute_exact_block_count_wnll_from_logits(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            responses=responses,
            gamma=2,
            response_mask=response_mask,
        )

        total_loss_sum = 0.0
        total_weight_sum = 0.0
        for idx in (torch.tensor([0, 1]), torch.tensor([2])):
            part = compute_exact_block_count_wnll_from_logits(
                teacher_logits=teacher_logits[idx],
                student_logits=student_logits[idx],
                responses=responses[idx],
                gamma=2,
                response_mask=response_mask[idx],
            )
            total_loss_sum += float(part.total_loss_sum.item())
            total_weight_sum += float(part.alpha_weight_sum.item())

        micro_aggregated = torch.tensor(total_loss_sum / total_weight_sum, dtype=torch.float32)
        self.assertTrue(torch.allclose(micro_aggregated, full.loss.detach().cpu(), atol=1e-6, rtol=0))


if __name__ == "__main__":
    unittest.main()
