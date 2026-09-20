import pytest
import torch

from mor_mlite.objective import apply_objective, objective_scales


def test_unequal_dp_and_microbatch_means_match_global_token_gradient():
    lm_counts = [[2, 7], [0, 3]]
    candidates = [[[4, 3], [8, 6]], [[1, 1], [5, 2]]]
    scales = objective_scales(lm_counts, candidates)
    weight = torch.tensor(0.75, requires_grad=True)
    observed = 0
    expected_lm, expected_aux = 0, [0, 0]
    for mb in range(2):
        for rank in range(2):
            value = 1 + 2 * mb + rank
            lm = weight.square() * value
            aux = torch.stack([weight * (value + r) for r in range(2)])
            output = {
                "loss": lm + aux.sum(),
                "mor_router_aux_loss": aux.sum(),
                "mor_router_aux_losses": aux,
            }
            observed = observed + apply_objective(output, scales[mb][rank]) / 4
            expected_lm += lm * lm_counts[mb][rank] / 12
            for r, denominator in enumerate((18, 12)):
                expected_aux[r] += aux[r] * candidates[mb][rank][r] / denominator
    expected = expected_lm + sum(expected_aux)
    torch.testing.assert_close(observed, expected)
    grad = torch.autograd.grad(observed, weight, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(expected, weight)[0])


@pytest.mark.parametrize("counts", [[[float("nan")]], [[-1]], [[True]], [[0]]])
def test_objective_rejects_invalid_lm_counts(counts):
    with pytest.raises(ValueError):
        objective_scales(counts, [[[1]]])
