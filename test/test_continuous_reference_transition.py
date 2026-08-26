from __future__ import annotations

import numpy as np

from planning.continuous_reference_transition import (
    CRITICAL_2_PERCENT_OMEGA_T,
    CRTConfig,
    ContinuousReferenceTransition,
)


def test_omega_mapping_is_two_percent_critical_settling() -> None:
    value = (1.0 + CRITICAL_2_PERCENT_OMEGA_T) * np.exp(
        -CRITICAL_2_PERCENT_OMEGA_T
    )
    assert np.isclose(value, 0.02, rtol=0.0, atol=1.0e-14)


def test_command_update_preserves_reference_position_and_velocity() -> None:
    transition = ContinuousReferenceTransition(CRTConfig(0.1, 0.3), [1.0, 2.0, 3.0])
    transition.velocity[:] = [0.1, -0.2, 0.3]
    executed_before = transition.executed.copy()
    velocity_before = transition.velocity.copy()
    assert transition.set_command([4.0, 5.0, 6.0])
    assert np.array_equal(transition.executed, executed_before)
    assert np.array_equal(transition.velocity, velocity_before)


def test_constant_command_converges_monotonically_for_one_dimensional_step() -> None:
    transition = ContinuousReferenceTransition(CRTConfig(0.1, 0.5), [0.0, 0.0, 0.0])
    transition.set_command([1.0, 0.0, 0.0])
    values = [float(transition.exact_step().executed_reference[0]) for _ in range(30)]
    assert all(left <= right for left, right in zip(values, values[1:]))
    assert 0.0 < values[0] < 1.0
    assert np.isclose(values[-1], 1.0, rtol=0.0, atol=1.0e-12)


def test_mid_transition_command_change_preserves_state_and_redirects() -> None:
    transition = ContinuousReferenceTransition(CRTConfig(0.1, 0.5), [0.0, 0.0, 0.0])
    transition.set_command([1.0, 0.0, 0.0])
    transition.exact_step()
    executed_before = transition.executed.copy()
    velocity_before = transition.velocity.copy()
    transition.set_command([-1.0, 0.0, 0.0])
    assert np.array_equal(transition.executed, executed_before)
    assert np.array_equal(transition.velocity, velocity_before)
    for _ in range(30):
        transition.exact_step()
    assert np.isclose(transition.executed[0], -1.0, rtol=0.0, atol=1.0e-12)


def test_exact_update_is_finite_and_deterministic() -> None:
    config = CRTConfig(0.1, 0.2)
    left = ContinuousReferenceTransition(config, [2.0, -1.0, 0.5])
    right = ContinuousReferenceTransition(config, [2.0, -1.0, 0.5])
    commands = ([3.0, 4.0, 1.0], [-2.0, 1.5, 0.8], [0.0, 0.0, 0.0])
    for command in commands:
        left.set_command(command)
        right.set_command(command)
        for _ in range(4):
            a = left.exact_step()
            b = right.exact_step()
            assert np.array_equal(a.executed_reference, b.executed_reference)
            assert np.array_equal(
                a.executed_reference_velocity, b.executed_reference_velocity
            )
            assert np.all(np.isfinite(a.executed_reference))
            assert np.all(np.isfinite(a.executed_reference_velocity))


def test_direct_replacement_is_explicit_and_zeroes_filter_velocity() -> None:
    transition = ContinuousReferenceTransition(CRTConfig(0.1, 0.3), [0.0, 0.0, 0.0])
    transition.set_command([1.0, 2.0, 3.0])
    transition.exact_step()
    result = transition.direct_replace_with_command()
    assert np.array_equal(result.executed_reference, transition.command)
    assert np.array_equal(result.executed_reference_velocity, np.zeros(3))
