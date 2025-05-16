"""Tests for the auction mechanism."""

import pytest

from mechanism import (
    Solution,
    Trade,
    FilterRankRewardMechanism,
    BaselineFilter,
    DirectSelection,
    DirectedTokenPairs,
    SubsetFilteringSelection,
    ReferenceReward,
)


@pytest.fixture
def mechanism():
    """Set up the mechanism from CIP-67."""
    reward_cap_upper = 12 * 10**15
    reward_cap_lower = 10 * 10**15
    winner_selection = DirectSelection(
        SubsetFilteringSelection(
            batch_compatibility=DirectedTokenPairs(), cumulative_filtering=False
        )
    )
    yield FilterRankRewardMechanism(
        BaselineFilter(),
        winner_selection,
        ReferenceReward(winner_selection, reward_cap_upper, reward_cap_lower),
    )


def test_single_bid(mechanism):
    """Only one bid submitted results in one winner with reward equal to score"""
    solutions = [
        Solution(
            "best batch",
            "solver 1",
            score=200,
            trades=[Trade("order 1", "A", "B", 100), Trade("order 2", "C", "D", 100)],
        )
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions
    assert winning_score == 200
    assert reference_scores == {"solver 1": 0}
    assert rewards == {"solver 1": 200}


def test_compatible_bids(mechanism):
    """Two compatible batches are both selected as winners"""
    solutions = [
        Solution(
            "best batch",
            "solver 1",
            score=200,
            trades=[Trade("order 1", "A", "B", 100), Trade("order 2", "C", "D", 100)],
        ),
        Solution(
            "compatible batch",
            "solver 2",
            score=100,
            trades=[Trade("order 3", "A", "C", 100)],
        ),
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions
    assert winning_score == 300
    assert reference_scores == {"solver 1": 100, "solver 2": 200}
    assert rewards == {"solver 1": 200, "solver 2": 100}


def test_multiple_solution_for_solver(mechanism):
    """Multiple compatible bids by a single solver are aggregated in rewards"""
    solutions = [
        Solution(
            "best batch",
            "solver 1",
            score=200,
            trades=[Trade("order 1", "A", "B", 100), Trade("order 2", "C", "D", 100)],
        ),
        Solution(
            "compatible batch",
            "solver 1",
            score=100,
            trades=[Trade("order 1", "A", "D", 100)],
        ),
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions
    assert winning_score == 300
    assert reference_scores == {"solver 1": 0}
    assert rewards == {"solver 1": 300}


def test_incompatible_bids(mechanism):
    """Incompatible bid does not win but reduces reward"""
    solutions = [
        Solution(
            "best batch",
            "solver 1",
            score=200,
            trades=[Trade("order 1", "A", "B", 100), Trade("order 2", "C", "D", 100)],
        ),
        Solution(
            "incompatible batch",
            "solver 2",
            score=100,
            trades=[Trade("order 1", "A", "B", 100)],
        ),
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions[0:1]
    assert winning_score == 200
    assert reference_scores == {"solver 1": 100}
    assert rewards == {"solver 1": 100}


def test_fairness_filtering(mechanism):
    """Unfair batch is filtered"""
    solutions = [
        Solution(
            "unfair batch",
            "solver 1",
            score=200,
            trades=[Trade("order 1", "A", "B", 100), Trade("order 2", "C", "D", 100)],
        ),
        Solution(
            "filtering batch",
            "solver 2",
            score=150,
            trades=[Trade("order 1", "A", "B", 150)],
        ),
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions[1:2]
    assert winning_score == 150
    assert reference_scores == {"solver 2": 0}
    assert rewards == {"solver 2": 150}


def test_aggregation_on_token_pair(mechanism):
    """Multiple trades on the same (directed) token pair are aggregated for filtering"""
    solutions = [
        Solution(
            "batch with aggregation",
            "solver 1",
            score=200,
            trades=[Trade("order 1", "A", "B", 100), Trade("order 2", "A", "B", 100)],
        ),
        Solution(
            "incompatible batch",
            "solver 2",
            score=150,
            trades=[Trade("order 1", "A", "B", 150)],
        ),
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions[0:1]
    assert winning_score == 200
    assert reference_scores == {"solver 1": 150}
    assert rewards == {"solver 1": 50}


def test_reference_better_than_winners(mechanism):
    """If reference winners generate more surplus than winners, rewards are set to zero"""
    solutions = [
        Solution(
            "best batch",
            "solver 1",
            score=300,
            trades=[
                Trade("order 1", "A", "B", 100),
                Trade("order 2", "C", "D", 100),
                Trade("order 3", "E", "F", 100),
            ],
        ),
        Solution(
            "incompatible batch 1",
            "solver 2",
            score=280,
            trades=[Trade("order 1", "A", "B", 140), Trade("order 2", "C", "D", 140)],
        ),
        Solution(
            "incompatible batch 2",
            "solver 3",
            score=100,
            trades=[Trade("order 3", "E", "F", 100)],
        ),
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions[0:1]
    assert winning_score == 300
    assert reference_scores == {"solver 1": 380}
    assert rewards == {"solver 1": 0}


def test_cap_from_above(mechanism):
    """The reward is capped from above"""
    solutions = [
        Solution(
            "best batch",
            "solver 1",
            score=13 * 10**15,
            trades=[Trade("order 1", "A", "B", 13 * 10**15)],
        )
    ]
    winners, rewards = mechanism.winners_and_rewards(solutions)
    winning_score = sum(solution.score for solution in winners)
    reference_scores = mechanism.reward_mechanism.compute_reference_scores(
        winners, mechanism.solution_filter.filter(solutions)
    )
    assert winners == solutions
    assert winning_score == 13 * 10**15
    assert reference_scores == {"solver 1": 0}
    assert rewards == {"solver 1": 12 * 10**15}
