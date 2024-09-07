"""This is a python script for computing winners and rewards for combinatorial auctions"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Solution:
    id: str
    solver: str
    scores: dict[tuple[str, str], int]

    def score(self):
        return sum(self.scores.values())


class AuctionMechanism(ABC):
    @abstractmethod
    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        """Compute winners and rewards for all solutions.
        The input is a list of solutions. The output is a dictionary with the ids of winning
        solutions as keys and a tuple of (reward, penalty) as value."""


class AbstractFilter(ABC):
    @abstractmethod
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        """Filter solutions"""


class NoFilter(AbstractFilter):
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        return list(solutions)


class OverlapFilter(AbstractFilter):
    def __init__(self, solution: Solution):
        self._solution = solution

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = [
            solution
            for solution in solutions
            if len(solution.scores.keys() & self._solution.scores.keys()) == 0
        ]
        return filtered_solutions


class SolverFilter(AbstractFilter):
    def __init__(self, solver: str):
        self._solver = solver

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = [
            solution for solution in solutions if solution.solver != self._solver
        ]
        return filtered_solutions


class BaselineFilter(AbstractFilter):
    def compute_baseline_solutions(
        self, solutions: list[Solution]
    ) -> dict[tuple[str, str], tuple[Solution, Solution | None]]:
        """Compute reference solutions per token pair"""
        baseline_solutions: dict[tuple[str, str], tuple[Solution, Solution | None]] = {}
        for solution in solutions:
            if len(solution.scores) == 1:
                token_pair = list(solution.scores.keys())[0]
                score = solution.scores[token_pair]
                baseline_1, baseline_2 = baseline_solutions.get(
                    token_pair, (None, None)
                )
                baseline_score_1 = baseline_1.score() if baseline_1 is not None else 0
                baseline_score_2 = baseline_2.score() if baseline_2 is not None else 0
                if score > baseline_score_2:
                    if score > baseline_score_1:
                        baseline_solutions[token_pair] = (solution, baseline_1)
                    else:
                        assert baseline_1 is not None
                        # this helps mypy, and cannot since
                        # baseline_score_1 >= score > baseline_score_2 == 0
                        # implies baseline_score_1 > 0 which is incompatible with baseline_1 == None
                        baseline_solutions[token_pair] = (baseline_1, solution)

        return baseline_solutions

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = []
        baseline_solutions = self.compute_baseline_solutions(solutions)
        for solution in solutions:
            if len(solution.scores) == 1 or all(
                score
                >= (
                    baseline_solutions[token_pair][0].score()
                    if token_pair in baseline_solutions
                    else 0
                )
                for token_pair, score in solution.scores.items()
            ):
                filtered_solutions.append(solution)
        return filtered_solutions


class WinnerSelection(ABC):
    @abstractmethod
    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        """Select winners"""


class SingleWinner(WinnerSelection):
    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        return [sorted(solutions, key=lambda solution: solution.score())[-1]]


class MultipleWinners(WinnerSelection):
    single_winner_mechanism = SingleWinner()

    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        winners: list[Solution] = []
        remaining_solutions = list(solutions)
        while remaining_solutions:
            new_winner = self.single_winner_mechanism.select_winners(
                remaining_solutions
            )[0]
            winners.append(new_winner)
            remaining_solutions = OverlapFilter(new_winner).filter(remaining_solutions)

        return winners


class RewardMechanism(ABC):
    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        """Compute rewards for all winning solutions"""


class NoReward(RewardMechanism):
    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        return {winner.id: (0, 0) for winner in winners}


class BatchSecondPriceReward(RewardMechanism):
    def __init__(self, upper_cap: int, lower_cap: int) -> None:
        self.upper_cap = upper_cap
        self.lower_cap = lower_cap

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        rewards: dict[str, tuple[int, int]] = {}
        remaining_winners = list(winners)
        remaining_solutions = list(solutions)
        while remaining_winners:
            winner = SingleWinner().select_winners(remaining_winners)[0]
            reference_solutions = SolverFilter(winner.solver).filter(
                remaining_solutions
            )
            reference_score = 0
            if reference_solutions:
                reference_score = (
                    SingleWinner().select_winners(reference_solutions)[0].score()
                )

            rewards[winner.id] = (
                min(winner.score() - reference_score, self.upper_cap),
                -min(reference_score, self.lower_cap),
            )

            remaining_winners.remove(winner)
            remaining_solutions = OverlapFilter(winner).filter(remaining_solutions)

        return rewards


class BaselineImprovementReward(RewardMechanism):
    def __init__(self, upper_cap: int, lower_cap: int) -> None:
        self.upper_cap = upper_cap
        self.lower_cap = lower_cap

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        rewards: dict[str, tuple[int, int]] = {}
        baseline_solutions = BaselineFilter().compute_baseline_solutions(solutions)
        for winner in winners:
            rewards[winner.id] = (0, 0)
            for token_pair, score in winner.scores.items():
                reward, penalty = rewards[winner.id]
                baseline_1, baseline_2 = baseline_solutions.get(
                    token_pair, (None, None)
                )
                baseline_score_1 = baseline_1.score() if baseline_1 is not None else 0
                baseline_score_2 = baseline_2.score() if baseline_2 is not None else 0
                if (
                    baseline_1 is not None and baseline_1.solver == winner.solver
                ):  # reference is by winning solver
                    rewards[winner.id] = (
                        reward + min(score - baseline_score_2, self.upper_cap),
                        penalty - min(baseline_score_2, self.lower_cap),
                    )
                else:
                    rewards[winner.id] = (
                        reward + min(score - baseline_score_1, self.upper_cap),
                        penalty - min(baseline_score_1, self.lower_cap),
                    )
        return rewards


class CIP38(AuctionMechanism):
    winner_selection = SingleWinner()
    reward_mechanism = BatchSecondPriceReward(upper_cap=150, lower_cap=100)

    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        """Select winner by score and reward using second-largest score."""
        winner = self.winner_selection.select_winners(solutions)[0]
        rewards = self.reward_mechanism.compute_rewards([winner], solutions)
        return rewards


class SimpleMultipleWinners(AuctionMechanism):
    winner_selection = MultipleWinners()
    reward_mechanism = BatchSecondPriceReward(upper_cap=150, lower_cap=100)

    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        """Iteratively select winner on mechanism with single winner and filtering."""
        winners = self.winner_selection.select_winners(solutions)
        rewards = self.reward_mechanism.compute_rewards(winners, solutions)
        return rewards


class FairCombinatorialAuction(AuctionMechanism):
    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        """Select winners by filtering on reference solutions
        1) Compute baseline solutions
        2) Choose winners
        3) Compute rewards
        """
        filter = BaselineFilter()
        winner_selection = MultipleWinners()
        reward_mechanism = BaselineImprovementReward(upper_cap=50, lower_cap=40)

        filtered_solutions = filter.filter(solutions)
        winners = winner_selection.select_winners(filtered_solutions)
        rewards = reward_mechanism.compute_rewards(winners, filtered_solutions)

        return rewards


solutions = [
    Solution(
        id="batch winner", solver="solver 1", scores={("A", "B"): 100, ("A", "C"): 50}
    ),
    Solution(
        id="unfair batch", solver="solver 2", scores={("A", "B"): 50, ("A", "C"): 60}
    ),
    Solution(
        id="overlapping batch",
        solver="solver 3",
        scores={("B", "A"): 50, ("A", "C"): 50},
    ),
    Solution(
        id="non-overlapping batch",
        solver="solver 4",
        scores={("B", "A"): 40, ("D", "E"): 50},
    ),
    Solution(
        id="non-overlapping batch unfair",
        solver="solver 5",
        scores={("B", "A"): 20, ("D", "E"): 100},
    ),
    Solution(id="reference A->B", solver="solver 1", scores={("A", "B"): 80}),
    Solution(id="runner up A->B", solver="solver 2", scores={("A", "B"): 40}),
    Solution(id="reference A->C", solver="solver 2", scores={("A", "C"): 40}),
    Solution(id="runner up A->C", solver="solver 1", scores={("A", "C"): 40}),
    Solution(id="reference B->A", solver="solver 7", scores={("B", "A"): 30}),
    Solution(id="reference F->G", solver="solver 8", scores={("F", "G"): 50}),
    Solution(id="runner up F->G", solver="solver 1", scores={("F", "G"): 40}),
    Solution(id="reference H->I", solver="solver 8", scores={("H", "I"): 50}),
]

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    for mechanism in [
        CIP38(),
        SimpleMultipleWinners(),
        FairCombinatorialAuction(),
    ]:
        winners_rewards = mechanism.winners_and_rewards(solutions)
        print(winners_rewards)
