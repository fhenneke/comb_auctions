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


def compute_reference_solutions(
    solutions: list[Solution], only_baseline: bool = True
) -> dict[tuple[str, str], tuple[Solution, Solution | None]]:
    reference_solutions: dict[tuple[str, str], tuple[Solution, Solution | None]] = {}
    for solution in solutions:
        if only_baseline and len(solution.scores) > 1:
            continue
        for token_pair, score in solution.scores.items():
            baseline_1, baseline_2 = reference_solutions.get(token_pair, (None, None))
            baseline_score_1 = 0
            baseline_score_2 = 0
            if baseline_1 is not None and token_pair in baseline_1.scores:
                baseline_score_1 = baseline_1.scores[token_pair]
            if baseline_2 is not None and token_pair in baseline_2.scores:
                baseline_score_2 = baseline_2.scores[token_pair]
            if score > baseline_score_2:
                if score > baseline_score_1:
                    if baseline_1 is not None and baseline_1.solver == solution.solver:
                        reference_solutions[token_pair] = (solution, baseline_2)
                    else:
                        reference_solutions[token_pair] = (solution, baseline_1)
                elif baseline_1 is not None and baseline_1.solver != solution.solver:
                    # `baseline_1 is not None` helps mypy, and cannot be false since
                    # baseline_score_1 >= score > baseline_score_2 == 0
                    # implies baseline_score_1 > 0 which is incompatible with baseline_1 == None
                    reference_solutions[token_pair] = (baseline_1, solution)

    return reference_solutions


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
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = []
        baseline_solutions = compute_reference_solutions(solutions)
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
    @abstractmethod
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


class TokenPairImprovementReward(RewardMechanism):
    def __init__(
        self, upper_cap: int, lower_cap: int, only_baseline: bool = True
    ) -> None:
        self.upper_cap = upper_cap
        self.lower_cap = lower_cap
        self.only_baseline = only_baseline

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        rewards: dict[str, tuple[int, int]] = {}
        reference_scores = compute_reference_solutions(
            solutions, only_baseline=self.only_baseline
        )
        for winner in winners:
            rewards[winner.id] = (0, 0)
            for token_pair, score in winner.scores.items():
                reward, penalty = rewards[winner.id]
                baseline_1, baseline_2 = reference_scores.get(token_pair, (None, None))
                baseline_score_1 = (
                    baseline_1.scores[token_pair] if baseline_1 is not None else 0
                )
                baseline_score_2 = (
                    baseline_2.scores[token_pair] if baseline_2 is not None else 0
                )
                if (
                    baseline_1 is not None and baseline_1.solver == winner.solver
                ):  # reference is by winning solver
                    rewards[winner.id] = (
                        reward + max(min(score - baseline_score_2, self.upper_cap), 0),
                        penalty - max(min(baseline_score_2, self.lower_cap), 0),
                    )
                else:
                    rewards[winner.id] = (
                        reward + max(min(score - baseline_score_1, self.upper_cap), 0),
                        penalty - max(min(baseline_score_1, self.lower_cap), 0),
                    )
        return rewards


class GeneralMechanism(AuctionMechanism):
    def __init__(self, filter, winner_selection, reward_mechanism):
        self.filter = filter
        self.winner_selection = winner_selection
        self.reward_mechanism = reward_mechanism

    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        filtered_solutions = self.filter.filter(solutions)
        winners = self.winner_selection.select_winners(filtered_solutions)
        rewards = self.reward_mechanism.compute_rewards(winners, filtered_solutions)
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
        GeneralMechanism(
            NoFilter(),
            SingleWinner(),
            BatchSecondPriceReward(150, 100),
        ),
        GeneralMechanism(
            NoFilter(),
            MultipleWinners(),
            BatchSecondPriceReward(150, 100),
        ),
        GeneralMechanism(
            BaselineFilter(),
            MultipleWinners(),
            TokenPairImprovementReward(50, 40, True),
        ),
        GeneralMechanism(
            BaselineFilter(),
            MultipleWinners(),
            TokenPairImprovementReward(50, 40, False),
        ),
    ]:
        winners_rewards = mechanism.winners_and_rewards(solutions)
        print(winners_rewards)
