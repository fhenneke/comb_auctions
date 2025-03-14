from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    id: str
    sell_token: str
    buy_token: str
    score: int | None = None


@dataclass(frozen=True)
class Solution:
    id: str
    solver: str
    score: int
    trades: list[Trade]


def get_orders(solutions: list[Solution]):
    return {trade.id for solution in solutions for trade in solution.trades}


def compute_total_score(solutions: list[Solution]) -> int:
    return sum(solution.score for solution in solutions)


def aggregate_scores(solution: Solution) -> dict[tuple[str, str], int]:
    scores: dict[tuple[str, str], int] = {}
    for trade in solution.trades:
        score = trade.score if trade.score is not None else 0
        scores[(trade.sell_token, trade.buy_token)] = (
            scores.get((trade.sell_token, trade.buy_token), 0) + score
        )
    return scores


def compute_baseline_solutions(
    solutions: list[Solution], only_baseline: bool = True
) -> dict[tuple[str, str], tuple[Solution, Solution | None]]:
    reference_solutions: dict[tuple[str, str], tuple[Solution, Solution | None]] = {}
    for solution in solutions:
        aggregated_scores = aggregate_scores(solution)
        if only_baseline and len(aggregated_scores) > 1:
            continue
        for token_pair, score in aggregated_scores.items():
            baseline_1, baseline_2 = reference_solutions.get(token_pair, (None, None))
            baseline_score_1 = 0
            baseline_score_2 = 0
            if baseline_1 is not None:
                baseline_scores_1 = aggregate_scores(baseline_1)
                baseline_score_1 = baseline_scores_1.get(token_pair, 0)
            if baseline_2 is not None:
                baseline_scores_2 = aggregate_scores(baseline_2)
                baseline_score_2 = baseline_scores_2.get(token_pair, 0)
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


class SolutionFilter(ABC):
    @abstractmethod
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        """Filter solutions"""


class NoFilter(SolutionFilter):
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        return list(solutions)


class DirectedTokenPairOverlapFilter(SolutionFilter):
    def __init__(self, solution: Solution):
        self.aggregated_scores = aggregate_scores(solution)

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = [
            solution
            for solution in solutions
            if len(aggregate_scores(solution).keys() & self.aggregated_scores.keys())
            == 0
        ]
        return filtered_solutions


@dataclass(frozen=True)
class SolverFilter(SolutionFilter):
    solver: str

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = [
            solution for solution in solutions if solution.solver != self.solver
        ]
        return filtered_solutions


@dataclass(frozen=True)
class SolverFilterBatches(SolutionFilter):
    solver: str

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = [
            solution
            for solution in solutions
            if solution.solver != self.solver or len(aggregate_scores(solution)) == 1
        ]
        return filtered_solutions


@dataclass(frozen=True)
class BaselineFilter(SolutionFilter):
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = []
        baseline_solutions = compute_baseline_solutions(solutions)
        for solution in solutions:
            aggregated_scores = aggregate_scores(solution)
            if len(aggregated_scores) == 1 or all(
                score
                >= (
                    sum(
                        (
                            trade.score if trade.score is not None else 0
                            for trade in baseline_solutions[token_pair][0].trades
                        ),
                        0,
                    )
                    if token_pair in baseline_solutions
                    else 0
                )
                for token_pair, score in aggregated_scores.items()
            ):
                filtered_solutions.append(solution)
        return filtered_solutions


class FilterProperty(ABC):
    @abstractmethod
    def get_filter_set(self, solution: Solution) -> set:
        pass


class DirectedTokenPairs(FilterProperty):
    def get_filter_set(self, solution: Solution) -> set:
        return {(trade.sell_token, trade.buy_token) for trade in solution.trades}


class TokenPairs(FilterProperty):
    def get_filter_set(self, solution: Solution) -> set:
        return {
            frozenset((trade.sell_token, trade.buy_token)) for trade in solution.trades
        }


class TradedTokens(FilterProperty):
    def get_filter_set(self, solution: Solution) -> set:
        sell_tokens = {trade.sell_token for trade in solution.trades}
        buy_tokens = {trade.buy_token for trade in solution.trades}
        return sell_tokens.union(buy_tokens)


class SolutionSelection(ABC):
    @abstractmethod
    def select_solutions(self, solutions: list[Solution]) -> list[Solution]:
        """Select solutions from a list of solutions.

        Solutions selected should be executable at the same time.
        """


class SingleSurplusSelection(SolutionSelection):
    def select_solutions(self, solutions: list[Solution]) -> list[Solution]:
        return [sorted(solutions, key=lambda solution: solution.score)[-1]]


@dataclass(frozen=True)
class SubsetFilteringSelection(SolutionSelection):
    cumulative_filtering: bool = True
    filtering_function: FilterProperty = TradedTokens()

    def select_solutions(self, solutions: list[Solution]) -> list[Solution]:
        sorted_solutions = sorted(
            solutions, key=lambda _solution: _solution.score, reverse=True
        )
        selection: list[Solution] = []
        filter_set: set[str] = set()
        for solution in sorted_solutions:
            solution_filter_set = self.filtering_function.get_filter_set(solution)
            if len(solution_filter_set & filter_set) == 0:
                selection.append(solution)
                if (
                    not self.cumulative_filtering
                ):  # if not cumulative, only filter for selection
                    filter_set = filter_set.union(solution_filter_set)
            if self.cumulative_filtering:  # if cumulative, always filter
                filter_set = filter_set.union(solution_filter_set)

        return selection


class WinnerSelection(ABC):
    @abstractmethod
    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        """Select winners"""


@dataclass(frozen=True)
class DirectSelection(WinnerSelection):
    selection_rule: SolutionSelection

    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        return self.selection_rule.select_solutions(solutions)


@dataclass(frozen=True)
class MonotoneSelection(WinnerSelection):
    selection_rule: SolutionSelection

    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        return self.winners_and_reference_scores(solutions)[0]

    def winners_and_reference_scores(
        self, solutions: list[Solution]
    ) -> tuple[list[Solution], dict[str, int]]:
        selection = self.selection_rule.select_solutions(solutions)
        score = compute_total_score(selection)
        solvers = list({solution.solver for solution in selection})
        reference_scores: dict[str, int] = {}
        for solver in solvers:
            filtered_solutions = SolverFilter(solver).filter(solutions)
            filtered_selection = self.selection_rule.select_solutions(
                filtered_solutions
            )
            filtered_score = compute_total_score(filtered_selection)
            reference_scores[solver] = filtered_score
        solver_max = max(reference_scores, key=reference_scores.get)

        if reference_scores[solver_max] > score:
            return self.winners_and_reference_scores(
                # SolverFilterBatches(solver_max).filter(solutions)
                SolverFilter(solver_max).filter(solutions)
            )

        return selection, reference_scores


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


@dataclass(frozen=True)
class ReferenceReward(RewardMechanism):
    winner_selection: WinnerSelection
    upper_cap: int
    lower_cap: int

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        rewards: dict[str, tuple[int, int]] = {}
        winning_solvers = {winner.solver for winner in winners}
        score = compute_total_score(winners)
        for solver in winning_solvers:
            filtered_solutions = SolverFilter(solver).filter(solutions)
            if filtered_solutions:
                reference_winners = self.winner_selection.select_winners(
                    filtered_solutions
                )
                reference_score = min(score, compute_total_score(reference_winners))
            else:
                reference_score = 0
            rewards[solver] = (
                min(score - reference_score, self.upper_cap),
                -min(reference_score, self.lower_cap),
            )

        return rewards


@dataclass(frozen=True)
class BatchOverlapSecondPriceReward(RewardMechanism):
    upper_cap: int
    lower_cap: int
    filtering_function: FilterProperty = TradedTokens()

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        rewards: dict[str, tuple[int, int]] = {}
        remaining_winners = list(winners)
        sorted_solutions = sorted(
            solutions, key=lambda _solution: _solution.score, reverse=True
        )
        for solution in sorted_solutions:
            if solution not in winners:
                continue
            winner_filter_set = self.filtering_function.get_filter_set(solution)
            reference_score = max(
                (
                    reference_solution.score
                    for reference_solution in solutions
                    if reference_solution.solver != solution.solver
                    and reference_solution.score <= solution.score
                    and len(
                        winner_filter_set
                        & self.filtering_function.get_filter_set(reference_solution)
                    )
                    != 0
                ),
                default=0,
            )

            rewards[solution.id] = (
                min(solution.score - reference_score, self.upper_cap),
                -min(reference_score, self.lower_cap),
            )

        return rewards


@dataclass(frozen=True)
class TokenPairImprovementReward(RewardMechanism):
    upper_cap: int
    lower_cap: int
    only_baseline: bool

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, tuple[int, int]]:
        rewards: dict[str, tuple[int, int]] = {}
        reference_scores = compute_baseline_solutions(
            solutions, only_baseline=self.only_baseline
        )
        for winner in winners:
            rewards[winner.id] = (0, 0)
            aggregated_scores = aggregate_scores(winner)
            for token_pair, score in aggregated_scores.items():
                reward, penalty = rewards[winner.id]
                baseline_1, baseline_2 = reference_scores.get(token_pair, (None, None))
                baseline_score_1 = (
                    aggregate_scores(baseline_1)[token_pair]
                    if baseline_1 is not None
                    else 0
                )
                baseline_score_2 = (
                    aggregate_scores(baseline_2)[token_pair]
                    if baseline_2 is not None
                    else 0
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


class AuctionMechanism(ABC):
    @abstractmethod
    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> tuple[list[Solution], dict[str, tuple[int, int]]]:
        """Select winners and compute their rewards"""


@dataclass(frozen=True)
class FilterRankRewardMechanism(AuctionMechanism):
    solution_filter: SolutionFilter
    winner_selection: WinnerSelection
    reward_mechanism: RewardMechanism

    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> tuple[list[Solution], dict[str, tuple[int, int]]]:
        filtered_solutions = self.solution_filter.filter(solutions)
        winners = self.winner_selection.select_winners(filtered_solutions)
        rewards = self.reward_mechanism.compute_rewards(winners, filtered_solutions)
        return winners, rewards
