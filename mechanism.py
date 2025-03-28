import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Trade:
    id: str
    sell_token: str
    buy_token: str
    score: int


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
    """Aggregates scores for trades by token pairs in a solution.

    This function processes a given solution containing trades and aggregates the
    scores for each unique token pair (sell_token, buy_token). The result is a
    dictionary where the keys are tuples representing token pairs, and the values
    are the aggregated score for that pair. The function iterates through the
    trades in the solution, summing the scores for trades with the same token pair.

    Parameters
    ----------
    solution : Solution
        An instance of the Solution class, which contains a list of trade objects.

    Returns
    -------
    dict[tuple[str, str], int]
        A dictionary where the keys are tuples of type (str, str), representing the
        token pairs (sell_token, buy_token), and the values are integers
        representing the aggregated score for each pair.

    """
    scores: dict[tuple[str, str], int] = {}
    for trade in solution.trades:
        scores[(trade.sell_token, trade.buy_token)] = (
            scores.get((trade.sell_token, trade.buy_token), 0) + trade.score
        )
    return scores


def compute_baseline_solutions(
    solutions: list[Solution],
) -> dict[tuple[str, str], Solution]:
    """Compute baseline solutions from a list of solutions.

    This function processes a list of `Solution` objects to determine the baseline
    solutions by analyzing their aggregated scores. For each token pair present in
    the aggregated scores, the function compares scores and selects the solution
    with the highest score for each unique token pair.

    Parameters
    ----------
    solutions: list of Solution
        A list of `Solution` objects to be analyzed. Each solution contains
        information, including an associated aggregated scores mapping
        token pairs to scores.

    Returns
    -------
    baseline_solutions: dict[tuple[str, str], Solution]
        A dictionary where keys are token pairs (tuples of two strings)
        and values are the baseline `Solution` objects associated with
        the highest score for each token pair.
    """
    baseline_solutions: dict[tuple[str, str], Solution] = {}
    for solution in solutions:
        aggregated_scores = aggregate_scores(solution)
        if len(aggregated_scores) > 1:
            continue
        for token_pair, score in aggregated_scores.items():
            if (
                token_pair not in baseline_solutions
                or score > baseline_solutions[token_pair].score
            ):
                baseline_solutions[token_pair] = solution

    return baseline_solutions


class SolutionFilter(ABC):
    @abstractmethod
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        """Filter solutions"""


class NoFilter(SolutionFilter):
    def filter(self, solutions: list[Solution]) -> list[Solution]:
        return list(solutions)


@dataclass(frozen=True)
class SolverFilter(SolutionFilter):
    solver: str

    def filter(self, solutions: list[Solution]) -> list[Solution]:
        filtered_solutions = [
            solution for solution in solutions if solution.solver != self.solver
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
                            trade.score
                            for trade in baseline_solutions[token_pair].trades
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


class BatchCompatibilityFilter(ABC):
    @abstractmethod
    def get_filter_set(self, solution: Solution) -> set:
        pass


class DirectedTokenPairs(BatchCompatibilityFilter):
    def get_filter_set(self, solution: Solution) -> set:
        return {(trade.sell_token, trade.buy_token) for trade in solution.trades}


class TokenPairs(BatchCompatibilityFilter):
    def get_filter_set(self, solution: Solution) -> set:
        return {
            frozenset((trade.sell_token, trade.buy_token)) for trade in solution.trades
        }


class TradedTokens(BatchCompatibilityFilter):
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
        if len(solutions) == 0:
            return []
        return [sorted(solutions, key=lambda solution: solution.score)[-1]]


@dataclass(frozen=True)
class SubsetFilteringSelection(SolutionSelection):
    cumulative_filtering: bool = False
    batch_compatibility: BatchCompatibilityFilter = DirectedTokenPairs()

    def select_solutions(self, solutions: list[Solution]) -> list[Solution]:
        sorted_solutions = sorted(
            solutions, key=lambda _solution: _solution.score, reverse=True
        )
        selection: list[Solution] = []
        filter_set: set[str] = set()
        for solution in sorted_solutions:
            solution_filter_set = self.batch_compatibility.get_filter_set(solution)
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
    """
    MonotoneSelection implements a selection of winners based on a first selecting
    candidates for winners using `selection_rule`, and then checking if their rewards
    were positive when using a reference solution based on `selection_rule`. If rewards
    are negative, the winner is removed and a new set of candidates is computed.

    Using this approach, the total surplus of candidates increases whenever a solver
    is excluded.

    Attributes
    ----------
    selection_rule : SolutionSelection
        The rule used for selecting solutions from the input list. This is
        implemented as an instance of `SolutionSelection`.
    """

    selection_rule: SolutionSelection

    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        return self.winners_and_reference_scores(solutions)[0]

    def winners_and_reference_scores(
        self, solutions: list[Solution]
    ) -> tuple[list[Solution], dict[str, int]]:
        if not solutions:
            return [], {}
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
                SolverFilter(solver_max).filter(solutions)
            )

        return selection, reference_scores


@dataclass(frozen=True)
class FullCombinatorialSelection(WinnerSelection):
    """
    FullCombinatorialSelection implements a selection of winners based on full
    combinatorial maximization of score.

    The algorithm is based on the following steps:
    - Compute the best solution for each unique bundle of directed token pairs. Only
      these solutions can appear in the final set of winners.
      This step is done by iterating once through the input list of solutions.
    - Select the best collection of solutions for each bundle. This iteration through bundles
      by number of directed token pairs. For each fixed bundle, one iterates through all of their
      subsets, and reduces the problem to smaller bundles.
      For k token pairs in total this iterates through all 2**k possible bundles
      and checks all 2**k possible solutions in each case, for a runtime of O(4**k).

    Compatibility of batches is based on directed token pairs.
    """

    def compute_best_bundle_solutions(
        self, solutions: list[Solution]
    ) -> dict[frozenset, Solution]:
        """
        Computes the best solution for each unique bundle of directed token pairs from a list
        of solutions. A bundle is defined as a set of directed token pairs representing trades.
        The best solution for a bundle is determined by its score, and only the highest-scored
        solution for each bundle is retained.

        Parameters
        ----------
        solutions : list[Solution]
            A list of `Solution` objects representing potential trade bundles. Each solution
            contains a collection of trades and a score attribute that determines its quality.

        Returns
        -------
        best_bundle_solutions: dict[frozenset, Solution]
            A dictionary where each key is a frozenset representing a unique bundle of
            directed token pairs (sell token, buy token), and the corresponding value
            is the best-scoring `Solution` for that bundle.

        """
        best_bundle_solutions: dict[frozenset, Solution] = {}
        for solution in solutions:
            directed_token_pairs = frozenset(
                (trade.sell_token, trade.buy_token) for trade in solution.trades
            )
            if len(directed_token_pairs) == 0:
                continue
            if (
                directed_token_pairs not in best_bundle_solutions
                or best_bundle_solutions[directed_token_pairs].score < solution.score
            ):
                best_bundle_solutions[directed_token_pairs] = solution

        return best_bundle_solutions

    def select_winners(self, solutions: list[Solution]) -> list[Solution]:
        best_bundle_solutions = self.compute_best_bundle_solutions(solutions)

        all_token_pairs = set().union(*list(best_bundle_solutions.keys()))
        if not all_token_pairs:
            return []

        best_partition_solutions: dict[frozenset, list[Solution]] = {frozenset(): []}

        for level in range(1, len(all_token_pairs) + 1):
            for subset in itertools.combinations(all_token_pairs, level):
                best_bundle: frozenset = frozenset()
                best_value = 0
                for bundle, solution in best_bundle_solutions.items():
                    if not bundle.issubset(frozenset(subset)):
                        continue
                    current_value = solution.score + compute_total_score(
                        best_partition_solutions[frozenset(subset) - bundle]
                    )
                    if current_value > best_value:
                        best_bundle = bundle
                        best_value = current_value
                if best_value == 0:
                    best_partition_solutions[frozenset(subset)] = []
                else:
                    best_partition_solutions[frozenset(subset)] = (
                        best_partition_solutions[frozenset(subset) - best_bundle]
                        + [best_bundle_solutions[best_bundle]]
                    )

        return list(best_partition_solutions[frozenset(all_token_pairs)])


class RewardMechanism(ABC):
    @abstractmethod
    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, int]:
        """
        Abstract method to compute rewards for solvers based on winners and solutions.

        This method calculates the rewards for solvers, mapping each solver to its
        respective reward value.

        It is expected to be overridden by concrete subclasses implementing specific reward
        computation strategies.

        Parameters
        ----------
        winners : list[Solution]
            A list of solutions that are marked as winners.
        solutions : list[Solution]
            A list of all solutions from which winners were selected.

        Returns
        -------
        dict[str, int]
            A dictionary where the keys are solvers and the values are the
            respective rewards (as integers in atoms of the native token).
        """


class NoReward(RewardMechanism):
    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, int]:
        return {winner.id: 0 for winner in winners}


@dataclass(frozen=True)
class ReferenceReward(RewardMechanism):
    """
    ReferenceReward computes rewards for winners based on reference solutions.

    Given a list of winners and a list of solutions, this class implements the
    computation of rewards for each winner based on a reference solution computed
    using `winner_selection` after filtering out solutions from the winner.

    Rewards are capped from above using upper cap.

    Penalties are currently not implemented (and the lower cap is not used).
    """

    winner_selection: WinnerSelection
    upper_cap: int
    lower_cap: int

    def compute_rewards(
        self, winners: list[Solution], solutions: list[Solution]
    ) -> dict[str, int]:
        rewards: dict[str, int] = {}
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
            rewards[solver] = min(score - reference_score, self.upper_cap)

        return rewards


class AuctionMechanism(ABC):
    @abstractmethod
    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> tuple[list[Solution], dict[str, int]]:
        """
        Determines the winners among the provided solutions and calculates their
        corresponding rewards.

        This method evaluates a list of solutions, identifies which ones are the
        winners based on a defined criterion, and assigns rewards accordingly.
        The winners are returned as a list, and the rewards are returned as a
        dictionary where the keys are identifiers of winning solvers, and the values
        represent the respective rewards.

        Parameters
        ----------
        solutions : list[Solution]
            A list of solution objects. Each solution contains information
            that is evaluated to determine if it qualifies as a winner.

        Returns
        -------
        tuple[list[Solution], dict[str, int]]
            A tuple containing:
            - A list of winning `Solution` objects.
            - A dictionary mapping solvers to their respective rewards.
        """


@dataclass(frozen=True)
class FilterRankRewardMechanism(AuctionMechanism):
    """
    FilterRankRewardMechanism class handles the combined operations of solution filtering, winner
    selection, and reward computation within an auction mechanism. It integrates these stages to
    determine winners and associated rewards.

    Attributes
    ----------
    solution_filter : SolutionFilter
        An instance responsible for filtering solutions based on predefined criteria.
    winner_selection : WinnerSelection
        An instance responsible for selecting winners from the filtered solutions.
    reward_mechanism : RewardMechanism
        An instance responsible for computing rewards for the selected winners.
    """

    solution_filter: SolutionFilter
    winner_selection: WinnerSelection
    reward_mechanism: RewardMechanism

    def winners_and_rewards(
        self, solutions: list[Solution]
    ) -> tuple[list[Solution], dict[str, int]]:
        filtered_solutions = self.solution_filter.filter(solutions)
        winners = self.winner_selection.select_winners(filtered_solutions)
        rewards = self.reward_mechanism.compute_rewards(winners, filtered_solutions)
        return winners, rewards
