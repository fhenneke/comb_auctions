from typing import Any

from mechanism import AuctionMechanism, Solution, get_orders


def run_counter_factual_analysis(
    auction_solutions_list: list[list[Solution]],
    mechanism: AuctionMechanism,
    remove_executed_orders: bool = True,
) -> list[tuple[list[Solution], dict[str, tuple[int, int]]]]:
    """Run a counterfactual analysis based on auction solutions.

    This function iterates through a list of auction solutions, evaluates the
    winners and their rewards using a specified auction mechanism, and optionally
    removes orders already executed. The results are aggregated for further
    analysis.

    Parameters
    ----------
    auction_solutions_list : list[list[Solution]]
        A list of auction solutions where each solution represents a possible
        outcome in the auction.
    mechanism : AuctionMechanism
        The auction mechanism used to determine the winners and their rewards
        based on the provided solutions.
    remove_executed_orders : bool, optional
        If True, excludes already-settled orders from the analysis. Defaults to
        True.

    Returns
    -------
    list[tuple[list[Solution], dict[str, tuple[int, int]]]]
        A list of tuples, where each tuple corresponds to the winners and their
        rewards for the respective solutions in the input list. Each tuple contains
        a list of winners and a dictionary mapping solvers to their rewards.
    """
    all_winners_rewards = []
    order_uids_settled: set[str] = set()
    for solutions in auction_solutions_list:
        # filter orders which are already settled
        if remove_executed_orders:
            solutions_filtered = [
                remove_order_from_solution(solution, order_uids_settled)
                for solution in solutions
            ]
        else:
            solutions_filtered = list(solutions)
        winners_rewards = mechanism.winners_and_rewards(solutions_filtered)
        winners, _ = winners_rewards
        all_winners_rewards.append(winners_rewards)
        order_uids_settled.update(get_orders(winners))
    return all_winners_rewards


def remove_order_from_solution(solution: Solution, order_uids: set[str]):
    """Removes specific orders from a given solution based on a set of order unique IDs.

    This function is designed to filter out trades from a given solution object whose
    unique IDs are specified in the provided set of order IDs. The resulting solution
    will retain all attributes from the original except for the filtered trades, and
    the score will be recalculated based on the remaining trades.

    Parameters
    ----------
    solution : Solution
        The original solution object containing all trades and associated metadata.
    order_uids : set[str]
        A set of unique IDs representing the orders to be removed from the solution.

    Returns
    -------
    Solution
        A new solution object that contains only the trades not filtered out
        based on the provided order unique IDs, with an updated score.
    """
    trades_filtered = [trade for trade in solution.trades if trade.id not in order_uids]
    solution_filtered = Solution(
        id=solution.id,
        solver=solution.solver,
        score=sum(trade.score for trade in trades_filtered),
        trades=trades_filtered,
    )
    return solution_filtered


def compute_statistics(
    auction_solutions_list: list[list[Solution]],
    all_winners_rewards: list[list[tuple[list[Solution], dict[str, tuple[int, int]]]]],
) -> None:
    """
    Compute and display statistics for multiple auction mechanisms, including metrics
    such as scores, rewards, and throughput. The statistics are calculated based on the
    solutions generated by auction mechanisms and the associated winners and rewards data.

    Parameters
    ----------
    auction_solutions_list : list of list of Solution
        A nested list where each sub-list contains `Solution` objects proposed
        during the auction processes.
    all_winners_rewards : list of list of tuple(list of Solution, dict of str to tuple of int, int)
        A nested list of tuples representing the winners and corresponding rewards
        for each solution proposed by multiple mechanisms. Each tuple contains a
        list of `Solution` objects representing the winners and a dictionary where
        keys are order identifiers (str) and values are tuples representing rewards
        (int) and other values (int).

    Returns
    -------
    None
        This function does not return any value. It prints the statistics for each
        mechanism to the standard output.

    Notes
    -----
    - The function iterates over the range of auction mechanisms to calculate the total
      scores from winners, total rewards, and throughput efficiency for every mechanism.
    - Throughput represents the percentage of orders settled immediately as soon as a
      solution is proposed.
    - The summary results for scores, rewards, and throughput are printed to the console
      along with relative changes or increases compared to the benchmark mechanism.
    - Statistical values, such as scores and rewards, are converted to ETH and formatted
      for better readability.
    """
    statistics: dict[str, list] = {"reward": [], "score": [], "throughput": []}
    K = len(all_winners_rewards)
    for k in range(K):
        score_for_mechanism = 0
        rewards_for_mechanism = 0
        orders_settled_immediately = 0
        all_orders: set[str] = set()
        for solutions, (winners, rewards) in zip(
            auction_solutions_list, all_winners_rewards[k]
        ):
            score_for_mechanism += sum(solution.score for solution in winners)
            rewards_for_mechanism += sum(reward for reward, _ in rewards.values())
            orders_settled = get_orders(winners)
            orders_proposed = get_orders(solutions)
            orders_settled_immediately += len(orders_settled - all_orders)
            all_orders.update(orders_proposed)

        statistics["score"].append(score_for_mechanism)
        statistics["reward"].append(rewards_for_mechanism)
        statistics["throughput"].append(orders_settled_immediately / len(all_orders))

    print("Statistics:")
    print("Score:")
    for k in range(K):
        print(
            f"mechanism {k} generated scores of {statistics["score"][k] / 10 ** 18} ETH "
            f"(relative change: {(statistics["score"][k] / statistics["score"][0] - 1) * 100:.2f}%)"
        )

    print("Reward:")
    for k in range(K):
        print(
            f"mechanism {k} generated rewards of {statistics["reward"][k] / 10 ** 18} ETH "
            f"(relative change: {(statistics["reward"][k] / statistics["reward"][0] - 1) * 100:.2f}%)"
        )

    print(
        "Throughput (percentage of orders executed the first time a solution is proposed):"
    )
    for k in range(K):
        print(
            f"mechanism {k} generated throughput of {statistics["throughput"][k] * 100:.2f}% ETH "
            f"(relative increase: {(statistics["throughput"][k] / statistics["throughput"][0] - 1) * 100:.2f}%)"
        )
