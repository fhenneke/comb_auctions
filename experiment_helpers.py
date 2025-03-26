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
            solutions_filtered = [
                solution for solution in solutions_filtered if solution.score > 0
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
    Computes and prints statistical analysis of auction mechanisms by processing solutions
    and reward data, such as scores, rewards, throughput, capped rewards, and negative
    rewards. The function iteratively processes data for different auction mechanisms,
    calculates relevant metrics, and outputs comparative statistics.

    Parameters
    ----------
    auction_solutions_list : list of list of Solution
        A list containing sublists of `Solution` objects for each auction, where
        each sublist represents solutions proposed by the mechanism.

    all_winners_rewards : list of list of tuple(list of Solution, dict of str to tuple of int, int)
        A list containing sublists of tuples for each auction mechanism, where each tuple
        consists of winners' solutions and a dictionary mapping order identifiers to reward
        and score values.

    Returns
    -------
    None
    """
    statistics: dict[str, list] = {
        "reward": [],
        "score": [],
        "throughput": [],
        "capped_rewards": [],
        "negative_reward": [],
    }
    K = len(all_winners_rewards)
    for k in range(K):

        # totals
        score_sum = sum(
            winner.score for winners, _ in all_winners_rewards[k] for winner in winners
        )
        reward_sum = sum(
            reward
            for _, rewards in all_winners_rewards[k]
            for reward, _ in rewards.values()
        )

        # capping
        reward_max = max(
            reward
            for _, rewards in all_winners_rewards[k]
            for reward, _ in rewards.values()
        )
        capped_rewards = sum(
            1
            for _, rewards in all_winners_rewards[k]
            if any(reward == reward_max for reward, _ in rewards.values())
        )

        negative_rewards = sum(
            1
            for _, rewards in all_winners_rewards[k]
            if any(reward == 0 for reward, _ in rewards.values())
        )

        # throughput
        orders_settled_immediately = 0
        all_orders: set[str] = set()
        for solutions, (winners, rewards) in zip(
            auction_solutions_list, all_winners_rewards[k]
        ):
            orders_settled = get_orders(winners)
            orders_proposed = get_orders(solutions)
            orders_settled_immediately += len(orders_settled - all_orders)
            all_orders.update(orders_proposed)

        statistics["score"].append(score_sum)
        statistics["reward"].append(reward_sum)
        statistics["negative_reward"].append(
            negative_rewards / len(auction_solutions_list)
        )
        statistics["capped_rewards"].append(
            capped_rewards / len(auction_solutions_list)
        )
        statistics["throughput"].append(orders_settled_immediately / len(all_orders))

    print(f"Number of auctions: {len(auction_solutions_list)}")

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
            f"mechanism {k} generated throughput of {statistics["throughput"][k] * 100:.2f}% "
            f"(relative increase: {(statistics["throughput"][k] / statistics["throughput"][0] - 1) * 100:.2f}%)"
        )

    print("Capping:")
    for k in range(K):
        print(
            f"mechanism {k} has capped rewards in "
            f"{statistics["capped_rewards"][k] * 100:.2f}% of auctions"
        )

    print("Negative reward:")
    for k in range(K):
        print(
            f"mechanism {k} generated negative rewards in "
            f"{statistics["negative_reward"][k] * 100:.2f}% of auctions"
        )
