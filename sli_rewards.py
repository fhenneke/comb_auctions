import altair as alt
import polars as pl

from data_fetching import fetch_auctions

from mechanism import (
    FilterRankRewardMechanism,
    BaselineFilter,
    DirectedTokenPairs,
    DirectSelection,
    SubsetFilteringSelection,
    ReferenceReward,
    Solution,
    Trade,
)

auction_start = 12187189 - 50000
auction_end = 12187189

solutions_batch = fetch_auctions(auction_start, auction_end)

reward_cap_upper = 10**24
reward_cap_lower = 10 * 10**15
mechanism = FilterRankRewardMechanism(
    BaselineFilter(),
    DirectSelection(
        SubsetFilteringSelection(
            batch_compatibility=DirectedTokenPairs(), cumulative_filtering=False
        )
    ),
    ReferenceReward(
        DirectSelection(
            SubsetFilteringSelection(
                batch_compatibility=DirectedTokenPairs(),
                cumulative_filtering=False,
            )
        ),
        reward_cap_upper,
        reward_cap_lower,
    ),
)

def surplus_metric(solutions):
    winners, _ = mechanism.winners_and_rewards(solutions)
    return sum(sum(trade.score / 1e18 for trade in winner.trades) for winner in winners)

def nr_orders_metric(solutions):
    winners, _ = mechanism.winners_and_rewards(solutions)
    return sum(len(solution.trades) for solution in winners)

def compute_participation_rates(solutions_batch):
    total_orders = len({trade.id for solutions in solutions_batch for solution in solutions for trade in solution.trades})
    all_solvers = {solution.solver for solutions in solutions_batch for solution in solutions}
    orders_per_solver = {solver: len({trade.id for solutions in solutions_batch for solution in solutions for trade in solution.trades if solution.solver == solver}) for solver in all_solvers}
    participation_rates = {solver: nr_orders / total_orders for solver, nr_orders in orders_per_solver.items()}
    return participation_rates

def robust_surplus_metric(solutions, participation_rates):
    robust_surplus = 0
    all_order_uids = {trade.id for solution in solutions for trade in solution.trades}
    for order_uid in all_order_uids:
        order_scores = [(solution.solver, trade.score) for solution in solutions for trade in solution.trades if trade.id == order_uid]
        # sorting
        order_scores = sorted(order_scores, key=lambda x: x[1], reverse=True)
        # removing duplicates
        solvers_seen = set()
        order_scores = [x for x in order_scores if not (x[0] in solvers_seen or solvers_seen.add(x[0]))]
        robust_surplus_order = 0.0
        remaining_probability = 1.
        for solver, score in order_scores:
            robust_surplus_order += remaining_probability * participation_rates[solver] * score / 1e18
            remaining_probability *= 1 - participation_rates[solver]
        robust_surplus += robust_surplus_order
    return robust_surplus

def compute_metrics(solutions_batch):
    participation_rates = compute_participation_rates(solutions_batch)
    metrics = [surplus_metric, nr_orders_metric, lambda x: robust_surplus_metric(x, participation_rates)]
    all_winners_rewards = []
    all_metrics = []
    reference_metrics = []
    single_metrics = []
    for solutions in solutions_batch:
        winners, rewards = mechanism.winners_and_rewards(solutions)
        all_winners_rewards.append((winners, rewards))
        # all_metrics.append([metric(winners) for metric in metrics])
        all_metrics.append([metric(solutions) for metric in metrics])
        reference_metrics.append({})
        single_metrics.append({})
        solvers = {solution.solver for solution in solutions}
        # metrics if solver were removed
        for solver in solvers:
            filtered_solutions = [solution for solution in solutions if solution.solver != solver]
            # reference_winners, _ = mechanism.winners_and_rewards(
            #     filtered_solutions
            # )
            # reference_metrics[-1][solver] = [metric(reference_winners) for metric in metrics]
            reference_metrics[-1][solver] = [metric(filtered_solutions) for metric in metrics]
        # metric if only one solver participated
        for solver in solvers:
            filtered_solutions = [solution for solution in solutions if solution.solver == solver]
            # reference_winners, _ = mechanism.winners_and_rewards(
            #     filtered_solutions
            # )
            # single_metrics[-1][solver] = [metric(reference_winners) for metric in metrics]
            single_metrics[-1][solver] = [metric(filtered_solutions) for metric in metrics]

    # set up polars data frames
    auction_metrics_df = pl.DataFrame(
        all_metrics,
        schema=["surplus", "nr_orders", "robust_surplus"],
        orient="row",
    ).with_row_index("index")
    metric_without_solver_df = pl.DataFrame(
        [[i, k, v[0], v[1], v[2]] for i, d in enumerate(reference_metrics) for k, v in d.items()],
        schema=["index", "solver", "surplus", "nr_orders", "robust_surplus"],
        orient="row",
    )
    metric_improvement_df = metric_without_solver_df.join(
        auction_metrics_df, on="index", suffix="_t2"
    ).select(
        "index",
        "solver",
        (pl.col("surplus_t2") - pl.col("surplus")).alias("surplus_marginal"),
        (pl.col("nr_orders_t2") - pl.col("nr_orders")).alias("nr_orders_marginal"),
        (pl.col("robust_surplus_t2") - pl.col("robust_surplus")).alias("robust_surplus_marginal"),
    )
    metric_with_single_solver_df = pl.DataFrame(
        [[i, k, v[0], v[1], v[2]] for i, d in enumerate(single_metrics) for k, v in d.items()],
        schema=["index", "solver", "surplus_individual", "nr_orders_individual", "robust_surplus_individual"],
        orient="row",
    )

    combined_metrics_df = metric_improvement_df.join(
        metric_with_single_solver_df, on=["index", "solver"]
    )

    return combined_metrics_df



combined_metrics_df = compute_metrics(solutions_batch)
combined_metrics_df.group_by(pl.col("solver")).sum().sort(by="surplus_marginal", descending=True)

combined_metrics_without_small_orders_df = compute_metrics([[solution for solution in solutions if not (solution.solver == "0xa9d635ef85bc37eb9ff9d6165481ea230ed32392" and sum(trade.volume for trade in solution.trades) < 10**18)] for solutions in solutions_batch])
combined_metrics_without_small_orders_df.group_by(pl.col("solver")).sum().sort(by="surplus_marginal", descending=True)

# Reshape to long format for grouped bars
df_long = combined_metrics_df.group_by(pl.col("solver")).sum().unpivot(
    index="solver",
    on=["surplus_marginal", "nr_orders_marginal", "robust_surplus_marginal", "surplus_individual", "nr_orders_individual", "robust_surplus_individual"],
    variable_name="metric",
    value_name="value",
).with_columns(
    (pl.col("value") / pl.col("value").sum().over("metric"))
    .alias("value")
)

solver_order = (
    df_long.filter(pl.col("metric") == "surplus_marginal")
    .sort("value", descending=True)
    .get_column("solver")
    .to_list()
)

chart = df_long.plot.bar(
    x=alt.X("solver", sort=solver_order),
    y="value",
    color="metric",
    xOffset="metric",
).properties(
    width="container",
    height=400,
)

chart.save("chart.html")

## test case

# participation_rates = {"solver 1": 0.9, "solver 2": 0.8, "solver 3": 0.95}
# solutions = [
#     Solution(
#         id="solution 1",
#         solver="solver 1",
#         score=10,
#         trades=[Trade(
#             id="order 1",
#             sell_token="A",
#             buy_token="B",
#             score=10,
#         )],
#     ),
#     Solution(
#         id="solution 2",
#         solver="solver 2",
#         score=10,
#         trades=[Trade(
#             id="order 1",
#             sell_token="A",
#             buy_token="B",
#             score=9,
#         )],
#     ),
#     Solution(
#         id="solution 3",
#         solver="solver 3",
#         score=10,
#         trades=[Trade(
#             id="order 1",
#             sell_token="A",
#             buy_token="B",
#             score=8,
#         )],
#     ),
# ]

# robust_surplus_metric(solutions)

# robust_surplus_metric(solutions) - robust_surplus_metric([solution for solution in solutions if solution.solver != "solver 1"])
# robust_surplus_metric(solutions) - robust_surplus_metric([solution for solution in solutions if solution.solver != "solver 2"])
# # robust_surplus_metric(solutions) - robust_surplus_metric([solution for solution in solutions if solution.solver != "solver 3"])
