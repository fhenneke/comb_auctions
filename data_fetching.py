"""Functionality for fetching solutions data from the competition endpoint."""
import itertools
import math
import pickle
from fractions import Fraction
from os import getenv
from typing import Any, Literal

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from mechanism import Solution, Trade, aggregate_scores

load_dotenv()

network = getenv("NETWORK", "")

database_urls = {
    "prod": getenv("PROD_DB_URL", "").replace("NETWORK", network),
    "barn": getenv("BARN_DB_URL", "").replace("NETWORK", network),
}


def fetch_auctions_from_db(start_id: int, end_id: int) -> list[list[Solution]]:
    engine = create_engine("postgresql+psycopg://" + database_urls["prod"], echo=True)

    query = text(f"""with trade_data as (select ps.*,
                           pte.order_uid,
                           coalesce(o.sell_token, pjo.sell_token)  as sell_token,
                           coalesce(o.buy_token, pjo.buy_token)    as buy_token,
                           pte.executed_sell                       as executed_sell_amount,
                           pte.executed_buy                        as executed_buy_amount,
                           coalesce(o.sell_amount, pjo.limit_sell) as limit_sell_amount,
                           coalesce(o.buy_amount, pjo.limit_buy)   as limit_buy_amount,
                           coalesce(o.kind, pjo.side)              as kind
                    from proposed_solutions as ps
                             left outer join proposed_trade_executions as pte
                                  on ps.auction_id = pte.auction_id and ps.uid = pte.solution_uid
                             left outer join orders as o
                                             on pte.order_uid = o.uid
                             left outer join proposed_jit_orders as pjo
                                             on ps.auction_id = pjo.auction_id and
                                                ps.uid = pjo.solution_uid and
                                                pte.order_uid = pjo.order_uid
                    where ps.auction_id between {start_id} and {end_id}),

     trade_data_with_prices as (select td.*,
                                       ap_sell.price as sell_token_price,
                                       ap_buy.price  as buy_token_price
                                from trade_data as td
                                         join auction_prices as ap_sell
                                              on td.auction_id = ap_sell.auction_id and
                                                 td.sell_token = ap_sell.token
                                         join auction_prices as ap_buy
                                              on td.auction_id = ap_buy.auction_id and
                                                 td.buy_token = ap_buy.token
                                                  )

select *
from trade_data_with_prices""")
    with Session(engine) as session:
        with session.begin():
            result = session.execute(query)

    solutions_batch_dict: dict[int, dict[int, dict[str, Any]]] = {}
    for row in result.fetchall():
        auction_id = int(row.auction_id)
        solution_uid = int(row.uid)
        solutions_batch_dict[auction_id] = solutions_batch_dict.get(auction_id, {})
        solutions_batch_dict[auction_id][solution_uid] = solutions_batch_dict[auction_id].get(
            solution_uid, {})
        solutions_batch_dict[auction_id][solution_uid] = {
            "solver": "0x" + row.solver.hex(),
            "score": row.score,
            "trades": solutions_batch_dict[auction_id][solution_uid].get("trades", []) + [
                {
                    "order_uid": "0x" + row.order_uid.hex(),
                    "sell_token": "0x" + row.sell_token.hex(),
                    "buy_token": "0x" + row.buy_token.hex(),
                    "limit_sell_amount": row.limit_sell_amount,
                    "limit_buy_amount": row.limit_buy_amount,
                    "executed_sell_amount": row.executed_sell_amount,
                    "executed_buy_amount": row.executed_buy_amount,
                    "kind": row.kind,
                    "sell_token_price": row.sell_token_price,
                    "buy_token_price": row.buy_token_price,
                }
            ]
        }

    solutions_batch: list[list[Solution]] = []
    for auction_id in sorted(solutions_batch_dict.keys()):
        auction_data = solutions_batch_dict[auction_id]
        solutions = []
        for solution_uid in sorted(auction_data.keys()):
            solution_data = auction_data[solution_uid]
            solver = solution_data["solver"]
            trades: list[Trade] = []
            for trade_data in solution_data["trades"]:
                order_uid: str = trade_data["order_uid"]
                sell_token = trade_data["sell_token"]
                buy_token = trade_data["buy_token"]
                score = compute_surplus(trade_data)
                trades.append(Trade(order_uid, sell_token, buy_token, score))
            solution = Solution(
                id=str(solution_uid) + "-" + solver,
                solver=solver,
                score=sum(trade.score for trade in trades),
                trades=trades,
            )
            solutions.append(solution)
        solutions_batch.append(solutions)

    return solutions_batch


def fetch_auctions(auction_start, auction_end):
    try:
        with open(f"batches_{auction_start}_{auction_end}.pickle", 'rb') as handle:
            solutions_batch = pickle.load(handle)
    except FileNotFoundError:
        solutions_batch = fetch_auctions_from_db(auction_start, auction_end)
        with open(f"batches_{auction_start}_{auction_end}.pickle", "wb") as handle:
            pickle.dump(solutions_batch, handle, protocol=-1)
    return solutions_batch


def compute_surplus(trade_data) -> int:
    limit_sell = int(trade_data["limit_sell_amount"])
    limit_buy = int(trade_data["limit_buy_amount"])
    executed_sell = int(trade_data["executed_sell_amount"])
    executed_buy = int(trade_data["executed_buy_amount"])
    sell_price = Fraction(int(trade_data["sell_token_price"]), 10 ** 18)
    buy_price = Fraction(int(trade_data["buy_token_price"]), 10 ** 18)
    if trade_data["kind"] == "sell":
        partial_limit_buy = math.ceil(Fraction(limit_buy * executed_sell, limit_sell))
        surplus = executed_buy - partial_limit_buy
        surplus_eth = math.floor(surplus * buy_price)
    else:
        partial_limit_sell = math.floor(Fraction(limit_sell * executed_buy, limit_buy))
        surplus = partial_limit_sell - executed_sell
        surplus_eth = math.floor(surplus * sell_price)
    return surplus_eth


def compute_split_solutions(
        solutions: list[Solution], efficiency_loss: float = 0.0,
        approach: Literal["simple", "complete"] = "simple"
) -> list[Solution]:
    split_solutions: list[Solution] = []
    for solution in solutions:
        split_solutions += compute_split_solution(solution, efficiency_loss, approach)
    return split_solutions


def compute_split_solution(solution: Solution, efficiency_loss: float = 0.0,
                           approach: Literal["simple", "complete"] = "simple"):
    split_solution: list[Solution] = [solution]
    scores = aggregate_scores(solution)
    # make the following its own function
    if len(scores) > 1:
        if approach == "simple":
            for token_pair in scores:
                solution_id = solution.id + "-" + str(token_pair)
                solver = solution.solver
                trades = [
                    trade
                    for trade in solution.trades
                    if (trade.sell_token, trade.buy_token) == token_pair
                ]

                assert all(
                    trade.score is not None for trade in trades
                ), f"score not set for all trades: {trades}"
                score = sum(
                    round((1 - efficiency_loss) * trade.score)
                    for trade in trades
                    if trade.score is not None
                )

                split_solution.append(Solution(solution_id, solver, score, trades))
        if approach == "complete":
            token_pairs = list(scores.keys())
            for r in {1, 2, len(scores) - 2, len(scores) - 1} & set(range(1, len(scores))):
                token_pair_subsets = itertools.combinations(token_pairs, r)
                for token_pair_subset in token_pair_subsets:
                    solution_id = solution.id + "-" + str(token_pair_subset)
                    solver = solution.solver
                    trades = [
                        Trade(
                            trade.id,
                            trade.sell_token,
                            trade.buy_token,
                            int(trade.score * (1 - efficiency_loss) ** (len(token_pairs) - r))
                        )
                        for trade in solution.trades
                        if (trade.sell_token, trade.buy_token) in token_pair_subset
                    ]

                    assert all(
                        trade.score is not None for trade in trades
                    ), f"score not set for all trades: {trades}"
                    score = sum(
                        trade.score
                        for trade in trades
                        if trade.score is not None
                    )

                    split_solution.append(Solution(solution_id, solver, score, trades))

    return split_solution


if __name__ == "__main__":
    solutions = fetch_auctions(10322553 - 10, 10322553)
    print(solutions)
