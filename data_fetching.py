"""Functionality for fetching solutions data from the competition endpoint."""

from fractions import Fraction
import math
from os import getenv
from typing import Any

from dotenv import load_dotenv
import requests
from sqlalchemy import create_engine, text, Table, MetaData, select
from sqlalchemy.orm import Session

from mechanism import Solution, Trade, aggregate_scores

load_dotenv()

network = getenv("NETWORK", "")

database_urls = {
    "prod": getenv("PROD_DB_URL", "").replace("NETWORK", network),
    "barn": getenv("BARN_DB_URL", "").replace("NETWORK", network),
}

orderbook_urls = {
    "prod": f"https://api.cow.fi/{network}/api/v1/",
    "barn": f"https://barn.api.cow.fi/{network}/api/v1/",
}
REQUEST_TIMEOUT = 5


def fetch_competition_data_from_orderbook(tx_hash: str) -> tuple[str, dict[str, Any]]:
    for environment, url in orderbook_urls.items():
        try:
            response = requests.get(
                url + f"solver_competition/by_tx_hash/{tx_hash}",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            competition_data = response.json()
            return environment, competition_data
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                pass
            else:
                raise err
    raise ValueError(f"Could not fetch solution data for hash {tx_hash!r}")


def fetch_competition_data_batch(start_id: int, end_id: int) -> list[dict[str, Any]]:
    engine = create_engine("postgresql+psycopg://" + database_urls["prod"], echo=True)
    with engine.connect() as connection:
        result = connection.execute(
            text(
                "select * from solver_competitions where id between :start_id and :end_id"
            ),
            {"start_id": start_id, "end_id": end_id},
        )
    competition_data_batch = [res[1] for res in result.fetchall()]
    return competition_data_batch


def fetch_order_data(
    competition_data_batch: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    order_uids: set[str] = set()
    for competition_data in competition_data_batch:
        solution_data: list[dict[str, Any]] = competition_data["solutions"]
        for solution in solution_data:
            for order in solution["orders"]:
                order_uids.add(order["id"])
    return fetch_order_data_from_database(order_uids)


def fetch_order_data_from_orderbook(order_uids: set[str]) -> dict[str, dict[str, Any]]:
    order_data: dict[str, dict[str, Any]] = {}
    for order_uid in order_uids:
        for url in orderbook_urls.values():
            try:
                response = requests.get(
                    url + f"orders/{order_uid}",
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                order_data[order_uid] = response.json()
                break
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    pass
                else:
                    raise err

    return order_data


def fetch_order_data_from_database(order_uids: set[str]) -> dict[str, dict[str, Any]]:
    engine = create_engine("postgresql+psycopg://" + database_urls["prod"], echo=True)

    order_table = Table("orders", MetaData(), autoload_with=engine)
    query = select(order_table).where(
        order_table.c.uid.in_(
            [bytes.fromhex(order_uid[2:]) for order_uid in order_uids]
        )
    )

    with Session(engine) as session:
        with session.begin():
            result = session.execute(query)

    order_data: dict[str, dict[str, Any]] = {}
    for row in result.fetchall():
        order_data["0x" + row.uid.hex()] = {
            "sellToken": "0x" + row.sell_token.hex(),
            "buyToken": "0x" + row.buy_token.hex(),
            "sellAmount": int(row.sell_amount),
            "buyAmount": int(row.buy_amount),
            "kind": row.kind,
        }

    return order_data


def get_solution_data_single(
    competition_data: dict[str, Any], order_data: dict[str, dict[str, Any]]
) -> list[Solution]:
    solution_data: list[dict[str, Any]] = competition_data["solutions"]
    native_prices: list[dict[str, Any]] = competition_data["auction"]["prices"]

    solutions = []
    for solution_id, solution in enumerate(solution_data):
        solver = solution["solverAddress"]
        trades: list[Trade] = []
        for order_execution in solution["orders"]:
            order_id: str = order_execution["id"]
            if order_id not in order_data:
                continue
            order = order_data[order_id]
            sell_token = order["sellToken"]
            buy_token = order["buyToken"]

            surplus = compute_surplus(order, order_execution, native_prices)

            trades.append(Trade(order_id, sell_token, buy_token, surplus))

        # use sum of trade scores instead of int(solution["score"])
        score = sum(trade.score for trade in trades)

        solution_obj = Solution(
            id=str(solution_id) + "-" + solution["solver"],
            solver=solver,
            score=score,
            trades=trades,
        )
        solutions.append(solution_obj)

    return solutions


def get_solution_data_batch(
    competition_data_batch: list[dict[str, Any]],
    order_data: dict[str, dict[str, Any]],
    split: bool = False,
    efficiency_loss: float = 0.0,
) -> list[list[Solution]]:
    solutions_batch: list[list[Solution]] = []
    for competition_data in competition_data_batch:
        solutions = get_solution_data_single(competition_data, order_data)
        if split:
            solutions = compute_split_solutions(solutions, efficiency_loss)

        solutions_batch.append(solutions)

    return solutions_batch


def compute_split_solutions(
    solutions: list[Solution], efficiency_loss: float = 0.0
) -> list[Solution]:
    split_solutions: list[Solution] = []
    for solution in solutions:
        split_solutions += compute_split_solution(solution, efficiency_loss)
    return split_solutions


def compute_split_solution(solution: Solution, efficiency_loss: float = 0.0):
    split_solution: list[Solution] = [solution]
    scores = aggregate_scores(solution)
    # make the following its own function
    if len(scores) > 1:
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
    return split_solution


def compute_surplus(order, order_execution, native_prices) -> int:
    limit_sell = int(order["sellAmount"])
    limit_buy = int(order["buyAmount"])
    executed_sell = int(order_execution["sellAmount"])
    executed_buy = int(order_execution["buyAmount"])
    sell_price = Fraction(int(native_prices[order["sellToken"]]), 10**18)
    buy_price = Fraction(int(native_prices[order["buyToken"]]), 10**18)
    if order["kind"] == "sell":
        partial_limit_buy = math.ceil(Fraction(limit_buy * executed_sell, limit_sell))
        surplus = executed_buy - partial_limit_buy
        surplus_eth = math.floor(surplus * buy_price)
    else:
        partial_limit_sell = math.floor(Fraction(limit_sell * executed_buy, limit_buy))
        surplus = partial_limit_sell - executed_sell
        surplus_eth = math.floor(surplus * sell_price)
    return surplus_eth


def fetch_solutions_batch(start_id: int, end_id: int) -> list[list[Solution]]:
    competition_data_batch = fetch_competition_data_batch(start_id, end_id)
    order_data = fetch_order_data(competition_data_batch)
    solutions_batch = get_solution_data_batch(competition_data_batch, order_data)

    return solutions_batch


def fetch_solutions_single(
    tx_hash: str, split_solutions: bool = False, efficiency_loss: float = 0.0
) -> list[Solution]:
    _, competition_data = fetch_competition_data_from_orderbook(tx_hash)
    order_data = fetch_order_data([competition_data])
    submitted_solutions = get_solution_data_single(competition_data, order_data)

    solutions: list[Solution] = []
    for solution in submitted_solutions:
        solutions.append(solution)
        scores = aggregate_scores(solution)
        # make the following its own function
        if len(scores) > 1 and split_solutions:
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

                solutions.append(Solution(solution_id, solver, score, trades))

    return solutions


if __name__ == "__main__":
    # solutions = fetch_solutions_single(
    #     "0x659a6b86aa25c01ba6bc65d63c4204a962f91073767372aa59d89e340aec219b"
    # )
    solutions = fetch_solutions_batch(9534992 - 100, 9534992)
    print(solutions)
