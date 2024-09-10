"""Functionality for fetching solutions data from the competition endpoint."""

from os import getenv
from typing import Any

from dotenv import load_dotenv
import requests
from sqlalchemy import create_engine, text

from mechanism import Solution

load_dotenv()

network = getenv("NETWORK")

database_urls = {
    "prod": getenv("PROD_DB_URL").replace("NETWORK", network),
    "barn": getenv("BARN_DB_URL").replace("NETWORK", network),
}

orderbook_urls = {
    "prod": f"https://api.cow.fi/{network}/api/v1/",
    "barn": f"https://barn.api.cow.fi/{network}/api/v1/",
}
REQUEST_TIMEOUT = 5


def fetch_competition_data(tx_hash: str) -> tuple[str, dict[str, Any]]:
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


def fetch_order_data(
    environment: str, competition_data: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    order_data: dict[str, dict[str, Any]] = {}
    solution_data: list[dict[str, Any]] = competition_data["solutions"]
    for solution in solution_data:
        for order in solution["orders"]:
            order_uid = order["id"]
            if order_uid in order_data:
                continue
            url = orderbook_urls[environment]
            response = requests.get(
                url + f"orders/{order_uid}",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            order_data[order_uid] = response.json()
    return order_data


def fetch_computition_data(start_id: int, end_id: int) -> list[dict[str, Any]]:
    engine = create_engine("postgresql+psycopg://" + database_urls["prod"], echo=True)
    with engine.connect() as connection:
        result = connection.execute(
            text("select * from solver_competitions order by id desc limit 10")
        )
        print(result)


def aggregate_solution_data(
    competition_data: dict[str, Any], order_data: dict[str, dict[str, Any]]
) -> list[Solution]:
    solution_data: list[dict[str, Any]] = competition_data["solutions"]
    native_prices: list[dict[str, Any]] = competition_data["auction"]["prices"]

    solutions = []
    for id, solution in enumerate(solution_data):
        solver = solution["solverAddress"]
        scores: dict[tuple[str, str], int] = {}
        for order_execution in solution["orders"]:
            order = order_data[order_execution["id"]]
            sell_token = order["sellToken"]
            buy_token = order["buyToken"]

            surplus = compute_surplus(order, order_execution, native_prices)
            scores[(sell_token, buy_token)] = (
                scores.get((sell_token, buy_token), 0) + surplus
            )

        solution_obj = Solution(id=str(id), solver=solver, scores=scores)
        solutions.append(solution_obj)

    return solutions


def compute_surplus(order, order_execution, native_prices) -> int:
    assert not order["partiallyFillable"]
    if order["kind"] == "sell":
        limit_buy = int(order["buyAmount"])
        surplus = int(order_execution["buyAmount"]) - limit_buy
        surplus_eth = surplus * int(native_prices[order["buyToken"]]) // 10**18
    else:
        limit_sell = int(order["sellAmount"])
        surplus = int(order_execution["sellAmount"]) - limit_sell
        surplus_eth = surplus * int(native_prices[order["sellToken"]]) // 10**18
    return surplus_eth


def fetch_solutions(tx_hash: str, efficiency_loss: float = 0.0) -> list[Solution]:
    environment, competition_data = fetch_competition_data(tx_hash)
    order_data = fetch_order_data(environment, competition_data)
    submitted_solutions = aggregate_solution_data(competition_data, order_data)

    solutions: list[Solution] = []
    for solution in submitted_solutions:
        solutions.append(solution)
        if len(solution.scores) > 1:
            for token_pair, score in solution.scores.items():
                id = solution.id + "-" + str(token_pair)
                solver = solution.solver
                scores = {token_pair: round(score * (1 - efficiency_loss))}
                solutions.append(Solution(id, solver, scores))

    return solutions


if __name__ == "__main__":
    tx_hash = "0x659a6b86aa25c01ba6bc65d63c4204a962f91073767372aa59d89e340aec219b"
    # environment, competition_data = fetch_competition_data(tx_hash)
    # order_data = fetch_order_data(environment, competition_data)
    solutions = fetch_solutions(tx_hash)
    fetch_computition_data(1, 1)
    print(solutions)
