"""Inspect a solver competition and visualize it in a small HTML UI.

Fetches competition data from the CoW Protocol API (by auction id or transaction
hash), computes surplus per trade analogously to data_fetching.compute_score,
reruns the combinatorial auction mechanism from mechanism.py on surplus-based
scores (baseline computation, fairness filtering, winner selection, reference
scores), and renders everything into a self-contained interactive HTML file.

Usage:
    uv run inspect_auction.py 13120691
    uv run inspect_auction.py 0x84b6497a...4bc167 --network mainnet
"""

import argparse
import json
import math
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import requests

from mechanism import (
    BaselineFilter,
    Solution,
    SubsetFilteringSelection,
    Trade,
    aggregate_scores,
    compute_baseline_solutions,
    compute_total_score,
)

API_BASE = "https://api.cow.fi/{network}/api"
TOKEN_LIST_URL = "https://files.cow.fi/tokens/CowSwap.json"
SOLVER_NETWORKS_URL = "https://cms.cow.fi/api/solver-networks"
NATIVE_TOKEN_ADDRESS = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
REQUEST_TIMEOUT = 30

NETWORKS: dict[str, dict[str, Any]] = {
    "mainnet": {
        "chain_id": 1,
        "explorer": "https://etherscan.io",
        "cow_explorer_prefix": "",
        "native_symbol": "ETH",
    },
    "xdai": {
        "chain_id": 100,
        "explorer": "https://gnosisscan.io",
        "cow_explorer_prefix": "gc/",
        "native_symbol": "xDAI",
    },
    "arbitrum_one": {
        "chain_id": 42161,
        "explorer": "https://arbiscan.io",
        "cow_explorer_prefix": "arb1/",
        "native_symbol": "ETH",
    },
    "base": {
        "chain_id": 8453,
        "explorer": "https://basescan.org",
        "cow_explorer_prefix": "base/",
        "native_symbol": "ETH",
    },
    "polygon": {
        "chain_id": 137,
        "explorer": "https://polygonscan.com",
        "cow_explorer_prefix": "pol/",
        "native_symbol": "POL",
    },
    "avalanche": {
        "chain_id": 43114,
        "explorer": "https://snowtrace.io",
        "cow_explorer_prefix": "avax/",
        "native_symbol": "AVAX",
    },
    "sepolia": {
        "chain_id": 11155111,
        "explorer": "https://sepolia.etherscan.io",
        "cow_explorer_prefix": "sepolia/",
        "native_symbol": "ETH",
    },
}


def fetch_competition(network: str, reference: str) -> dict:
    """Fetch competition data by auction id, settlement tx hash, or 'latest'."""
    base = API_BASE.format(network=network) + "/v2/solver_competition"
    if reference.startswith("0x") and len(reference) == 66:
        url = f"{base}/by_tx_hash/{reference}"
    elif reference == "latest":
        url = f"{base}/latest"
    else:
        url = f"{base}/{int(reference)}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_orders(network: str, order_uids: list[str]) -> dict[str, dict | None]:
    """Fetch order data for all uids; missing orders (e.g. JIT) map to None."""
    base = API_BASE.format(network=network) + "/v1/orders/"

    def fetch_one(uid: str) -> dict | None:
        response = requests.get(base + uid, timeout=REQUEST_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    with ThreadPoolExecutor(max_workers=8) as executor:
        order_data = executor.map(fetch_one, order_uids)
    return dict(zip(order_uids, order_data))


def fetch_token_info(chain_id: int) -> dict[str, dict]:
    """Fetch symbols and decimals from the CoW token list; empty dict on failure."""
    try:
        response = requests.get(TOKEN_LIST_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        tokens = response.json()["tokens"]
    except (requests.RequestException, KeyError, ValueError):
        return {}
    return {
        token["address"].lower(): {
            "symbol": token["symbol"],
            "decimals": token["decimals"],
        }
        for token in tokens
        if token["chainId"] == chain_id
    }


def fetch_solver_names(chain_id: int) -> dict[str, str]:
    """Fetch submission address -> solver display name from the CoW CMS.

    Uses the same data source as the CoW explorer ("solved by" on order pages);
    returns an empty dict on failure.
    """
    names: dict[str, str] = {}
    page = 1
    try:
        while True:
            response = requests.get(
                SOLVER_NETWORKS_URL,
                params={
                    "filters[network][chainId][$eq]": str(chain_id),
                    "populate[solver]": "*",
                    "pagination[pageSize]": "100",
                    "pagination[page]": str(page),
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            for entry in result["data"]:
                attributes = entry["attributes"]
                solver = attributes["solver"]["data"]
                if solver is not None:
                    names[attributes["address"].lower()] = solver["attributes"][
                        "displayName"
                    ]
            if page >= result["meta"]["pagination"]["pageCount"]:
                break
            page += 1
    except (requests.RequestException, KeyError, ValueError):
        return {}
    return names


def compute_trade_surplus(
    order: dict | None,
    executed_sell: int,
    executed_buy: int,
    native_prices: dict[str, int],
) -> int | None:
    """Compute the surplus of a trade in atoms of the native token.

    Mirrors data_fetching.compute_score, with order data and native prices
    coming from the API instead of the database. Returns None if the order is
    unknown (JIT order) or no native price is available for the buy token.
    """
    if order is None:
        return None
    buy_token_price = native_prices.get(order["buyToken"].lower())
    if buy_token_price is None:
        return None
    limit_sell = int(order["sellAmount"])
    limit_buy = int(order["buyAmount"])
    buy_price = Fraction(buy_token_price, 10**18)
    if order["kind"] == "sell":
        partial_limit_buy = math.ceil(Fraction(limit_buy * executed_sell, limit_sell))
        surplus = executed_buy - partial_limit_buy
        score = math.floor(surplus * buy_price)
    else:
        partial_limit_sell = math.floor(Fraction(limit_sell * executed_buy, limit_buy))
        surplus = partial_limit_sell - executed_sell
        score = math.floor(surplus * Fraction(limit_buy, limit_sell) * buy_price)
    return score


def pair_key(sell_token: str, buy_token: str) -> str:
    """Serialize a directed token pair into a string key for use in JSON."""
    return f"{sell_token}|{buy_token}"


def build_solutions(
    competition: dict, orders: dict[str, dict | None]
) -> tuple[list[Solution], list[dict[str, int | None]]]:
    """Build mechanism.Solution objects with surplus-based scores.

    Solution ids are list indices into competition["solutions"]; trades with
    unknown surplus (JIT orders, missing prices) enter with score 0. The second
    return value contains the raw surplus per order uid for each solution, with
    None marking unknown surplus.
    """
    native_prices = {
        token.lower(): int(price)
        for token, price in competition["auction"]["prices"].items()
    }
    solutions = []
    surpluses: list[dict[str, int | None]] = []
    for index, solution_data in enumerate(competition["solutions"]):
        trades = []
        trade_surpluses: dict[str, int | None] = {}
        for order_execution in solution_data["orders"]:
            uid = order_execution["id"]
            surplus = compute_trade_surplus(
                orders[uid],
                int(order_execution["sellAmount"]),
                int(order_execution["buyAmount"]),
                native_prices,
            )
            trade_surpluses[uid] = surplus
            trades.append(
                Trade(
                    id=uid,
                    sell_token=order_execution["sellToken"].lower(),
                    buy_token=order_execution["buyToken"].lower(),
                    score=surplus if surplus is not None else 0,
                )
            )
        solutions.append(
            Solution(
                id=str(index),
                solver=solution_data["solverAddress"],
                score=sum(trade.score for trade in trades),
                trades=trades,
            )
        )
        surpluses.append(trade_surpluses)
    return solutions, surpluses


def analyze_mechanism(solutions: list[Solution]) -> dict:
    """Rerun the auction mechanism on surplus-based scores.

    Uses the mechanism currently deployed for the combinatorial auction:
    fairness (baseline) filtering and greedy winner selection compatible on
    directed token pairs, with reference scores computed by rerunning the
    selection without each winning solver (cf. counter_factual_analysis).
    """
    selection = SubsetFilteringSelection()
    baseline_solutions = compute_baseline_solutions(solutions)
    filtered_solutions = BaselineFilter().filter(solutions)
    filtered_ids = {solution.id for solution in filtered_solutions}
    winners = selection.select_solutions(filtered_solutions)
    winners_total = compute_total_score(winners)

    baselines = {
        pair_key(*token_pair): {
            "solutionIndex": int(solution.id),
            "surplus": str(solution.score),
        }
        for token_pair, solution in baseline_solutions.items()
    }

    # for solutions removed by the baseline filter, find the violating pairs
    violations: dict[str, list[dict]] = {}
    for solution in solutions:
        if solution.id in filtered_ids:
            continue
        solution_violations = []
        for token_pair, score in aggregate_scores(solution).items():
            baseline = baseline_solutions.get(token_pair)
            baseline_score = baseline.score if baseline is not None else 0
            if score < baseline_score:
                solution_violations.append(
                    {
                        "pair": pair_key(*token_pair),
                        "surplus": str(score),
                        "baselineSurplus": str(baseline_score),
                        "baselineSolutionIndex": (
                            int(baseline.id) if baseline is not None else None
                        ),
                    }
                )
        violations[solution.id] = solution_violations

    references = {}
    for solver in {winner.solver for winner in winners}:
        rest = [
            solution for solution in filtered_solutions if solution.solver != solver
        ]
        reference_winners = selection.select_solutions(rest)
        references[solver] = {
            "score": str(compute_total_score(reference_winners)),
            "solutionIndices": [int(winner.id) for winner in reference_winners],
        }

    return {
        "baselines": baselines,
        "filteredIndices": sorted(
            int(solution.id)
            for solution in solutions
            if solution.id not in filtered_ids
        ),
        "violations": {int(key): value for key, value in violations.items()},
        "winnerIndices": [int(winner.id) for winner in winners],
        "winnersTotal": str(winners_total),
        "references": references,
    }


def assemble_view_data(
    network: str,
    competition: dict,
    orders: dict[str, dict | None],
    solutions: list[Solution],
    surpluses: list[dict[str, int | None]],
    analysis: dict,
    token_info: dict[str, dict],
    solver_names: dict[str, str],
) -> dict:
    """Assemble the JSON object embedded into the HTML template."""
    network_config = NETWORKS.get(network, NETWORKS["mainnet"])
    token_info = token_info | {
        NATIVE_TOKEN_ADDRESS: {
            "symbol": network_config["native_symbol"],
            "decimals": 18,
        }
    }

    solution_views = []
    for index, solution_data in enumerate(competition["solutions"]):
        solution = solutions[index]
        trades = {}
        for order_execution, trade in zip(solution_data["orders"], solution.trades):
            surplus = surpluses[index][trade.id]
            trades[trade.id] = {
                "sellAmount": order_execution["sellAmount"],
                "buyAmount": order_execution["buyAmount"],
                "surplus": str(surplus) if surplus is not None else None,
            }
        pair_surplus = {
            pair_key(*token_pair): str(score)
            for token_pair, score in aggregate_scores(solution).items()
        }
        solution_views.append(
            {
                "index": index,
                "solver": solution_data["solverAddress"],
                "ranking": solution_data.get("ranking"),
                "scoreApi": solution_data["score"],
                "isWinner": solution_data["isWinner"],
                "filteredOutApi": solution_data["filteredOut"],
                "txHash": solution_data.get("txHash"),
                "referenceScoreApi": solution_data.get("referenceScore"),
                "surplusTotal": str(solution.score),
                "trades": trades,
                "pairSurplus": pair_surplus,
            }
        )

    # group orders by directed token pair, ordered by total surplus over solutions
    pair_orders: dict[str, dict] = {}
    for solution in solutions:
        for trade in solution.trades:
            key = pair_key(trade.sell_token, trade.buy_token)
            pair = pair_orders.setdefault(
                key,
                {
                    "sellToken": trade.sell_token,
                    "buyToken": trade.buy_token,
                    "orderUids": [],
                    "_total": 0,
                },
            )
            if trade.id not in pair["orderUids"]:
                pair["orderUids"].append(trade.id)
            pair["_total"] += trade.score
    pairs = sorted(pair_orders.values(), key=lambda pair: -pair["_total"])
    for pair in pairs:
        del pair["_total"]

    order_views = {
        uid: (
            {
                "found": True,
                "kind": order["kind"],
                "class": order["class"],
                "limitSell": order["sellAmount"],
                "limitBuy": order["buyAmount"],
                "sellToken": order["sellToken"].lower(),
                "buyToken": order["buyToken"].lower(),
            }
            if order is not None
            else {"found": False}
        )
        for uid, order in orders.items()
    }

    return {
        "network": network,
        "nativeSymbol": network_config["native_symbol"],
        "explorerBase": network_config["explorer"],
        "cowExplorerOrderBase": "https://explorer.cow.fi/"
        + network_config["cow_explorer_prefix"]
        + "orders/",
        "auctionId": competition["auctionId"],
        "auctionStartBlock": competition.get("auctionStartBlock"),
        "auctionDeadlineBlock": competition.get("auctionDeadlineBlock"),
        "transactionHashes": competition.get("transactionHashes", []),
        "referenceScoresApi": competition.get("referenceScores", {}),
        "tokens": token_info,
        "solverNames": solver_names,
        "orders": order_views,
        "pairs": pairs,
        "solutions": solution_views,
        "analysis": analysis,
    }


def render_html(view_data: dict) -> str:
    """Embed the view data into the HTML template."""
    template = (Path(__file__).parent / "viewer_template.html").read_text(
        encoding="utf-8"
    )
    return template.replace("__DATA_JSON__", json.dumps(view_data))


def print_summary(view_data: dict) -> None:
    """Print a one-line-per-solution summary of the competition to stdout."""
    analysis = view_data["analysis"]
    print(f"auction {view_data['auctionId']} on {view_data['network']}")
    print(
        f"{len(view_data['solutions'])} solutions, {len(view_data['pairs'])} token pairs"
    )
    for solution in view_data["solutions"]:
        marks = []
        if solution["isWinner"]:
            marks.append("winner")
        if solution["filteredOutApi"]:
            marks.append("filtered (api)")
        if solution["index"] in analysis["filteredIndices"]:
            marks.append("filtered (local)")
        if solution["index"] in analysis["winnerIndices"]:
            marks.append("winner (local)")
        surplus_eth = int(solution["surplusTotal"]) / 10**18
        score_eth = int(solution["scoreApi"]) / 10**18
        solver = view_data["solverNames"].get(
            solution["solver"].lower(), solution["solver"][:10] + "…"
        )
        print(
            f"  #{solution['ranking']:>2} {solver:<20.20} "
            f"score {score_eth:.6f} surplus {surplus_eth:.6f} {' '.join(marks)}"
        )


def main() -> None:
    """Run the auction inspector CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference",
        help="transaction hash (0x…, 66 characters), auction id, or 'latest'",
    )
    parser.add_argument(
        "--network",
        default="mainnet",
        help=f"network slug of the CoW API ({', '.join(NETWORKS)})",
    )
    parser.add_argument("--output", help="output HTML file path")
    parser.add_argument(
        "--no-open", action="store_true", help="do not open the result in a browser"
    )
    args = parser.parse_args()

    try:
        competition = fetch_competition(args.network, args.reference)
    except requests.HTTPError as error:
        parser.error(f"could not fetch competition data: {error}")
    order_uids = list(
        {
            order_execution["id"]
            for solution_data in competition["solutions"]
            for order_execution in solution_data["orders"]
        }
    )
    orders = fetch_orders(args.network, order_uids)
    chain_id = NETWORKS.get(args.network, NETWORKS["mainnet"])["chain_id"]
    token_info = fetch_token_info(chain_id)
    solver_names = fetch_solver_names(chain_id)

    solutions, surpluses = build_solutions(competition, orders)
    analysis = analyze_mechanism(solutions)
    view_data = assemble_view_data(
        args.network,
        competition,
        orders,
        solutions,
        surpluses,
        analysis,
        token_info,
        solver_names,
    )

    output_path = Path(
        args.output or f"auction_{args.network}_{competition['auctionId']}.html"
    )
    output_path.write_text(render_html(view_data), encoding="utf-8")
    print_summary(view_data)
    print(f"written to {output_path}")
    if not args.no_open:
        webbrowser.open(output_path.resolve().as_uri())


if __name__ == "__main__":
    main()
