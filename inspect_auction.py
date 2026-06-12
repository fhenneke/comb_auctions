"""Inspect a solver competition and visualize it in a small HTML UI.

Fetches competition data from the CoW Protocol API (by auction id or transaction
hash), computes surplus per trade via mechanism.compute_surplus_score,
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
from typing import Any, NamedTuple

import requests

from mechanism import (
    BaselineFilter,
    Solution,
    SubsetFilteringSelection,
    Trade,
    aggregate_scores,
    compute_baseline_solutions,
    compute_surplus_score,
    compute_total_score,
)

API_BASE = "https://api.cow.fi/{network}/api"
COW_EXPLORER_BASE = "https://explorer.cow.fi/"
TOKEN_LIST_URL = "https://files.cow.fi/tokens/CowSwap.json"
SOLVER_NETWORKS_URL = "https://cms.cow.fi/api/solver-networks"
BLOCKSCOUT_TOKEN_URL = "https://{slug}.blockscout.com/api/v2/tokens/{address}"
CHAIN_REGISTRY_URL = "https://chainid.network/chains.json"
NATIVE_TOKEN_ADDRESS = "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
REQUEST_TIMEOUT = 30

NETWORKS: dict[str, dict[str, Any]] = {
    "mainnet": {
        "chain_id": 1,
        "blockscout": "eth",
        "explorer": "https://etherscan.io",
        "cow_explorer_prefix": "",
        "native_symbol": "ETH",
    },
    "xdai": {
        "chain_id": 100,
        "blockscout": "gnosis",
        "explorer": "https://gnosisscan.io",
        "cow_explorer_prefix": "gc/",
        "native_symbol": "xDAI",
    },
    "arbitrum_one": {
        "chain_id": 42161,
        "blockscout": "arbitrum",
        "explorer": "https://arbiscan.io",
        "cow_explorer_prefix": "arb1/",
        "native_symbol": "ETH",
    },
    "base": {
        "chain_id": 8453,
        "blockscout": "base",
        "explorer": "https://basescan.org",
        "cow_explorer_prefix": "base/",
        "native_symbol": "ETH",
    },
    "polygon": {
        "chain_id": 137,
        "blockscout": "polygon",
        "explorer": "https://polygonscan.com",
        "cow_explorer_prefix": "pol/",
        "native_symbol": "POL",
    },
    "avalanche": {
        "chain_id": 43114,
        "blockscout": None,
        "explorer": "https://snowtrace.io",
        "cow_explorer_prefix": "avax/",
        "native_symbol": "AVAX",
    },
    "sepolia": {
        "chain_id": 11155111,
        "blockscout": "eth-sepolia",
        "explorer": "https://sepolia.etherscan.io",
        "cow_explorer_prefix": "sepolia/",
        "native_symbol": "ETH",
    },
}


def fetch_competition(network: str, reference: str) -> dict[str, Any]:
    """Fetch competition data by auction id, settlement tx hash, or 'latest'."""
    base = API_BASE.format(network=network) + "/v2/solver_competition"
    if reference.startswith("0x") and len(reference) == 66:
        url = f"{base}/by_tx_hash/{reference}"
    elif reference == "latest":
        url = f"{base}/latest"
    elif reference.isdigit():
        url = f"{base}/{reference}"
    else:
        raise ValueError(
            f"invalid reference {reference!r}: expected a transaction hash"
            " (0x…, 66 characters), an auction id, or 'latest'"
        )
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


def _decode_eth_call_string(hex_result: str) -> str | None:
    """Decode a string returned by eth_call (ABI string or legacy bytes32)."""
    raw = bytes.fromhex(hex_result.removeprefix("0x"))
    if not raw:
        return None
    try:
        if len(raw) >= 64:
            offset = int.from_bytes(raw[:32])
            length = int.from_bytes(raw[offset : offset + 32])
            value = raw[offset + 32 : offset + 32 + length]
        else:
            value = raw.rstrip(b"\x00")
        return value.decode("utf-8", errors="replace") or None
    except (IndexError, ValueError):
        return None


def _fetch_tokens_via_rpc(chain_id: int, addresses: list[str]) -> dict[str, dict]:
    """Fetch token symbol/decimals on-chain via a public RPC.

    The RPC endpoint is discovered through the chainid.network registry, so no
    per-network RPC configuration is needed; unresolved tokens are omitted.
    """
    symbol_selector, decimals_selector = "0x95d89b41", "0x313ce567"

    def eth_call(rpc: str, to: str, data: str) -> str | None:
        response = requests.post(
            rpc,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_call",
                "params": [{"to": to, "data": data}, "latest"],
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("result")

    try:
        chains = requests.get(CHAIN_REGISTRY_URL, timeout=REQUEST_TIMEOUT).json()
        rpcs = [
            url
            for chain in chains
            if chain["chainId"] == chain_id
            for url in chain["rpc"]
            if url.startswith("https://") and "${" not in url
        ]
    except (requests.RequestException, KeyError, ValueError):
        return {}

    found: dict[str, dict] = {}
    for rpc in rpcs:
        try:
            for address in addresses:
                if address in found:
                    continue
                symbol_hex = eth_call(rpc, address, symbol_selector)
                decimals_hex = eth_call(rpc, address, decimals_selector)
                if not symbol_hex or not decimals_hex:
                    continue
                symbol = _decode_eth_call_string(symbol_hex)
                decimals = int(decimals_hex, 16)
                if symbol is not None and 0 <= decimals <= 77:
                    found[address] = {"symbol": symbol, "decimals": decimals}
            if len(found) == len(addresses):
                return found
        except (requests.RequestException, ValueError):
            continue  # try the next public RPC
    return found


def fetch_missing_token_info(
    network_config: dict[str, Any], addresses: list[str]
) -> dict[str, dict]:
    """Fetch symbol/decimals for tokens missing from the CoW token list.

    Tries the Blockscout explorer API first (if the network has an instance),
    then falls back to on-chain lookups via a public RPC. Tokens that cannot
    be resolved are omitted and the viewer shows raw atom amounts for them.
    """
    found: dict[str, dict] = {}
    slug = network_config["blockscout"]
    if slug is not None:

        def fetch_one(address: str) -> dict | None:
            try:
                response = requests.get(
                    BLOCKSCOUT_TOKEN_URL.format(slug=slug, address=address),
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                token = response.json()
                return {
                    "symbol": token["symbol"],
                    "decimals": int(token["decimals"]),
                }
            except (requests.RequestException, KeyError, TypeError, ValueError):
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(fetch_one, addresses)
        found = {
            address: info for address, info in zip(addresses, results) if info
        }

    unresolved = [address for address in addresses if address not in found]
    if unresolved:
        found |= _fetch_tokens_via_rpc(network_config["chain_id"], unresolved)
        unresolved = [address for address in addresses if address not in found]
    if unresolved:
        print(f"warning: no token info found for {', '.join(unresolved)}")
    return found


def fetch_solver_names(chain_id: int) -> dict[str, str]:
    """Fetch submission address -> solver display name from the CoW CMS.

    Uses the same data source as the CoW explorer ("solved by" on order pages);
    returns an empty dict on failure (with a warning, addresses are shown instead).
    """
    for _ in range(2):
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
            return names
        except (requests.RequestException, KeyError, ValueError) as error:
            last_error = error
    print(f"warning: could not fetch solver names ({last_error})")
    return {}


def compute_trade_surplus(
    order: dict | None,
    executed_sell: int,
    executed_buy: int,
    native_prices: dict[str, int],
) -> int | None:
    """Compute the surplus of a trade in atoms of the native token.

    Applies mechanism.compute_surplus_score to order data and native prices
    coming from the API instead of the database. Returns None if the order is
    unknown (JIT order) or no native price is available for the buy token.
    """
    if order is None:
        return None
    buy_token_price = native_prices.get(order["buyToken"].lower())
    if buy_token_price is None:
        return None
    return compute_surplus_score(
        kind=order["kind"],
        limit_sell=int(order["sellAmount"]),
        limit_buy=int(order["buyAmount"]),
        executed_sell=executed_sell,
        executed_buy=executed_buy,
        buy_token_price=buy_token_price,
    )


def compute_effective_quote(order: dict | None) -> tuple[int, int] | None:
    """Effective (sell amount, buy amount) of the order's quote.

    The quote on the order endpoint is given as sell amount, buy amount, and fee
    amount; the fee is folded into the amounts to make them comparable to
    executions: for sell orders the buy amount becomes
    (sell - fee) * buy / sell, for buy orders the sell amount becomes
    sell + fee. (Volume fees are not corrected for yet.)
    """
    if order is None or not order.get("quote"):
        return None
    quote = order["quote"]
    quote_sell = int(quote["sellAmount"])
    quote_buy = int(quote["buyAmount"])
    fee = int(quote["feeAmount"])
    if order["kind"] == "sell":
        if quote_sell == 0:
            return None
        return quote_sell, (quote_sell - fee) * quote_buy // quote_sell
    return quote_sell + fee, quote_buy


def compute_trade_quote_score(
    order: dict | None,
    executed_sell: int,
    executed_buy: int,
    native_prices: dict[str, int],
) -> int | None:
    """Surplus the trade would have had if executed at the quoted price.

    Uses the same fill size as the actual execution, with the counterpart
    amount taken from the effective quote price; can be negative if the quote
    is below the limit price.
    """
    effective = compute_effective_quote(order)
    if effective is None:
        return None
    effective_sell, effective_buy = effective
    assert order is not None
    if order["kind"] == "sell":
        if effective_sell == 0:
            return None
        synthetic_buy = math.floor(
            Fraction(executed_sell * effective_buy, effective_sell)
        )
        return compute_trade_surplus(order, executed_sell, synthetic_buy, native_prices)
    if effective_buy == 0:
        return None
    synthetic_sell = math.ceil(Fraction(executed_buy * effective_sell, effective_buy))
    return compute_trade_surplus(order, synthetic_sell, executed_buy, native_prices)


def pair_key(sell_token: str, buy_token: str) -> str:
    """Serialize a directed token pair into a string key for use in JSON."""
    return f"{sell_token}|{buy_token}"


class TradeScores(NamedTuple):
    """Surplus and quote-based score of a trade; None marks unknown values."""

    surplus: int | None
    quote_score: int | None


def build_solutions(
    competition: dict[str, Any], orders: dict[str, dict | None]
) -> tuple[list[Solution], list[dict[str, TradeScores]]]:
    """Build mechanism.Solution objects with surplus-based scores.

    Solution ids are list indices into competition["solutions"]; trades with
    unknown surplus (JIT orders, missing prices) enter with score 0. The second
    return value contains the trade scores per order uid for each solution.
    """
    native_prices = {
        token.lower(): int(price)
        for token, price in competition["auction"]["prices"].items()
    }
    solutions = []
    surpluses: list[dict[str, TradeScores]] = []
    for index, solution_data in enumerate(competition["solutions"]):
        trades = []
        trade_surpluses: dict[str, TradeScores] = {}
        for order_execution in solution_data["orders"]:
            uid = order_execution["id"]
            surplus = compute_trade_surplus(
                orders[uid],
                int(order_execution["sellAmount"]),
                int(order_execution["buyAmount"]),
                native_prices,
            )
            quote_score = compute_trade_quote_score(
                orders[uid],
                int(order_execution["sellAmount"]),
                int(order_execution["buyAmount"]),
                native_prices,
            )
            trade_surpluses[uid] = TradeScores(surplus, quote_score)
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


def analyze_mechanism(solutions: list[Solution]) -> dict[str, Any]:
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
    violations: dict[int, list[dict]] = {}
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
        violations[int(solution.id)] = solution_violations

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
        "violations": violations,
        "winnerIndices": [int(winner.id) for winner in winners],
        "winnersTotal": str(winners_total),
        "references": references,
    }


def opt_str(value: int | None) -> str | None:
    """Serialize an optional big integer for JSON (None stays None)."""
    return str(value) if value is not None else None


def assemble_view_data(
    network: str,
    competition: dict[str, Any],
    orders: dict[str, dict | None],
    solutions: list[Solution],
    surpluses: list[dict[str, TradeScores]],
    analysis: dict[str, Any],
    token_info: dict[str, dict],
    solver_names: dict[str, str],
) -> dict[str, Any]:
    """Assemble the JSON object embedded into the HTML template."""
    network_config = NETWORKS[network]
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
            trade_scores = surpluses[index][trade.id]
            trades[trade.id] = {
                "sellAmount": order_execution["sellAmount"],
                "buyAmount": order_execution["buyAmount"],
                "surplus": opt_str(trade_scores.surplus),
                "quoteSurplus": opt_str(trade_scores.quote_score),
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
                    "key": key,
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

    order_views: dict[str, dict[str, Any]] = {}
    for uid, order in orders.items():
        if order is None:
            order_views[uid] = {"found": False}
            continue
        effective_quote = compute_effective_quote(order)
        order_views[uid] = {
            "found": True,
            "kind": order["kind"],
            "class": order["class"],
            "limitSell": order["sellAmount"],
            "limitBuy": order["buyAmount"],
            "sellToken": order["sellToken"].lower(),
            "buyToken": order["buyToken"].lower(),
            "quoteSell": opt_str(effective_quote[0]) if effective_quote else None,
            "quoteBuy": opt_str(effective_quote[1]) if effective_quote else None,
        }

    return {
        "network": network,
        "nativeSymbol": network_config["native_symbol"],
        "explorerBase": network_config["explorer"],
        "cowExplorerOrderBase": COW_EXPLORER_BASE
        + network_config["cow_explorer_prefix"]
        + "orders/",
        "auctionId": competition["auctionId"],
        "auctionStartBlock": competition.get("auctionStartBlock"),
        "auctionDeadlineBlock": competition.get("auctionDeadlineBlock"),
        "transactionHashes": competition.get("transactionHashes", []),
        "referenceScoresApi": competition.get("referenceScores", {}),
        "nativePrices": {
            token.lower(): str(price)
            for token, price in competition["auction"]["prices"].items()
        },
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
    # escape "</" so external strings (token symbols etc.) cannot terminate the
    # script tag the data is embedded in
    data_json = json.dumps(view_data).replace("</", r"<\/")
    return template.replace("__DATA_JSON__", data_json)


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
        ranking = "–" if solution["ranking"] is None else solution["ranking"]
        print(
            f"  #{ranking:>2} {solver:<20.20} "
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
        choices=list(NETWORKS),
        help="network slug of the CoW API",
    )
    parser.add_argument("--output", help="output HTML file path")
    parser.add_argument(
        "--no-open", action="store_true", help="do not open the result in a browser"
    )
    args = parser.parse_args()

    try:
        competition = fetch_competition(args.network, args.reference)
    except (requests.HTTPError, ValueError) as error:
        parser.error(f"could not fetch competition data: {error}")
    order_uids = list(
        {
            order_execution["id"]
            for solution_data in competition["solutions"]
            for order_execution in solution_data["orders"]
        }
    )
    orders = fetch_orders(args.network, order_uids)
    chain_id = NETWORKS[args.network]["chain_id"]
    token_info = fetch_token_info(chain_id)
    traded_tokens = {
        order_execution[field].lower()
        for solution_data in competition["solutions"]
        for order_execution in solution_data["orders"]
        for field in ("sellToken", "buyToken")
    }
    # the native token sentinel gets its info in assemble_view_data
    missing_tokens = sorted(traded_tokens - set(token_info) - {NATIVE_TOKEN_ADDRESS})
    if missing_tokens:
        token_info |= fetch_missing_token_info(NETWORKS[args.network], missing_tokens)
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
