# Combinatorial auctions on CoW Swap

Code for testing different implementations of combinatorial auctions.

## Installing dependencies

Installing dependencies is easiest via [uv](https://docs.astral.sh/uv/). Just run the script or the jupyter notebook and dependencies will be installed automatically into the environment .venv.

## Data fetching

If you want to test historical auctions, you need to provide database credentials in a `.env` file, see `.env.example`.

At the moment, the data fetching can take 1 minute the first time it is run. Data is then stored in a 80MB file and not fetched again (for the same range of auctions).

If you have git-lfs installed, a set of auctions should be automatically downloaded with the repo. Otherwise, you can download a set of auctions from the release page. In this way you can run the experiment without access to the database.

## Inspecting a single auction

To inspect a single solver competition in an interactive HTML view, run

```sh
uv run inspect_auction.py <auction id | settlement tx hash | latest> [--network mainnet]
```

This fetches the competition data from the [CoW Protocol API](https://api.cow.fi/docs/)
(no database credentials needed), computes surplus per trade, reruns the auction
mechanism from `mechanism.py` on surplus-based scores (baseline solutions, fairness
filtering, winner selection, reference scores), and writes a self-contained HTML file
which opens in the browser. The view shows a matrix of trades (rows, grouped by
directed token pair) against solutions (columns, ranked) with winners, baseline
solutions, and filtered solutions marked, including which trade caused a solution to
be filtered. Clicking a winner highlights its reference solutions and compares the
API reference score (score-based) with the locally recomputed one (surplus-based).

Note on scores: ranking in the protocol is by score = surplus + fees, but fees of
non-executed bids are not available from the API. The view therefore shows the API
score, the computed surplus, and their difference as implied fees; the local rerun
of the mechanism uses surplus only.

## Running the script


Running the script and notebook is easiest via [uv](https://docs.astral.sh/uv/).

The script for analysing historical auctions is run via
```sh
uv run counter_factual_analysis.py
```

Some parameters of the simulation can be modified using command line arguments.
```
usage: counter_factual_analysis.py [-h] [--auction_start AUCTION_START] [--auction_end AUCTION_END]
                                   [--efficiency_loss EFFICIENCY_LOSS] [--approach APPROACH]
                                   [--reward_upper_cap REWARD_UPPER_CAP] [--reward_lower_cap REWARD_LOWER_CAP]

Run counterfactual analysis on auction solutions.

options:
  -h, --help            show this help message and exit
  --auction_start AUCTION_START
                        Start block for fetching auctions (default: 10322553 - 50000)
  --auction_end AUCTION_END
                        End block for fetching auctions (default: 10322553)
  --efficiency_loss EFFICIENCY_LOSS
                        Efficiency loss parameter (default: 0.01)
  --approach APPROACH   Approach type for solution splitting (default: complete)
  --reward_upper_cap REWARD_UPPER_CAP
                        Upper cap for rewards in wei (default: 12 * 10^15)
  --reward_lower_cap REWARD_LOWER_CAP
                        Lower cap for rewards in wei (default: 10^16)
```
