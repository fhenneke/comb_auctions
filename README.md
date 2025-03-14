# Combinatorial auctoins on CoW Swap

Code for testing different implementations of combinatorial auctions.

## Running the script

Running the notebook is easiest via [uv](https://docs.astral.sh/uv/). The jupyter notebook server can be started with

```sh
uv run jupyter notebook
```

From within jupyter, the notebook `multiple_winners.ipynb` can be selected and run.

At the moment, the data fetching can take 20 minutes the first time it is run. Data is then stored in a 80MB file and not fetched again (for the same range of auctions).
