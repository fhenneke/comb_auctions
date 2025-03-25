import argparse

from data_fetching import fetch_auctions, compute_split_solutions
from experiment_helpers import run_counter_factual_analysis, compute_statistics
from mechanism import (
    FilterRankRewardMechanism,
    NoFilter,
    BaselineFilter,
    DirectedTokenPairs,
    DirectSelection,
    MonotoneSelection,
    SubsetFilteringSelection,
    ReferenceReward,
    SingleSurplusSelection,
)

def main():
    """Main function to run the counterfactual analysis."""
    parser = argparse.ArgumentParser(
        description="Run counterfactual analysis on auction solutions.")
    parser.add_argument("--auction_start", type=int, default=10322553 - 50000,
                        help="Start block for fetching auctions (default: 10322553 - 50000)")
    parser.add_argument("--auction_end", type=int, default=10322553,
                        help="End block for fetching auctions (default: 10322553)")
    parser.add_argument("--efficiency_loss", type=float, default=0.01,
                        help="Efficiency loss parameter (default: 0.01)")
    parser.add_argument("--approach", type=str, default="complete",
                        help="Approach type for solution splitting (default: complete)")
    parser.add_argument("--reward_upper_cap", type=int, default=12 * 10 ** 15,
                        help="Upper cap for rewards in wei (default: 12 * 10^15)")
    parser.add_argument("--reward_lower_cap", type=int, default=10 ** 16,
                        help="Lower cap for rewards in wei (default: 10^16)")

    args = parser.parse_args()

    # fetch auctions and split
    # this can take around 1 minute the first time it is run and creates a file of 80MB
    auction_start = args.auction_start
    auction_end = args.auction_end
    print(f"Fetching auctions from {auction_start} to {auction_end}...")
    solutions_batch = fetch_auctions(auction_start, auction_end)
    efficiency_loss = args.efficiency_loss
    approach = args.approach
    print(
        f"Splitting solutions with efficiency loss {efficiency_loss} "
        f"and approach \"{approach}\"..."
    )
    solutions_batch_split = [
        compute_split_solutions(solutions, efficiency_loss=efficiency_loss, approach=approach)
        for solutions in solutions_batch
    ]

    # compare 3 mechanisms
    # 1. current, single winner
    # 2. currently implemented multiple winners, with less restrictive filtering
    # 3. slightly more sophisticated selection of multiple winners
    reward_cap_upper = args.reward_upper_cap
    reward_cap_lower = args.reward_lower_cap
    print(f"Using reward caps of {reward_cap_upper / 10 ** 18} and {reward_cap_lower / 10 ** 18}")

    mechanisms = [
        # our current mechanism
        FilterRankRewardMechanism(
            NoFilter(),
            DirectSelection(SingleSurplusSelection()),
            ReferenceReward(DirectSelection(SingleSurplusSelection()), reward_cap_upper,
                            reward_cap_lower),
        ),
        # greedy choice of batches by surplus, with fairness filtering
        FilterRankRewardMechanism(
            BaselineFilter(),
            DirectSelection(
                SubsetFilteringSelection(
                    filtering_function=DirectedTokenPairs(), cumulative_filtering=False
                )
            ),
            ReferenceReward(DirectSelection(
                SubsetFilteringSelection(
                    filtering_function=DirectedTokenPairs(), cumulative_filtering=False
                )
            ), reward_cap_upper, reward_cap_lower),
        ),
        # greedy choice of batches by surplus, in iteration checking for positive rewards
        # with fairness filtering
        FilterRankRewardMechanism(
            BaselineFilter(),
            MonotoneSelection(
                SubsetFilteringSelection(
                    filtering_function=DirectedTokenPairs(), cumulative_filtering=False
                )
            ),
            ReferenceReward(DirectSelection(
                SubsetFilteringSelection(
                    filtering_function=DirectedTokenPairs(), cumulative_filtering=False
                )
            ), reward_cap_upper, reward_cap_lower),
        )
    ]

    print("Running counterfactual analysis...")
    all_winners_results = run_counter_factual_analysis(solutions_batch_split, mechanisms)

    compute_statistics(solutions_batch_split, all_winners_results)


if __name__ == "__main__":
    main()
