"""This is a python script for computing winners and rewards for combinatorial auctions"""

from data_fetching import fetch_solutions_single, fetch_solutions_batch
from mechanism import (
    Trade,
    Solution,
    FilterRankRewardMechanism,
    NoFilter,
    BaselineFilter,
    TokenPairs,
    TradedTokens,
    SingleWinner,
    TokenPairFilteringWinners,
    SubsetFilteringWinners,
    BatchSecondPriceReward,
    BatchOverlapSecondPriceReward,
    TokenPairImprovementReward,
)


solutions_batch = [
    [  # batch vs single order solutions
        Solution(
            "batch winner",
            solver="solver 1",
            score=200,
            trades=[Trade("1", "A", "B", 100), Trade("2", "C", "D", 100)],
        ),
        Solution(
            "best on first trade",
            solver="solver 2",
            score=150,
            trades=[Trade("1", "A", "B", 150)],
        ),
        Solution(
            "best on second trade",
            solver="solver 3",
            score=150,
            trades=[Trade("2", "C", "D", 150)],
        ),
    ],
    [  # solutions without overlap
        Solution(
            "best on first trade",
            solver="solver 1",
            score=150,
            trades=[Trade("1", "A", "B", 150)],
        ),
        Solution(
            "best on second trade",
            solver="solver 2",
            score=140,
            trades=[Trade("2", "C", "D", 140)],
        ),
        Solution(
            "bad batch",
            solver="solver 3",
            score=100,
            trades=[Trade("1", "A", "B", 50), Trade("2", "C", "D", 50)],
        ),
    ],
    [  # batch in between solutions without overlap
        Solution(
            "best on first trade",
            solver="solver 1",
            score=150,
            trades=[Trade("1", "A", "B", 150)],
        ),
        Solution(
            "batch with overlap",
            solver="solver 3",
            score=100,
            trades=[Trade("1", "A", "B", 50), Trade("2", "C", "D", 50)],
        ),
        Solution(
            "best on second trade",
            solver="solver 2",
            score=90,
            trades=[Trade("2", "C", "D", 90)],
        ),
    ],
    [  # reference is not from winner
        Solution(
            "batch with overlap",
            solver="solver 1",
            score=200,
            trades=[Trade("1", "A", "B", 150), Trade("2", "C", "D", 50)],
        ),
        Solution(
            "best on first trade",
            solver="solver 1",
            score=100,
            trades=[Trade("1", "A", "B", 100)],
        ),
        Solution(
            "best on second trade",
            solver="solver 2",
            score=90,
            trades=[Trade("2", "C", "D", 90)],
        ),
    ],
    [  # token overlap but not on the same token pair
        Solution(
            "batch with overlap",
            solver="solver 1",
            score=100,
            trades=[Trade("1", "A", "B", 100)],
        ),
        Solution(
            "best on first trade",
            solver="solver 2",
            score=90,
            trades=[Trade("1", "A", "C", 90)],
        ),
    ],
    [
        Solution(
            id="batch winner",
            solver="solver 1",
            score=150,
            trades=[Trade("1", "A", "B", 100), Trade("2", "A", "C", 50)],
        ),
        Solution(
            id="unfair batch",
            solver="solver 2",
            score=110,
            trades=[Trade("1", "A", "B", 50), Trade("2", "A", "C", 60)],
        ),
        Solution(
            id="overlapping batch",
            solver="solver 3",
            score=100,
            trades=[Trade("3", "B", "A", 50), Trade("2", "A", "C", 50)],
        ),
        Solution(
            id="non-overlapping batch",
            solver="solver 4",
            score=100,
            trades=[Trade("3", "B", "A", 40), Trade("4", "D", "E", 60)],
        ),
        Solution(
            id="non-overlapping batch unfair",
            solver="solver 5",
            score=120,
            trades=[Trade("3", "B", "A", 20), Trade("4", "D", "E", 100)],
        ),
        Solution(
            id="reference A->B",
            solver="solver 1",
            score=80,
            trades=[Trade("1", "A", "B", 80)],
        ),
        Solution(
            id="reference A->C",
            solver="solver 2",
            score=40,
            trades=[Trade("2", "A", "C", 40)],
        ),
        Solution(
            id="runner up A->B",
            solver="solver 2",
            score=40,
            trades=[Trade("1", "A", "B", 40)],
        ),
        Solution(
            id="runner up A->C",
            solver="solver 1",
            score=40,
            trades=[Trade("2", "A", "C", 40)],
        ),
        Solution(
            id="reference B->A",
            solver="solver 7",
            score=30,
            trades=[Trade("3", "B", "A", 30)],
        ),
        Solution(
            id="reference F->G",
            solver="solver 8",
            score=50,
            trades=[Trade("5", "F", "G", 50)],
        ),
        Solution(
            id="runner up F->G",
            solver="solver 1",
            score=40,
            trades=[Trade("5", "F", "G", 40)],
        ),
        Solution(
            id="reference H->I",
            solver="solver 8",
            score=50,
            trades=[Trade("6", "H", "I", 50)],
        ),
    ],
]

if __name__ == "__main__":
    # solutions_batch = [
    #     fetch_solutions_single(
    #         "0x659a6b86aa25c01ba6bc65d63c4204a962f91073767372aa59d89e340aec219b",
    #         split_solutions=True,
    #         efficiency_loss=0.01,
    #     )
    # ]
    solutions_batch = fetch_solutions_batch(9534992 - 1000, 9534992)

    print(f"number of auctions: {len(solutions_batch)}")

    mechanisms = [
        FilterRankRewardMechanism(
            NoFilter(),
            SingleWinner(),
            BatchSecondPriceReward(12 * 10**15, 10**16),
        ),
        # FilterRankRewardMechanism(
        #     NoFilter(),
        #     TokenPairFilteringWinners(),
        #     BatchSecondPriceReward(12 * 10**15, 10**16),
        # ),
        FilterRankRewardMechanism(
            NoFilter(),
            SubsetFilteringWinners(
                filtering_function=TradedTokens(), cumulative_filtering=True
            ),
            BatchOverlapSecondPriceReward(12 * 10**15, 10**16, TradedTokens()),
        ),
        # FilterRankRewardMechanism(
        #     BaselineFilter(),
        #     TokenPairFilteringWinners(),
        #     TokenPairImprovementReward(12 * 10**15, 10**16, True),
        # ),
        # FilterRankRewardMechanism(
        #     BaselineFilter(),
        #     TokenPairFilteringWinners(),
        #     TokenPairImprovementReward(12 * 10**15, 10**16, False),
        # ),
    ]
    all_rewards: list[list[dict[str, tuple[int, int]]]] = []
    for solutions in solutions_batch:
        print(solutions)

        rewards = [mechanism.winners_and_rewards(solutions) for mechanism in mechanisms]
        print(rewards)

        all_rewards.append(rewards)

    print(all_rewards)

