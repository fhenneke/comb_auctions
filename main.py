"""This is a python script for computing winners and rewards for combinatorial auctions"""

from data_fetching import fetch_solutions
from mechanism import (
    Trade,
    Solution,
    AuctionMechanism,
    NoFilter,
    BaselineFilter,
    SingleWinner,
    MultipleWinners,
    BatchSecondPriceReward,
    TokenPairImprovementReward,
)


solutions = [
    Solution(
        id="batch winner",
        solver="solver 1",
        trades=[Trade("1", "A", "B", 100), Trade("2", "A", "C", 50)],
    ),
    Solution(
        id="unfair batch",
        solver="solver 2",
        trades=[Trade("1", "A", "B", 50), Trade("2", "A", "C", 60)],
    ),
    Solution(
        id="overlapping batch",
        solver="solver 3",
        trades=[Trade("3", "B", "A", 50), Trade("2", "A", "C", 50)],
    ),
    Solution(
        id="non-overlapping batch",
        solver="solver 4",
        trades=[Trade("3", "B", "A", 40), Trade("4", "D", "E", 60)],
    ),
    Solution(
        id="non-overlapping batch unfair",
        solver="solver 5",
        trades=[Trade("3", "B", "A", 20), Trade("4", "D", "E", 100)],
    ),
    Solution(id="reference A->B", solver="solver 1", trades=[Trade("1", "A", "B", 80)]),
    Solution(id="runner up A->B", solver="solver 2", trades=[Trade("1", "A", "B", 40)]),
    Solution(id="reference A->C", solver="solver 2", trades=[Trade("2", "A", "C", 40)]),
    Solution(id="runner up A->C", solver="solver 1", trades=[Trade("2", "A", "C", 40)]),
    Solution(id="reference B->A", solver="solver 7", trades=[Trade("3", "B", "A", 30)]),
    Solution(id="reference F->G", solver="solver 8", trades=[Trade("5", "F", "G", 50)]),
    Solution(id="runner up F->G", solver="solver 1", trades=[Trade("5", "F", "G", 40)]),
    Solution(id="reference H->I", solver="solver 8", trades=[Trade("6", "H", "I", 50)]),
]

if __name__ == "__main__":
    tx_hash = "0x659a6b86aa25c01ba6bc65d63c4204a962f91073767372aa59d89e340aec219b"
    solutions = fetch_solutions(tx_hash, efficiency_loss=0.01)
    print(solutions)
    for mechanism in [
        AuctionMechanism(
            NoFilter(),
            SingleWinner(),
            BatchSecondPriceReward(12 * 10**15, 10**16),
        ),
        AuctionMechanism(
            NoFilter(),
            MultipleWinners(),
            BatchSecondPriceReward(12 * 10**15, 10**16),
        ),
        AuctionMechanism(
            BaselineFilter(),
            MultipleWinners(),
            TokenPairImprovementReward(12 * 10**15, 10**16, True),
        ),
        AuctionMechanism(
            BaselineFilter(),
            MultipleWinners(),
            TokenPairImprovementReward(12 * 10**15, 10**16, False),
        ),
    ]:
        rewards = mechanism.winners_and_rewards(solutions)
        print(rewards)
        print(
            "total surplus: "
            f"{sum(solution.score() for solution in solutions if solution.id in rewards)}"
        )

    filtered_solutions = mechanism.solution_filter.filter(solutions)
    winners = mechanism.winner_selection.select_winners(filtered_solutions)
    rewards = mechanism.reward_mechanism.compute_rewards(winners, filtered_solutions)
    print(
        "total surplus: "
        f"{sum(solution.score() for solution in solutions if solution.id in rewards)}"
    )
