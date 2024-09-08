"""This is a python script for computing winners and rewards for combinatorial auctions"""

from data_fetching import fetch_solutions
from mechanism import (
    Solution,
    GeneralMechanism,
    NoFilter,
    BaselineFilter,
    SingleWinner,
    MultipleWinners,
    BatchSecondPriceReward,
    TokenPairImprovementReward,
)


solutions = [
    Solution(
        id="batch winner", solver="solver 1", scores={("A", "B"): 100, ("A", "C"): 50}
    ),
    Solution(
        id="unfair batch", solver="solver 2", scores={("A", "B"): 50, ("A", "C"): 60}
    ),
    Solution(
        id="overlapping batch",
        solver="solver 3",
        scores={("B", "A"): 50, ("A", "C"): 50},
    ),
    Solution(
        id="non-overlapping batch",
        solver="solver 4",
        scores={("B", "A"): 40, ("D", "E"): 50},
    ),
    Solution(
        id="non-overlapping batch unfair",
        solver="solver 5",
        scores={("B", "A"): 20, ("D", "E"): 100},
    ),
    Solution(id="reference A->B", solver="solver 1", scores={("A", "B"): 80}),
    Solution(id="runner up A->B", solver="solver 2", scores={("A", "B"): 40}),
    Solution(id="reference A->C", solver="solver 2", scores={("A", "C"): 40}),
    Solution(id="runner up A->C", solver="solver 1", scores={("A", "C"): 40}),
    Solution(id="reference B->A", solver="solver 7", scores={("B", "A"): 30}),
    Solution(id="reference F->G", solver="solver 8", scores={("F", "G"): 50}),
    Solution(id="runner up F->G", solver="solver 1", scores={("F", "G"): 40}),
    Solution(id="reference H->I", solver="solver 8", scores={("H", "I"): 50}),
]

if __name__ == "__main__":
    tx_hash = "0x659a6b86aa25c01ba6bc65d63c4204a962f91073767372aa59d89e340aec219b"
    solutions = fetch_solutions(tx_hash, efficiency_loss=0.01)
    print(solutions)
    for mechanism in [
        GeneralMechanism(
            NoFilter(),
            SingleWinner(),
            BatchSecondPriceReward(12 * 10**15, 10**16),
        ),
        GeneralMechanism(
            NoFilter(),
            MultipleWinners(),
            BatchSecondPriceReward(12 * 10**15, 10**16),
        ),
        GeneralMechanism(
            BaselineFilter(),
            MultipleWinners(),
            TokenPairImprovementReward(12 * 10**15, 10**16, True),
        ),
        GeneralMechanism(
            BaselineFilter(),
            MultipleWinners(),
            TokenPairImprovementReward(12 * 10**15, 10**16, False),
        ),
    ]:
        winners_rewards = mechanism.winners_and_rewards(solutions)
        print(winners_rewards)
