"""Registry of every arm in the study."""

from crossover.agents.multi import MultiAgent
from crossover.agents.react import ReactAgent
from crossover.agents.self_consistency import SelfConsistency
from crossover.agents.single import SingleAgent

SYSTEMS = {
    "L0": lambda: ReactAgent(summarize=False),
    "L0-sum": lambda: ReactAgent(summarize=True),
    "S-shared": lambda: SingleAgent(isolated=False),
    "S-iso": lambda: SingleAgent(isolated=True),
    "M-iso": lambda: MultiAgent(shared=False, parallel=True),
    "M-iso-seq": lambda: MultiAgent(shared=False, parallel=False),
    "M-shared": lambda: MultiAgent(shared=True, parallel=True),
    "SC": lambda: SelfConsistency(),
}


def make_system(name: str):
    if name not in SYSTEMS:
        raise KeyError(f"Unknown system {name!r}; choose from {sorted(SYSTEMS)}")
    return SYSTEMS[name]()
