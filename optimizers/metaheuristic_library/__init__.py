"""
Metaheuristic Optimizer Library — 24 pseudo-implementations.
Reference: Energy Conversion and Management, Vol 258, 115521 (2022)
           DOI: 10.1016/j.enconman.2022.115521

Usage:
    from metaheuristic_library import HoneyBadgerOptimizer
    opt = HoneyBadgerOptimizer(obj_func=my_func, dim=5, lb=-10, ub=10)
    best_pos, best_fit = opt.optimize()
"""

from .base import BaseOptimizer

# Animal-inspired
from .animal_optimizers import (
    HoneyBadgerOptimizer,       # HBO
    GreyWolfOptimizer,          # GWO
    WhaleOptimizer,             # WO / WOA
    SharkSmellOptimizer,        # SSO
    ChaoticHarrisHawkOptimizer, # CHHO
    CoyoteOptimizer,            # CO
    BonoboOptimizer,            # BO
)

# Marine/aquatic-inspired
from .marine_optimizers import (
    MantaRayForagingOptimizer,  # MRFO
    JellyfishSearchOptimizer,   # JSO
    MarinePredatorOptimizer,    # MPO
    TunicateSwarmOptimizer,     # TSO
)

# Nature/ecology-inspired
from .nature_optimizers import (
    GrasshopperOptimizer,                   # GO
    BlackWidowOptimizer,                    # BWO
    TreeSeedAlgorithm,                      # TSA
    SineTreeSeedAlgorithm,                  # STSA
    FlowerPollinationOptimizer,             # FPO
    SlimeMouldOptimizer,                    # SMO
    TreeGrowthOptimizer,                    # TGO
    ImprovedArtificialEcosystemOptimizer,   # IAEO
)

# Hybrid/mathematical
from .hybrid_optimizers import (
    GradientBasedOptimizer,             # GBO
    NeuralNetworkOptimizer,             # NNO
    PoliticalOptimizer,                 # PO
    PathfinderOptimizer,                # PFO
    EquilibriumSlimeMouldOptimizer,     # ESMO
)

# Lookup by abbreviation
OPTIMIZER_REGISTRY = {
    "HBO": HoneyBadgerOptimizer,
    "GWO": GreyWolfOptimizer,
    "WO": WhaleOptimizer,
    "WOA": WhaleOptimizer,
    "SSO": SharkSmellOptimizer,
    "CHHO": ChaoticHarrisHawkOptimizer,
    "CO": CoyoteOptimizer,
    "BO": BonoboOptimizer,
    "MRFO": MantaRayForagingOptimizer,
    "JSO": JellyfishSearchOptimizer,
    "MPO": MarinePredatorOptimizer,
    "TSO": TunicateSwarmOptimizer,
    "GO": GrasshopperOptimizer,
    "BWO": BlackWidowOptimizer,
    "TSA": TreeSeedAlgorithm,
    "STSA": SineTreeSeedAlgorithm,
    "FPO": FlowerPollinationOptimizer,
    "SMO": SlimeMouldOptimizer,
    "TGO": TreeGrowthOptimizer,
    "IAEO": ImprovedArtificialEcosystemOptimizer,
    "GBO": GradientBasedOptimizer,
    "NNO": NeuralNetworkOptimizer,
    "PO": PoliticalOptimizer,
    "PFO": PathfinderOptimizer,
    "ESMO": EquilibriumSlimeMouldOptimizer,
}


def get_optimizer(abbrev):
    """Get optimizer class by abbreviation (e.g. 'HBO', 'GWO')."""
    return OPTIMIZER_REGISTRY[abbrev.upper()]


def list_optimizers():
    """List all available optimizer abbreviations."""
    return list(OPTIMIZER_REGISTRY.keys())
