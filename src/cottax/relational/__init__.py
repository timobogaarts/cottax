r'''
The language a condition node is written in: relations, objectives, and rewrites over them.

Dependency:

relational_condition             (Rel / Eq / Le, Relation, Sense / Min / Max, Objective,
 ||                               RelationalCondition, its `+`)
 ||----------------------------
 \/            \/             \/
rewrites      programme      classification   (rewrites: FixedPointCut, cuts closed by a
                              ||               problem on their copies;
                              \/               programme: Programme, a condition read as a
                             contracts         nonlinear programme, places only;
                                               classification: the shapes, as predicates;
                                               contracts: the shipped driver contracts
                                               and their defaults, CONTRACTS)

Imported by name, not here: `contracts`, and what the language tells each consumer
(`consumers/`), its closing among them (`consumers/architectures`).

'''

from .relational_condition import (
    Eq, Le, Max, Min, Objective, Rel, Relation, RelationalCondition, Sense,
)
from .classification import (
    is_constrained_optimise, is_equalities, is_feasibility, is_fixed_point,
    is_optimise, is_root_find,
    is_unconstrained_optimise,
)
from .rewrites import FixedPointCut
from .programme import Programme, programme

__all__ = [
    'Eq', 'Le', 'Rel', 'Relation', 'RelationalCondition', 'Sense', 'Min', 'Max', 'Objective',
    'is_constrained_optimise', 'is_equalities', 'is_feasibility', 'is_fixed_point',
    'is_optimise', 'is_root_find',
    'is_unconstrained_optimise',
    'FixedPointCut',
    'Programme', 'programme',
]
