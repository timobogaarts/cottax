'''
What makes an answer right, read off the relations. Each predicate answers False for a
node that is not a `RelationalCondition`.
'''

from .relational_condition import Eq, Le, RelationalCondition


def _determines(node: object) -> bool:
    '''A relational condition with unknowns: what a shape is a shape *of*.'''
    return isinstance(node, RelationalCondition) and bool(node.unknowns)


# ---------------------------------------------------------------- the two levels

def is_optimise(node: object) -> bool:
    '''Exactly one objective. `is_unconstrained_optimise` / `is_constrained_optimise`.'''
    return _determines(node) and len(node.objectives) == 1


def is_equalities(node: object) -> bool:
    '''No objective, every relation an equality. `is_fixed_point` / `is_root_find`.'''
    return (_determines(node) and not node.objectives and bool(node.relations)
            and all(r.op is Eq for r in node.relations))


# ---------------------------------------------------------------- the five finals

def is_unconstrained_optimise(node: object) -> bool:
    '''An optimise with no relation beside the objective.'''
    return is_optimise(node) and not node.relations


def is_constrained_optimise(node: object) -> bool:
    '''An optimise with relations to respect as well.'''
    return is_optimise(node) and bool(node.relations)


def is_feasibility(node: object) -> bool:
    '''No objective and an inequality among the relations: any point satisfying them
    will do.'''
    return (_determines(node) and not node.objectives
            and any(r.op is Le for r in node.relations))


def is_fixed_point(node: object) -> bool:
    '''Equalities of which every one is `u = g` (`fixed_point_shaped`).'''
    return is_equalities(node) and fixed_point_shaped(node.relations, node.unknowns)


def fixed_point_shaped(relations: tuple, unknowns: tuple) -> bool:
    '''Whether every relation is `u = g`: one of `unknowns` on the left, and a place on
    the right.'''
    return all(r.rhs is not None and r.lhs in unknowns for r in relations)


def is_root_find(node: object) -> bool:
    '''Every other system of equalities: `a(x) = b(x)`, which is `c(x) = 0` WLOG.'''
    return is_equalities(node) and not is_fixed_point(node)
