r'''
The language a condition node is written in as an ODE: `x' = r`, integrated by a contract.

Dependency:

ode_condition                    (Dt, x' = r from x0; OdeCondition, its states with their
                                  rates and initial values over a span, `+` the join; is_ode)
 ||
 \/
contracts                        (Times, the data; Trajectory, the report;
                                  Integrator, the level; Integration, its final, `trajectory`
                                  its flag; CONTRACTS, the default)

Imported by name, not here: `contracts`, and what the language tells each consumer
(`consumers/`).

'''

from .ode_condition import Dt, OdeCondition, is_ode

__all__ = ['Dt', 'OdeCondition', 'is_ode']
