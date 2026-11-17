r'''
The language a condition node is written in as a design of experiments: each factor set to
every point of its plan, each response recorded there, answered by a contract.
Dependency:
doe_condition                    (Varied, a factor over its plan; Recorded, a response into
                                  its record; DoeCondition, the factors and the responses;
                                  is_doe)
 ||
 \/
contracts                        (Sampler, the level; Sampling, its final; CONTRACTS, the
                                  default)
Imported by name, not here: `contracts`, and what the language tells each consumer
(`consumers/`).
'''

from .doe_condition import DoeCondition, Recorded, Varied, is_doe

__all__ = ['DoeCondition', 'Recorded', 'Varied', 'is_doe']
