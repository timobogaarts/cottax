r'''
Running a realised graph in plain Python.

Dependency:

interface_types                  (what a driver's implementation is handed (Posed) and tries
 ||                               values through (ConditionMap))
 \/
interface      refusals          (interface: what a feature registers, of a place, the
 ||             ||                DataStructure for its class; what a study chooses, what
 \/             ||                runs a driver (DriverImplementation);
run <-----------'                 refusals: every refusal kind)
 ||                              (run: CallableGraph over realisation.RealisedGraph, every
 ||                               implementation callable, a driver's one of it; run,
 ||                               refusing a graph proved below a CallableGraph; walk, its
 ||                               loop over the steps with an Env and a Solve handed to an
 ||                               AtSolve; InteriorConditionMap,
 ||                               a problem posed by its interior)
 \/
jax_jit                          (JitGraph, every node a jit cache key; Traced, the run
                                  traced once: the graph's jaxpr, every place's shape)

interface, refusals
 ||
 \/
structures                       (Inputs, the key of a variable's own value; slots, each place
 ||                               a run takes a value at and what the caller calls it;
 ||                               read_from, each read where a caller's structure holds it;
 ||                               valued_once, one value per input; write_back,
 ||                               write_role)
 \/
drive                            (Drive: every undriven problem driven and implemented by the
 ||                               first implementation accepting it, its data and reports at
 ||                               the places a rule it is handed gives)
 \/
part <-- declaration.interface   (part: Driving, `drive` as a surface's method, its contracts
                                  and places the fields it holds; Running, a model valued and
                                  run by name or from a caller's structures)

Imported by name, not here: `jax_jit`; `run`'s `blamed` and `check_callable`.

'''

from .run import CallableGraph, run
from .structures import Inputs, read_from, slots, valued_once, write_back, write_role
from .drive import Drive
from .part import Driving, Results, Running

__all__ = ['CallableGraph', 'run', 'Inputs', 'read_from', 'slots', 'valued_once', 'write_back', 'write_role', 'Drive', 'Driving',
           'Running', 'Results']
