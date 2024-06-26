from .rockmate import Rockmate
from . import solvers
from rkgb import partitioned

class PureRotor(Rockmate):
    @classmethod
    def solver(cls, force_python=False):
        return solvers.RK_rotor(force_python=force_python)

    @classmethod
    def partitioner(cls):
        return partitioned.PartitionerSequence(sub_partitioner=partitioned.Partitioner()) 

    def __init__(self, model, model_inputs, budget=None, rotor_solve_with_python=False, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        super().__init__(model, model_inputs, budget,
                         list_solvers=[self.solver(force_python=rotor_solve_with_python)],
                         partitioners=[self.partitioner()],
                         **kwargs)

class PureCheckmate(Rockmate):
    @classmethod
    def solver(cls):
        hilp_solver = solvers.HILP(ilp_solver="PULP_CBC_CMD")
        hilp_solver.config.offload = False
        hilp_solver.config.solve_only_top_level = True
        return hilp_solver

    def __init__(self, model, model_inputs, budget=None, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        super().__init__(model, model_inputs, budget=budget,
                         list_solvers=[self.solver()],
                         partitioners=[],
                         **kwargs)

class PureRockmate(Rockmate):
    @classmethod
    def solvers(self, force_python=False):
        hilp_solver = solvers.HILP(ilp_solver="PULP_CBC_CMD")
        hilp_solver.config.offload = False
        hilp_solver.config.solve_only_top_level = False
        hilp_solver.config.nb_total_nodes_top_level = 0
        rk_solver = PureRotor.solver(force_python=force_python)
        return [ hilp_solver, rk_solver ]

    @classmethod
    def partitioner(cls):
        return PureRotor.partitioner()

    def __init__(self, model, model_inputs, budget=None, rotor_solve_with_python=False, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        super().__init__(model, model_inputs, budget=budget,
                         list_solvers=self.solvers(), 
                         partitioners=[self.partitioner()],
                         **kwargs)

class Hiremate(Rockmate):
    valid_algorithms = ["hilp", "twremat", "rotor"]

    def __init__(self, model, model_inputs, budget=None, algorithms=valid_algorithms, partitioner_params={}, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        assert all(alg in self.valid_algorithms for alg in algorithms), f"invalid algorithm list '{','.join(algorithms)}'"
        assert "can_use_rotor" not in partitioner_params
        assert "main_graph_as_any_other" not in partitioner_params

        if not ('rotor' in algorithms): # hilp, twremat+hilp
            list_partitioners = [ 
                partitioned.PartitionerBottomToTop(
                    can_use_rotor=False,
                    **partitioner_params
                )
            ]
        else: # twremat+hilp+rotor
            list_partitioners = [
                partitioned.PartitionerBottomToTop(
                    can_use_rotor=True,
                    **partitioner_params
                ),
                partitioned.PartitionerSequence(
                    partitioned.PartitionerBottomToTop(
                        can_use_rotor=True,
                        main_graph_as_any_other=True,
                        **partitioner_params
                    )
                )
            ]
        
        list_solvers = []
        if "hilp" in algorithms:
            list_solvers.append(solvers.HILP(config=solvers.HILP.Config(time_limit=60)))
        if "rotor" in algorithms:
            list_solvers.append(solvers.RK_rotor())
        if "twremat" in algorithms:
            list_solvers.append(solvers.TwRemat())

        super().__init__(model, model_inputs, budget=budget,
                         list_solvers=list_solvers,
                         partitioners=list_partitioners,
                         **kwargs)
# TODO: Careful, Offmate does not have the same assumptions about 1/
# optimizer 2/ where the model is at the start 3/ do we handle
# parameter gradients or leave them untouched in memory
class Offmate(Rockmate):
    pass
