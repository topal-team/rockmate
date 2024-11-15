from .rockmate import Rockmate
from . import solvers
from rkgb import partitioned
from configmypy import Bunch
import yaml
from copy import copy

import inspect

available_solvers = {
    "hilp": solvers.HILP,
    "rotor": solvers.RK_rotor,
    "cheap": solvers.CheapSolver,
    "twremat": solvers.TwRemat
}

available_partitioners = {
    "sequence": partitioned.PartitionerSequence,
    "bottom_to_top": partitioned.PartitionerBottomToTop,
    "repetitive": partitioned.PartitionerRecognizeRepetitivePattern,
    "base": partitioned.Partitioner,
    "dagp": partitioned.PartitionerDagp,
}

def _default_config(func):
    config = Bunch()
    sign = inspect.signature(func)
    sample = func()
    for arg in sign.parameters.values():
        if arg.default is not arg.empty:
            config[arg.name] = getattr(sample, arg.name)
    return config

def default_config(name):
    if name in available_solvers:
        func = available_solvers[name].Config
    elif name in available_partitioners:
        func = available_partitioners[name].Config
    else:
        raise ValueError(f"default_config: unknown solver or partitioner {name}")
    return _default_config(func)

def _add_solver(config, name, position):
    defaults = _default_config(available_solvers[name].Config)
    config.solver[position][name] = defaults

def _add_top_solver(config, name):
    _add_solver(config, name, "top")
def _add_bottom_solver(config, name):
    _add_solver(config, name, "bottom")

def _add_partitioner(config, name):
    defaults = _default_config(available_partitioners[name].Config)
    config.partitioner[name] = defaults

def generate_config(config_type):
    result = Bunch()
    result.solver = Bunch()
    result.solver.top = Bunch()
    result.solver.bottom = Bunch()
    result.partitioner = Bunch()
    if config_type == "rotor":
        _add_top_solver(result, "rotor")
        _add_partitioner(result, "sequence")
    elif config_type == "rockmate":
        _add_bottom_solver(result, "hilp")
        _add_top_solver(result, "rotor")
        result.solver.bottom.hilp.nb_bdg_save = 10
        result.solver.bottom.hilp.nb_bdg_peak = 10
        _add_partitioner(result, "sequence")
    elif config_type == "checkmate":
        _add_top_solver(result, "hilp")
        result.solver.top.hilp.time_limit = 3600
        result.solver.top.hilp.nb_total_nodes = 10000
    elif config_type == "hilp":
        _add_bottom_solver(result, "hilp")
        _add_top_solver(result, "hilp")
        _add_partitioner(result, "bottom_to_top")
        result.solver.top.hilp.nb_total_nodes = result.partitioner.bottom_to_top.max_estimate_for_main_graph
        result.solver.top.hilp.time_limit *= 20
        result.solver.bottom.hilp.accurate_mem = False
        result.partitioner.bottom_to_top.can_use_rotor = False
    elif config_type == "hiremate":
        for solver in available_solvers.keys():
            _add_top_solver(result, solver)
            _add_bottom_solver(result, solver)
        result.solver.top.hilp.nb_total_nodes = 100
        result.solver.top.hilp.time_limit *= 20
        result.solver.bottom.hilp.accurate_mem = False
        for partitioner in available_partitioners.keys():
            _add_partitioner(result, partitioner)
    elif config_type == "offmate":
        _add_top_solver(result, "hilp")
        _add_bottom_solver(result, "cheap")
        result.solver.top.hilp.offload = True
        result.solver.top.hilp.nb_total_nodes = 100
        result.solver.top.hilp.nb_total_sched = 500
        result.solver.top.hilp.time_limit = 1200
        result.solver.bottom.cheap.add_offload = True
        _add_partitioner(result, "repetitive")
    elif config_type == "noremat":
        _add_top_solver(result, "cheap")
        result.solver.top.cheap.cheap_factor = 1e9
        _add_partitioner(result, "base")
    else:
        raise ValueError(f"Unknown config type {config_type}. Valid values are:"
                         "rotor, rockmate, checkmate, hilp, hiremate")
    return result

_yaml_inited = False
def save_config(config, filename):
    global _yaml_inited
    if not _yaml_inited:
        from yaml.representer import Representer
        yaml.add_representer(Bunch, Representer.represent_dict)
        _yaml_inited = True
    with open(filename, "w") as f:
        yaml.dump(config, f)

def load_config(filename):
    with open(filename, "r") as f:
        result = yaml.load(f, yaml.SafeLoader)
    return Bunch(result)

def _make_solver(name: str, b: Bunch):
    return available_solvers[name](**b)

def _make_partitioner(name: str, b: Bunch):
    return available_partitioners[name](**b)

def from_config(model, model_inputs, budget=None, config_type="hiremate", config=None, **kwargs):
    if config is None:
        config = generate_config(config_type)
    top_solvers = [ _make_solver(name, conf) for name, conf in config.solver.top.items() if conf is not None ]
    bottom_solvers = [ _make_solver(name, conf) for name, conf in config.solver.bottom.items() if conf is not None ]
    partitioners = [ _make_partitioner(name, conf) for name, conf in config.partitioner.items() if conf is not None ]
    if not partitioners:
        partitioners = [ partitioned.Partitioner() ]
    return Rockmate(model, model_inputs, budget, top_solvers=top_solvers, bottom_solvers=bottom_solvers,
                    partitioners=partitioners, **kwargs)

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
                         top_solvers=[self.solver(force_python=rotor_solve_with_python)],
                         partitioners=[self.partitioner()],
                         **kwargs)

class PureCheckmate(Rockmate):
    @classmethod
    def solver(cls):
        hilp_solver = solvers.HILP(ilp_solver="PULP_CBC_CMD", offload=False)
        return hilp_solver

    def __init__(self, model, model_inputs, budget=None, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        super().__init__(model, model_inputs, budget=budget,
                         top_solvers=[self.solver()],
                         bottom_solvers=[],
                         partitioners=[],
                         **kwargs)

class PureRockmate(Rockmate):

    @classmethod
    def partitioner(cls):
        return PureRotor.partitioner()

    def __init__(self, model, model_inputs, budget=None, rotor_solve_with_python=False, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        hilp_solver = solvers.HILP(ilp_solver="PULP_CBC_CMD", offload=False)
        rk_solver = PureRotor.solver(force_python=rotor_solve_with_python)
        super().__init__(model, model_inputs, budget=budget,
                         top_solvers=[rk_solver],
                         bottom_solvers=[hilp_solver],
                         partitioners=[self.partitioner()],
                         **kwargs)

class Hiremate(Rockmate):
    ## valid_algorithms = ["hilp", "twremat", "rotor"]
    valid_algorithms = ["hilp", "rotor"]

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
        
        top_solvers = []
        bottom_solvers = []
        if "hilp" in algorithms:
            top_solvers.append(solvers.HILP(time_limit=600, nb_total_nodes=100))
            bottom_solvers.append(solvers.HILP(time_limit=60))
        if "rotor" in algorithms:
            top_solvers.append(solvers.RK_rotor())
            bottom_solvers.append(solvers.RK_rotor())
        if "twremat" in algorithms:
            top_solvers.append(solvers.TwRemat())
            bottom_solvers.append(solvers.TwRemat())

        super().__init__(model, model_inputs, budget=budget,
                         top_solvers=top_solvers,
                         bottom_solvers=bottom_solvers,
                         partitioners=list_partitioners,
                         **kwargs)
        
# TODO: Careful, Offmate does not have the same assumptions about 1/
# optimizer 2/ where the model is at the start 3/ do we handle
# parameter gradients or leave them untouched in memory
class Offmate(Rockmate):
    valid_algorithms = ["hilp"]

    def __init__(self, model, model_inputs, nlayers=32, budget=None, algorithms=valid_algorithms, partitioner_params={}, **kwargs):
        assert "list_solvers" not in kwargs
        assert "partitioners" not in kwargs

        assert all(alg in self.valid_algorithms for alg in algorithms), f"invalid algorithm list '{','.join(algorithms)}'"
        # assert "can_use_rotor" not in partitioner_params
        assert "main_graph_as_any_other" not in partitioner_params

        # if not ('rotor' in algorithms): # hilp, twremat+hilp
        #     list_partitioners = [
        #         partitioned.PartitionerBottomToTop(
        #             can_use_rotor=False,
        #             **partitioner_params
        #         )
        #     ]

        partitioners = [partitioned.PartitionerRecognizeRepetitivePattern(
            strict_max_number_of_top_level_nodes=nlayers+4,
            max_number_of_patterns=nlayers+2,
            min_percentage_covered_required=0.75)]
        # else: # twremat+hilp+rotor
        #     list_partitioners = [
        #         partitioned.PartitionerBottomToTop(
        #             can_use_rotor=True,
        #             **partitioner_params
        #         ),
        #         partitioned.PartitionerSequence(
        #             partitioned.PartitionerBottomToTop(
        #                 can_use_rotor=True,
        #                 main_graph_as_any_other=True,
        #                 **partitioner_params
        #             )
        #         )
        #     ]
        
        solver = solvers.HILP(ilp_solver="PULP_CBC_CMD")
        solver.config.offload = True
        solver.config.solve_only_top_level = True
        top_solvers = [solver]
        bottom_solvers = [solvers.CheapSolver()]
        # if "hilp" in algorithms:
        #     top_solvers.append(solvers.HILP(time_limit=600, nb_total_nodes=100))
        #     bottom_solvers.append(solvers.HILP(time_limit=60))
        # if "rotor" in algorithms:
        #     top_solvers.append(solvers.RK_rotor())
        #     bottom_solvers.append(solvers.RK_rotor())

        super().__init__(model, model_inputs, budget=budget,
                         top_solvers=top_solvers,
                         bottom_solvers=bottom_solvers,
                         partitioners=partitioners,
                        #  gpu_optim=torch.optim.Adam,
                         minor_offload_size=10*1024**2,
                         **kwargs)
