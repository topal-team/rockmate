from pulp import LpVariable

def set_hcn_list_sched(hcn, list_sched):
    """
    As part of preprocess for the hierarchical solver, we selection
    schedules and store in hcn.list_sched, which is temporarily used 
    by the ILP. 
    """
    setattr(hcn, "list_sched", list_sched)

def clean_hcn_list_sched(hcn):
    delattr(hcn, "list_sched")

def set_hg_parameter_groups(hg, parameter_groups):
    setattr(hg, "parameter_groups", parameter_groups)

def clean_hg_parameter_groups(hg):
    delattr(hg, "parameter_groups")



class RkLpVariable(LpVariable):
    def __init__(self, name, lowBound=None, upBound=None, cat="Continuous", e=None, solution=None):
        super().__init__(name=name, lowBound=lowBound, upBound=upBound, cat=cat, e=e)
        self.solution = solution

    def value(self):
        return self.solution or self.varValue

    @classmethod
    def dicts(
        cls,
        name,
        indices=None,
        lowBound=0,
        upBound=1,
        cat="Continuous",
        indexStart=[],
    ):
        d = {}
        for index in indices:
            var_name = name + "_" + "_".join([str(i) for i in index])
            d[index] = RkLpVariable(
                var_name, lowBound=lowBound, upBound=upBound, cat=cat
            )
        return d

    def __repr__(self):
        if self.varValue:return str(self.varValue)
        return super().__repr__()

    def prefill(self, value):
        self.setInitialValue(value)
        self.fixValue()
