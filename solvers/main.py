import rkgb
from rkgb.Htools import H_C_node, H_D_node, H_graph, H_cluster
from rkgb.Ktools import K_C_node, K_D_node


class Op:
    def __init__(self, kn, fast_forward=False, disabled=False, detach=True):
        self.kn = kn
        self.name = kn.name
        self.fast_forward = fast_forward
        self.disabled = disabled
        self.detach = detach
        self.is_del = isinstance(kn, K_D_node)

    def __repr__(self):
        return "Disabled" * self.disabled + self.name


class Solver():
    def Config():
        pass
    def __init__(self, config = None):
        self.config = config \
            if config is not None \
            else type(self).Config()
    def __call__(self, cluster : H_cluster, budgets=None):
        # -> RETURN list of Op_sched
        pass



def H_cluster_method_get_best_solutions(self,nb_sol):
    representee = self.representee_cluster
    if self is not representee:
        repr_sols = representee.get_best_solutions(self,nb_sol)
        ano_sols = [representee.translator.translate(sol) for sol in repr_sols]
        return [self.translator.reverse_translate(sol) for sol in ano_sols]
    else:
        return self.solutions[:nb_sol]# TODO: apply selection function

def H_cluster_method_solve(self, solver : Solver):
    if self is not self.representee_cluster:
        self.representee_cluster.solve(solver)
    elif solver.stop_condition(self):
        pass
    else:
        self.solutions.extend(solver(self))
setattr(H_cluster, "get_best_solutions", H_cluster_method_get_best_solutions)
setattr(H_cluster, "solve", H_cluster_method_solve)


class OpSchedule:

    self.solver = None



def solve_recursive(h_cluster):
    # assume it's representee
    for hg in h_cluster.possible_hg:
        for sub_cluster in hg.sub_graphs:
            if not stop_condition(sub_cluster.representee, h_cluster): # e.g. already solved/bottom level
                solve_recursive(sub_cluster.representee)
    for solver in list_solvers:
        h_cluster.solve(solver)
        h_cluster.solutions.extend(solver(h_cluster))


# save_all & fast_forward
# they should be based on the toposorted order of KCN
# adding them to H_cluster recursively



# def fast_fwd_overhead(self):
#     def _can_del(i, hdn):
#         for hcn in hdn.users:
#             if not hcn.is_fwd:
#                 continue
#             if self.list_hcn.index(hcn) > i:
#                 return False
#         return True

#     loss_idx = self.list_hcn.index(self.loss_hcn)
#     op_list = []
#     alive_mem = []
#     alive_list = []
#     alive_status = np.zeros(len(self.list_hdn), dtype=bool)
#     for hdn in self.inputs_hdn_data:
#         alive_status[self.list_hdn.index(hdn)] = 1
#     loss_idx = self.list_hcn.index(self.loss_hcn)
#     for i, hcn in enumerate(self.list_hcn[:loss_idx]):
#         op_list += hcn.ff_op_list
#         for hdn in hcn.users:
#             alive_status[self.list_hdn.index(hdn)] = 1
#         alive_list.append(alive_status.copy())
#         alive_mem.append(
#             sum(
#                 hdn.mem
#                 for j, hdn in enumerate(self.list_hdn)  # No phantom in FF
#                 if alive_status[j]
#             )
#         )

#         for j, hdn in enumerate(self.list_hdn):
#             # if False:  # TODO: if hdn is interface
#             # if hdn in self.inputs_hdn_data:
#             #     continue
#             if alive_status[j] and _can_del(i, hdn):
#                 alive_status[j] = 0
#                 # op_list.append(H_op(hdn.name, hdn, is_del=True))
#                 op_list.append(Op(hdn.kdn))
#                 alive_list.append(alive_status.copy())
#                 alive_mem.append(
#                     sum(
#                         hdn.mem
#                         for j, hdn in enumerate(self.list_hdn)
#                         if alive_status[j]
#                     )
#                 )
#     return max(alive_mem) - alive_mem[-1], op_list



# def add_sched(self, sched):
#     pareto = True
#     for opt in self.list_sched:
#         if (
#             opt.fwd_time + opt.bwd_time < sched.fwd_time + sched.bwd_time
#         ) and (opt.mem < sched.mem):
#             # should consider other factors like req_inputs
#             pareto = False
#         if (
#             opt.fwd_time + opt.bwd_time > sched.fwd_time + sched.bwd_time
#         ) and (opt.mem > sched.mem):
#             self.list_sched.remove(opt)
#     if pareto:
#         self.list_sched.append(sched)
#     # self.refine_scheds()

# def refine_scheds(self, expect_num=10):
#     def exclude_one():
#         def worse_sched(sched1, sched2):
#             if (
#                 sched1.fwd_time + sched1.bwd_time
#                 > sched2.fwd_time + sched2.bwd_time
#             ):
#                 return sched1
#             else:
#                 return sched2

#         # find the pair to remove one
#         diff = 100
#         for i, _sched1 in enumerate(self.list_sched[1:]):
#             for _sched2 in self.list_sched[i + 1 :]:

#                 if (
#                     abs((_sched2.mem - _sched1.mem + 1) / (_sched2.mem + 1))
#                     < diff
#                 ):
#                     sched1 = _sched1
#                     sched2 = _sched2
#                     diff = abs(
#                         (_sched2.mem - _sched1.mem + 1) / (_sched2.mem + 1)
#                     )
#         # assert sched1 in self.list_sched
#         # assert sched2 in self.list_sched
#         self.list_sched.remove(worse_sched(sched1, sched2))

#     while len(self.list_sched) > expect_num:
#         # print(len(self.list_sched))
#         exclude_one()





# class Solver_HILP(Solver):
#     # take_h_cluster = False # otherwise take h_graph
#     class Config():
#         sub_solvers : list[Solver] = None
#         nb_diff_budget = 10
#         def __init__(self, list_solvers, nb_diff_budget = 10):
#             self.sub_solvers = list_solvers
#         def get_budgets_to_solve_on(self,cluster,max_budget = None):
#             # return the list of budgets
#             pass

#     def __init__(list_solvers):
        

#     def __call__(self, H_obj, recursive=True):
#         if self.stop_solving:
#             return []
#         if isinstance()

#         for sub_clusters: sub_c.solve()

