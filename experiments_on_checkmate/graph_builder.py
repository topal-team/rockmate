# =======================================
# = CONVERT K_GRAPH TO A GRAPH ON WHICH =
# =   WE CAN APPLY ORIGINAL CHECKMATE   =
# =======================================

from rkgb.utils import *
from rkgb import Ktools

class CHK_node():
    def __init__(self,kcn : Ktools.K_C_node):
        for attr in [
            "name",
            "main_target",
            "all_targets",
            "tensor_targets",
            "inplace_targets",
            "container_targets",
            "is_fwd",
            "is_rand",
            "main_code",
            "inplace_code",
            "body_code",
            "phantom_names",
            "overhead",
            "time",
            "unique_id"]:
            setattr(self,attr,getattr(kcn,attr))
        for attr in [
            "run_mem",
            "fgt_mem",
            "del_mem",
            "info"]:
            setattr(self,attr,None) # -> later
        for attr in [
            "deps_real",
            "deps_fake",
            "users_real",
            "users_fake"]:
            setattr(self,attr,set()) # -> init
    def __hash__(self):
        return self.unique_id
    def get_main_code(self,force_special_kwargs=False):
        return ast_add_on.make_str_assign(
            self.main_code,force_special_kwargs)
    def get_code(self,*args, **kwargs):
        return shared_methods.get_code(self,*args, **kwargs)
    def full_code(self,*args, **kwargs):
        return shared_methods.full_code(self,*args, **kwargs)


class CHK_graph():
    def __init__(self,kg : Ktools.K_graph):
        self.init_code = kg.init_code
        self.dict_info = kg.dict_info
        self.inputs    = kg.input_kdn_data.all_targets
        self.outputs   = kg.output_kdn_data.all_targets
        list_CHK_nodes = self.list_nodes = []
        dict_CHK_nodes = self.dict_nodes = dict()
        # -- build the nodes --
        for kcn in kg.list_kcn:
            CHK_n = CHK_node(kcn)
            list_CHK_nodes.append(CHK_n)
            dict_CHK_nodes[CHK_n.name] = CHK_n
            # -- buid mem attrs and info attr --
            mem_d  = irotor.MemSize(0)
            mem_ph = irotor.MemSize(0)
            for kdn in kcn.users:
                if kdn.kdn_type == "data":
                    mem_d = kdn.mem
                    CHK_n.info    = kdn.info
                elif kdn.kdn_type == "phantoms":
                    mem_ph = kdn.mem
            CHK_n.fgt_mem = mem_d
            CHK_n.run_mem = CHK_n.del_mem = mem_d + mem_ph

        # -- loss_node --
        self.loss_node = dict_CHK_nodes[kg.loss_kcn.name]

        # -- build edges --
        for CHK_n,kcn in zip(list_CHK_nodes,kg.list_kcn):
            for user_kdn in kcn.users:
                for user_kcn in user_kdn.users_real:
                    user_CHK_n = dict_CHK_nodes[user_kcn.name]
                    CHK_n.users_real.add(user_CHK_n)
                    user_CHK_n.deps_real.add(CHK_n)
                for user_kcn in user_kdn.users_fake:
                    user_CHK_n = dict_CHK_nodes[user_kcn.name]
                    CHK_n.users_fake.add(user_CHK_n)
                    user_CHK_n.deps_fake.add(CHK_n)



color_fwd  = "blue"
color_bwd  = "blueviolet"
color_special  = "green"

def print_CHK_graph(CHK_g : CHK_graph,name=None,open=True,render_format="svg"):
    if name is None: name = "CHK_graph"
    print(
        f"Graph for CHK with {len(CHK_g.list_nodes)} nodes")
    dot = graphviz.Digraph(name,
        comment="CHK-graph")
    
    def print_node(cn):
        mt = cn.main_target
        if mt == "loss": dot.node(cn.name,"LOSS KCN",color=color_special)
        else:
            lbl = cn.get_code() if cn.is_fwd else f"backward of {mt}"
            dot.node(cn.name,lbl,
                color = color_fwd if cn.is_fwd else color_bwd,
                tooltip = (
                f"Time : {cn.time}\n"\
                f"run_mem : {cn.run_mem}\n"\
                f"fgt_mem : {cn.fgt_mem}\n"\
                f"del_mem : {cn.del_mem}\n"\
                f"Mem overhead : {cn.overhead}"))
    for cn in CHK_g.list_nodes: print_node(cn)
    # ** edges **
    for cn in CHK_g.list_nodes:
        for req_cn in cn.deps_real:
            dot.edge(req_cn.name,cn.name)
        for req_cn in cn.deps_fake:
            dot.edge(req_cn.name,cn.name,style="dashed")

    small_fcts.graph_render(dot,open,"CHK",render_format)

