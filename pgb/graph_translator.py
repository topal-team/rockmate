# Anonymize graphs
# A way to recognize similar blocks
# for instance for GPT2 -> Transformer blocks
from .utils import *
from . import Stools
from . import Ktools
#from .Stools import S_node,S_graph,copy_S_graph
#from .Ktools import K_C_node,K_D_node,K_graph,copy_K_graph

# ==================
# ====== INIT ======
# ==================

class Graph_Translator():
    def __init__(self,sg=None,reverse_translator=None):
        """ There are two ways to __init__ a graph_translator,
        either you give a S_graph and it creates a translator to
        anonymize the graph, or you give it a translator and it
        creates the reverse translator. """
        if (sg is None) == (reverse_translator is None):
            raise Exception(
                "To __init__ a graph_translator, you must either give\n"\
                "it a S_graph or the reverse translator, not both.")
        if reverse_translator != None:
            self.reverse_translator = reverse_translator
            self.dict = d = dict()
            for s1,s2 in reverse_translator.dict.items():
                d[s2] = s1
        else:
            self.dict = r_to_a = dict()
            nb_var = 0
            def handle_str(real):
                if not real in r_to_a:
                    nonlocal nb_var
                    nb_var += 1
                    ano = f"a{nb_var}"
                    r_to_a[real] = ano
            def handle_ast(a):
                if isinstance(a,ast.AST):
                    if isinstance(a,ast.Name):
                        real = a.id
                        if real[:2] == "__":
                            handle_str(real)
                    else:
                        for s in a._fields:
                            try: handle_ast(getattr(a,s))
                            except: pass
                elif hasattr(a,"__iter__"):
                    for sub_a in a: handle_ast(sub_a)
            snodes = [sg.init_node] + sg.nodes
            for sn in snodes:
                mt,mc = sn.main_code
                handle_str(mt)
                handle_ast(mc)
                for st,sc in sn.body_code:
                    handle_str(st)
                    handle_ast(sc)
            self.reverse_translator = (
                Graph_Translator(reverse_translator=self))

# ==================



# ==================
# === TRANSLATE ====
# ==================

    def translate(self,x):
        # x's type can be :
        # -> str
        # -> ast.AST
        # -> S_node (/!\ in place /!\)
        # -> K_C/D_node (/!\ in place /!\)
        # -> S_graph
        # -> K_graph
        # -> an iterable with elts of types mentioned above
        translate = self.translate
        # -- STR --
        if isinstance(x,str):
            if x not in self.dict: return x
            else: return self.dict[x]

        # -- AST --
        elif isinstance(x,ast.AST):
            ty = type(x)
            if ty == ast.Name:
                return ty(translate(x.id))
            elif ty == ast.Call:
                return ty(x.func,translate(x.args),translate(x.keywords))
            elif ty == ast.keyword:
                return ty(x.arg,translate(x.value))
            elif ty == ast.List or ty == ast.Tuple:
                return ty(translate(x.elts))
            elif ty == ast.Subscript:
                return ty(translate(x.value),x.slice)
            elif ty == ast.UnaryOp:
                return ty(x.op,translate(x.operand))
            elif ty == ast.BinOp:
                return ty(translate(x.left),x.op,translate(x.right))
            elif ty == ast.Assign:
                return ty(translate(x.targets),translate(x.value))
            elif ty == ast.Module:
                return make_ast_module(translate(x.body))
            elif ty == ast.Constant:
                return x
            else: raise Exception(
                f"{x}'s type ({ty}) is not handled by the translator")

        # -- S_NODE --
        elif isinstance(x,Stools.S_node): # /!\ inplace /!\
            # op done inplace because it's impossible to change deps/users
            x.main_code   = translate(x.main_code)
            x.body_code   = translate(x.body_code)
            x.main_target = translate(x.main_target)
            x.all_targets = translate(x.all_targets)
            x.tensor_targets = translate(x.tensor_targets)
            # due to S_node.__hash__ we cannot change dict inplace
            deps  = list(x.deps.items())  ; x.deps.clear()
            users = list(x.users.items()) ; x.users.clear()
            for req_sn,st in deps:   x.deps[req_sn]   = translate(st)
            for user_sn,st in users: x.users[user_sn] = translate(st)
            return ()

        # -- K_C_NODE --
        elif isinstance(x,Ktools.K_C_node): # /!\ inplace like S_node /!\
            x.main_code   = translate(x.main_code)
            x.body_code   = translate(x.body_code)
            x.all_targets = translate(x.all_targets)
            x.main_target = mt = translate(x.main_target)
            x.tensor_targets = translate(x.tensor_targets)
            x.name = f"fwd_{mt}" if x.is_fwd else f"bwd_{mt}"
            return ()

        # -- K_D_NODE --
        elif isinstance(x,Ktools.K_D_node): # /!\ inplace like S_node /!\
            x.all_targets = translate(x.all_targets)
            x.main_target = mt = translate(x.main_target)
            x.name = f"{mt} {x.kdn_type}"

        # -- S_GRAPH --
        elif isinstance(x,Stools.S_graph):
            sg = Stools.copy_S_graph(x) # to protect x : NOT inplace
            snodes = [sg.init_node] + sg.nodes
            translate(snodes)
            # since S_nodes' name changed, and S_node.__hash__ is terrible
            # I prefer to regenerate all the dict once again
            for sn in snodes:
                deps  = list(sn.deps.items())  ; sn.deps.clear()
                users = list(sn.users.items()) ; sn.users.clear()
                for req_sn,st in deps:   sn.deps[req_sn]   = st
                for user_sn,st in users: sn.users[user_sn] = st
            # dict_info is currently shared by all the graphs
            # it's annoying so I clean it up.
            dict_info_keys = set(sg.dict_info.keys())
            for k in dict_info_keys:
                if k not in self.dict: del sg.dict_info[k]
            for attr in [
                "hidden_inputs","direct_inputs","dict_info",
                "hidden_output","direct_outputs","dict_rand"]:
                setattr(sg,attr,translate(getattr(sg,attr)))
            return sg

        # -- K_GRAPH --
        elif isinstance(x,Ktools.K_graph):
            kg = Ktools.copy_K_graph(x)
            translate(kg.list_kcn)
            translate(kg.list_kdn)
            translate(kg.input_kdn_data)
            translate(kg.input_kdn_grad)
            dkn = list(kg.dict_kn.values()) ; kg.dict_kn.clear()
            for kn in dkn: kg.dict_kn[kn.name] = kn
            dict_info_keys = set(kg.dict_info.keys())
            for k in dict_info_keys:
                if k not in self.dict: del kg.dict_info[k]
            for attr in ["init_code","dict_info","dict_rand"]:
                setattr(kg,attr,translate(getattr(kg,attr)))
            return kg

        # -- ITERABLE --
        elif type(x) in [list,tuple,set]:
            return type(x)(translate(sub_x) for sub_x in x)
        elif isinstance(x,dict):
            return dict(translate(c) for c in x.items())
        elif x is None: return None
        else: return x

    def reverse_translate(self,x):
        return self.reverse_translator.translate(x)

# ==================



# ==================
# == Utilisation ===
# ==================

# cc : connexe componente
def make_list_kg_eco(list_sg,model,verbose=None,device=None):
    nb_sg = len(list_sg)
    # 1) anonymize S_graphs and recognize similar S_graphs
    list_translator = [None] * nb_sg
    sg_num_to_cc_num = [None] * nb_sg
    tab_S_repr_cc = [] # cc_num -> S_graph
    for sg_num in range(nb_sg):
        sg = list_sg[sg_num]
        list_translator[sg_num] = translator = Graph_Translator(sg)
        ano_sg = translator.translate(sg)
        b = False ; cc_num = 0 ; nb_cc = len(tab_S_repr_cc)
        while not b and cc_num < nb_cc:
            if ano_sg == tab_S_repr_cc[cc_num]: b = True
            else: cc_num += 1
        if not b: tab_S_repr_cc.append(ano_sg)
        sg_num_to_cc_num[sg_num] = cc_num

    # 2) move anonymized graphs from S to K
    Ktools.aux_init_S_to_K(model,verbose,device)
    tab_K_repr_cc = []
    dict_info_global = list_sg[0].dict_info
    for ano_sg in tab_S_repr_cc:
        ano_sg.dict_info.update(dict_info_global)
        tab_K_repr_cc.append(Ktools.aux_build_S_to_K(ano_sg,model,None))
    list_kg = []
    for sg_num,cc_num in enumerate(sg_num_to_cc_num):
        ano_kg = tab_K_repr_cc[cc_num]
        list_kg.append(list_translator[sg_num].reverse_translate(ano_kg))

    # 3) link the K blocks
    for i in range(1,nb_sg):
        prev_kg = list_kg[i-1]
        kg      = list_kg[i]
        real_inp_data = prev_kg.output_kdn_data
        real_inp_grad = prev_kg.output_kdn_grad
        fake_inp_data = kg.input_kdn_data
        fake_inp_grad = kg.input_kdn_grad
        for fst_kcn in fake_inp_data.users_global:
            fst_kcn.deps_global.discard(fake_inp_data)
            fst_kcn.deps_global.add(real_inp_data)
            real_inp_data.users_global.add(fst_kcn)
        for lst_kcn in fake_inp_grad.deps_global:
            lst_kcn.users_global.discard(fake_inp_grad)
            lst_kcn.users_global.add(real_inp_grad)
            real_inp_grad.deps_global.add(lst_kcn)
        assert(real_inp_data.main_target == fake_inp_data.main_target)
        assert(real_inp_grad.main_target == fake_inp_grad.main_target)

    return list_translator,list_kg














