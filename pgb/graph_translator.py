# Anonymize graphs
# A way to recognize similar blocks
# for instance for GPT2 -> Transformer blocks
from .utils import *
from .Stools import S_node,S_graph,copy_graph

class graph_translator():
    def __init__(self,sg):
        self.ano_to_real = a_to_r = dict()
        self.real_to_ano = r_to_a = dict()
        nb_var = 0
        def handle_str(real):
            if not real in r_to_a:
                nonlocal nb_var
                nb_var += 1
                ano = f"a{nb_var}"
                r_to_a[real] = ano
                a_to_r[ano] = real
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

    def anonymize(self,x):
        # x's type can be :
        # -> str
        # -> ast.AST
        # -> S_node (/!\ in place /!\)
        # -> S_graph
        # -> an iterable with elts of types mentioned above
        anonymize = self.anonymize
        a_to_r = self.ano_to_real
        r_to_a = self.real_to_ano
        if isinstance(x,str):
            if x not in r_to_a: return x
            else: return r_to_a[x]
        elif isinstance(x,ast.AST):
            ty = type(x)
            if ty == ast.Name:
                return ty(anonymize(x.id))
            elif ty == ast.Call:
                return ty(x.func,anonymize(x.args),anonymize(x.keywords))
            elif ty == ast.keyword:
                return ty(x.arg,anonymize(x.value))
            elif ty == ast.List or ty == ast.Tuple:
                return ty(anonymize(x.elts))
            elif ty == ast.Subscript:
                return ty(anonymize(x.value),x.slice)
            elif ty == ast.UnaryOp:
                return ty(x.op,anonymize(x.operand))
            elif ty == ast.BinOp:
                return ty(anonymize(x.left),x.op,anonymize(x.right))
            elif ty == ast.Constant:
                return x
            else: raise Exception(
                f"{x}'s type ({ty}) is not handled by the translator")
        elif isinstance(x,S_node): # /!\ inplace /!\
            x.main_code   = anonymize(x.main_code)
            x.body_code   = anonymize(x.body_code)
            x.main_target = anonymize(x.main_target)
            x.all_targets = anonymize(x.all_targets)
            x.tensor_targets = anonymize(x.tensor_targets)
            # due to S_node.__hash__ we cannot change dict inplace
            deps = list(x.deps.items())   ; x.deps.clear()
            users = list(x.users.items()) ; x.users.clear()
            for req_sn,st in deps:
                x.deps[req_sn] = anonymize(st)
            for user_sn,st in users:
                x.users[user_sn] = anonymize(st)
            # -> no need to anonymize deps/users node
            # -> that's why ano(S_node) is inplace
        elif isinstance(x,S_graph):
            sg = copy_graph(x) # to protect x : NOT inplace
            anonymize(sg.init_node)
            anonymize(sg.nodes)
            # dict_info is currently shared by all the graphs
            # it's annoying so I clean it up.
            keys = set(sg.dict_info.keys())
            for k in keys:
                if k not in r_to_a:
                    del sg.dict_info[k]
            for attr in [
                "hidden_inputs","direct_inputs","dict_info",
                "hidden_output","direct_outputs","dict_rand"]:
                setattr(sg,attr,anonymize(getattr(sg,attr)))
            return sg
        elif type(x) in [list,tuple,set]:
            return type(x)(anonymize(sub_x) for sub_x in x)
        elif isinstance(x,dict):
            return dict(anonymize(c) for c in x.items())
        elif x is None: return None
        else: return x






