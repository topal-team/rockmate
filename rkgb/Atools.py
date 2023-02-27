# =======================
# == Anonymized graphs ==
# =======================

# A way to recognize similar blocks
# e.g. for GPT2 -> Transformer blocks
from rkgb.utils import *
from rkgb import Stools
from rkgb import Ktools

# Note : to handle parameters anonymization :
# 1) I need to check "info" equality, -> I need the model
# 2) It's impossible to run inspection with anonymized params
#    -> I need to reverse_translate the parameters first
#    -> then re-translate, and finally reverse_translate
#    -> for all the copies of the ano graph


# ==================
# ====== INIT ======
# ==================

class Graph_Translator():
    def __init__(self,sg=None,model=None,reverse_translator=None):
        """ There are two ways to __init__ a graph_translator,
        either you give a S_graph and it creates a translator to
        anonymize the graph, or you give it a translator and it
        creates the reverse translator.
        Note: to fully translate S_graph, I try to translate
        parameters too, to do so I need to precise their shape."""
        if not reverse_translator is None:
            self.reverse_translator = rev = reverse_translator
            self.main_dict  = md = dict()
            self.param_dict = pd = dict()
            self.const_dict = cd = dict()
            for s1,s2 in rev.main_dict.items(): md[s2] = s1
            for s1,s2 in rev.const_dict.items(): cd[s2] = s1
            for s1,(s2,info) in rev.param_dict.items():
                pd[s2] = (s1,info)
        elif not sg is None:
            # we want to respect the original order of sn.num
            # -> so we first gather all the names, then sort
            # -> them based on sn.num, and anonymize them.

            ########## FIRST PART ########## 
            all_real_vars   = []
            all_real_csts   = []
            all_real_params = []
            def handle_str(real_str):
                if (real_str[:2] == "__"
                and not real_str in all_real_vars):
                    all_real_vars.append(real_str)
                elif (real_str[:5] == "self."
                or real_str[:5] == "self["
                or real_str[:13] == "getattr(self."
                and not real_str in all_real_params):
                    all_real_params.append(real_str)
                elif (real_str[:5] == "_cst_"
                and not real_str in all_real_csts):
                    all_real_csts.append(real_str)
            def search_through(a):
                if isinstance(a,ast.AST):
                    if isinstance(a,ast.Name):
                        handle_str(a.id)
                    else:
                        for s in a._fields:
                            try: search_through(getattr(a,s))
                            except: pass
                elif isinstance(a,str): handle_str(a)
                elif hasattr(a,"__iter__"):
                    for sub_a in a: search_through(sub_a)

            snodes = [sg.init_node] + sg.nodes
            for sn in snodes:
                search_through(sn.main_code)
                search_through(sn.body_code)

            ########## SECOND PART ########## 
            # Now that "all_real_vars" is complete, we generate the dict
            all_real_vars = sorted(
                all_real_vars,key = shared_methods.get_num_tar)
            self.main_dict = r_to_a = dict()
            nb_var = 0
            for real_name in all_real_vars:
                nb_var += 1
                ano_name = f"__{nb_var}_ano"
                r_to_a[real_name] = ano_name

            # Same for "all_real_csts", ie constants
            all_real_csts = sorted(
                all_real_csts,key = shared_methods.get_num_cst)
            self.const_dict = cst_r_to_a = dict()
            nb_cst = 0
            for real_name in all_real_csts:
                nb_cst += 1
                ano_name = f"_cst_{nb_cst}_ano"
                cst_r_to_a[real_name] = ano_name

            # Try to anonymize parameters:
            self.param_dict = param_dict = dict()
            nb_param = 0
            if model:
                for param_full_name in all_real_params: # strings
                    # -> e.g. param_full_name = "self.layer1.weight"
                    param = eval(param_full_name,{"self":model},{})
                    info_param = def_info.Var_info(param)
                    nb_param += 1
                    param_dict[param_full_name] = (
                        f"self.param_{nb_param}",info_param)
            # To finish, build the reverse_translator :
            self.reverse_translator = (
                Graph_Translator(reverse_translator=self))
        else:
            self.main_dict = dict()
            self.param_dict = dict()
            self.const_dict = dict()
            self.reverse_translator = self

# ==================



# ==================
# === TRANSLATE ====
# ==================

    def translate(self,x):
        # x's type can be :
        # -> str
        # -> ast.AST
        # -> Var_info (/!\ in place /!\)
        # -> S_node (/!\ in place /!\)
        # -> K_C/D_node (/!\ in place /!\)
        # -> S_graph
        # -> K_graph
        # -> an iterable with elts of types mentioned above
        translate = self.translate
        # -- STR --
        if isinstance(x,str):
            if x[:2] == "__" and x in self.main_dict:
                return self.main_dict[x]
            elif x[:5] == "_cst_" and x in self.const_dict:
                return self.const_dict[x]
            elif (x[:5] == "self." 
            or x[:5] == "self["
            or x[:13] == "getattr(self."
            and x in self.param_dict):
                return self.param_dict[x][0]
            elif ".grad_fn" in x:
                var = x.split('.')[0]
                if var in self.main_dict:
                    new_var = self.main_dict[var]
                    return new_var + x[len(var):]
            return x

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
                return ast_add_on.make_ast_module(translate(x.body))
            elif ty == ast.Constant:
                return x
            else: raise Exception(
                f"{x}'s type ({ty}) is not handled by the translator")

        # -- info --
        elif isinstance(x,def_info.Var_info):
            new_x = x.copy()
            new_x.data_owner_name = translate(new_x.data_owner_name)
            new_x.data_direct_parent_name = (
                translate(new_x.data_direct_parent_name))
            return new_x

        # -- S_NODE --
        elif isinstance(x,Stools.S_node): # /!\ inplace /!\
            # op done inplace because it's impossible to change deps/users
            x.main_code   = translate(x.main_code)
            x.inplace_code= translate(x.inplace_code)
            x.body_code   = translate(x.body_code)
            x.main_target = translate(x.main_target)
            x.all_targets       = translate(x.all_targets)
            x.tensor_targets    = translate(x.tensor_targets)
            x.inplace_targets   = translate(x.inplace_targets)
            x.container_targets = translate(x.container_targets)
            # Since S_node.__hash__ isn't changed, we change dict inplace
            for req_sn,st in x.deps.items():
                x.deps[req_sn] = translate(st)
            for user_sn,st in x.users.items():
                x.users[user_sn] = translate(st)
            return ()

        # -- K_C_NODE --
        elif isinstance(x,Ktools.K_C_node): # /!\ inplace like S_node /!\
            for attr in [
                "main_code","inplace_code","body_code",
                "main_target","container_targets",
                "tensor_targets","all_targets",
                "inplace_targets","phantom_names",
                "alias_in_users_phantoms"]:
                setattr(x,attr,translate(getattr(x,attr)))
            mt = x.main_target
            x.name = f"fwd_{mt}" if x.is_fwd else f"bwd_{mt}"
            return ()

        # -- K_D_NODE --
        elif isinstance(x,Ktools.K_D_node): # /!\ inplace like S_node /!\
            for attr in [
                "main_target","container_targets",
                "tensor_targets","all_targets","inplace_targets",
                "alias_in_users_phantoms"]:
                setattr(x,attr,translate(getattr(x,attr)))
            mt = x.main_target
            x.name = f"{mt} {x.kdn_type}"

        # -- S_GRAPH --
        elif isinstance(x,Stools.S_graph):
            sg = Stools.copy_S_graph(x) # to protect x : NOT inplace
            snodes = [sg.init_node] + sg.nodes
            translate(snodes)
            # dict_info is currently shared by all the graphs
            # thus it contains unknown names for each block
            # -> impossible to translate -> so I clean it up.
            # -> I also disconnect inputs'info from the previous block
            dict_info_keys = set(sg.dict_info.keys())
            if len(self.main_dict) != 0: # to avoid special case
                for k in dict_info_keys:
                    if k not in self.main_dict:
                        del sg.dict_info[k]
                    elif k in sg.direct_inputs:
                        info = sg.dict_info[k]
                        info.data_owner_name = k
                        info.data_direct_parent_name = k
                        info.is_inplace = False
                        info.is_view = False
            for attr in [
                "direct_inputs","dict_info",
                "dict_rand",
                "hidden_output","direct_outputs"]:
                setattr(sg,attr,translate(getattr(sg,attr)))
            new_dict_constants = dict()
            for old,new in self.const_dict.items():
                new_dict_constants[new] = x.dict_constants[old]
            sg.dict_constants = new_dict_constants
            # -> I do NOT translate hidden/direct_inputs 
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
            if len(self.main_dict) != 0: # to avoid special case
                for k in dict_info_keys:
                    if k not in self.main_dict: del kg.dict_info[k]
            for attr in ["init_code","dict_info"]:
                setattr(kg,attr,translate(getattr(kg,attr)))
            new_dict_constants = dict()
            for old,new in self.const_dict.items():
                new_dict_constants[new] = x.dict_constants[old]
            kg.dict_constants = new_dict_constants
            for kdn in kg.list_kdn:
                kdn.info = kg.dict_info[kdn.main_target]
                new_set = set()
                for r in kdn.users_impossible_to_restore:
                    new_set.add((r[0],translate(r[1])))
                kdn.users_impossible_to_restore = new_set
            for kcn in kg.list_kcn:
                new_set = set()
                for r in kcn.deps_impossible_to_restore:
                    new_set.add((r[0],translate(r[1])))
                kcn.deps_impossible_to_restore = new_set
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
def S_list_to_K_list_eco(
        list_sg,model,verbose=None,
        device=None,print_cc = False):
    nb_sg = len(list_sg)
    # 1) anonymize S_graphs and recognize similar S_graphs
    list_translator = [None] * nb_sg
    sg_num_to_cc_num = [None] * nb_sg
    tab_S_repr_cc = [] # cc_num -> S_graph
    cc_num_to_repr_sg_num = []
    for sg_num in range(nb_sg):
        sg = list_sg[sg_num]
        list_translator[sg_num] = translator = (
            Graph_Translator(sg,model=model))
        ano_sg = translator.translate(sg)
        b = False ; cc_num = 0 ; nb_cc = len(tab_S_repr_cc)
        while not b and cc_num < nb_cc:
            if ano_sg == tab_S_repr_cc[cc_num]:
                # -> We also need to manualy check param_info equalities
                sort_key = lambda v : int(v[0][11:])
                repr_tr = list_translator[cc_num_to_repr_sg_num[cc_num]]
                ano_param_sg = sorted(
                    translator.param_dict.values(),key=sort_key)
                ano_param_repr = sorted(
                    repr_tr.param_dict.values(),key=sort_key)
                if ano_param_sg == ano_param_repr:
                    b = True
                else: cc_num += 1
            else: cc_num += 1
        if not b:
            tab_S_repr_cc.append(ano_sg)
            cc_num_to_repr_sg_num.append(sg_num)
        sg_num_to_cc_num[sg_num] = cc_num

    # 1') Compute and print connexe components
    nb_cc = len(tab_S_repr_cc)
    cc = [[] for _ in range(nb_cc)]
    for sg_num in range(nb_sg):
        cc[sg_num_to_cc_num[sg_num]].append(sg_num)
    if print_cc:
        for cc_num in range(nb_cc):
            print(f"Connexe component n°{cc_num}: {cc[cc_num]}")
        print(
          f"We now have {nb_cc} blocks "\
          f"to handle, instead of {nb_sg}")

    # 2) move anonymized graphs from S to K
    # -> /!\ attention to params /!\
    dict_info_global = list_sg[0].dict_info # we lost some global info
    dict_constants_global = list_sg[0].dict_constants
    Ktools.aux_init_S_to_K(model,verbose,device)
    tab_K_repr_cc = []
    for cc_num,ano_sg in enumerate(tab_S_repr_cc):
        repr_trans = list_translator[cc_num_to_repr_sg_num[cc_num]]
        tmp_trans_to_handle_params = Graph_Translator()
        tmp_trans_to_handle_params.param_dict = repr_trans.param_dict
        tmp_trans_to_handle_params.reverse_translator = (
            Graph_Translator(
                reverse_translator=tmp_trans_to_handle_params))
        save_dict_constants = ano_sg.dict_constants
        ano_sg = tmp_trans_to_handle_params.reverse_translate(ano_sg)
        ano_sg.dict_info.update(dict_info_global)
        ano_sg.dict_constants = save_dict_constants
        ano_kg = Ktools.aux_build_S_to_K(ano_sg,model,None)
        ano_kg = tmp_trans_to_handle_params.translate(ano_kg)
        ano_kg.dict_constants = save_dict_constants
        tab_K_repr_cc.append(ano_kg)
    list_kg = []
    for sg_num,cc_num in enumerate(sg_num_to_cc_num):
        ano_kg = tab_K_repr_cc[cc_num]
        real_kg = list_translator[sg_num].reverse_translate(ano_kg)
        real_kg.dict_info.update(dict_info_global)
        real_kg.dict_constants.update(dict_constants_global)
        list_kg.append(real_kg)

    # 3) link the K blocks
    for i in range(1,nb_sg):
        prev_kg = list_kg[i-1]
        kg      = list_kg[i]
        real_inp_data = prev_kg.output_kdn_data
        real_inp_grad = prev_kg.output_kdn_grad
        fake_inp_data = kg.input_kdn_data
        fake_inp_grad = kg.input_kdn_grad
        kg.input_kdn_data = real_inp_data
        kg.input_kdn_grad = real_inp_grad
        for fst_kcn in fake_inp_data.users_global:
            fst_kcn.deps_global.discard(fake_inp_data)
            fst_kcn.deps_global.add(real_inp_data)
            real_inp_data.users_global.add(fst_kcn)
        for lst_kcn in fake_inp_grad.deps_global:
            lst_kcn.users_global.discard(fake_inp_grad)
            lst_kcn.users_global.add(real_inp_grad)
            real_inp_grad.deps_global.add(lst_kcn)
        #assert(real_inp_data.main_target == fake_inp_data.main_target)
        #assert(real_inp_grad.main_target == fake_inp_grad.main_target)
        # We cannot make this assertion because I don't 
        # translate hidden inputs because we don't care 
        # about how direct_inputs was generated. But it 
        # implies that fake_inp_data targets are dummy.

    return cc,list_kg,tab_S_repr_cc




