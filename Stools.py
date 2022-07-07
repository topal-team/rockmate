from Btools import *
import ast

class S_node():
    def __init__(self,code,target=None):
        self.real_node = None # boolean, "size" nodes are artefacts
        self.main_code = code
        self.body_code = ast.Module() # .body : list of ast.Assign
        self.main_target = target # str
        self.all_targets = [target]

class S_graph():
    def __init__(self,dg : D_graph = None):
        self.nodes = []
        self.init_code = None
        if dg:
            self.output    = dg.output
            self.dict_info = dg.dict_info
            self.dict_rand = dg.dict_rand
        else:
            self.output    = None
            self.dict_info = {}
            self.dict_rand = {}

