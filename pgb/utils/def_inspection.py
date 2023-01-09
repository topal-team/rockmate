# One day everything concerning inspection should be moved here
# Ktools is to big and I prefer having a simple "inspect_snode" function
# but we should keep the "inspector" code in case one day someone need it

class inspection_result():
    def __init__(self):
        self.mem_del_fwd  = MemSize(0)
        self.overhead_fwd = MemSize(0)
        self.overhead_bwd = MemSize(0)
        self.mem_run_fwd  = MemSize(0)
        self.mem_run_bwd  = MemSize(0)
        self.mem_fgt_fwd  = MemSize(0)
        self.mem_fgt_bwd  = MemSize(0)
        self.time_run_fwd = 0
        self.time_run_bwd = 0

# -- Auxilary function to measure time taken by a function --
# -> between each iteration of the function, inter_fct is called
def measure_time(timer, fct, inter_fct=None):
    t = timer.measure_median(fct,samples=1)
    nb_repeat = 1
    measures = [t] ; tot = t
    while (tot < time_min_duration or nb_repeat < time_min_repeat):
        if inter_fct: inter_fct()
        t = timer.measure_median(fct,samples=1)
        measures.append(t)
        tot += t ; nb_repeat += 1
    if len(measures)>2:
        return (sum(measures)-max(measures)-min(measures))/(len(measures)-2)
    else:
        return np.median(measures)

def inspect_snode(sn,sg,our_global):
    result = inspection_result()
    mt   = sn.main_target
    info = sg.dict_info[mt]
    measure_time = rotor.timing.make_time(device)
    measure_mem  = rotor.memory.MeasureMemory(device)
    tmp_local = generate_tmp_local(sn,sg,our_global)

    # === FORWARD ===
    # -> First we wrap the code we want to inspect in Python functions
    def fct_run_fwd():
        code_run_fwd = sn.get_code()
        exec(code_run_fwd, self.our_global, self.tmp_local)
    def fct_fgt_fwd():
        for tar in self.sn.tensor_targets:
            self.tmp_local[tar].data = torch.zeros(0,device=device)
    def fct_del_fwd():
        code = ""
        for tar in self.sn.tensor_targets:
            code += f"del {tar};"
        self.code_del_fwd = code
        exec(self.code_del_fwd, self.our_global, self.tmp_local)


        gc.disable()
        _ , mem_run_fwd , peak_fwd = self.memUsage.measure(fct_run_fwd)
        overhead_fwd = peak_fwd - mem_run_fwd
        self.ret["overhead_fwd"] = overhead_fwd
        self.ret["mem_run_fwd"] = mem_run_fwd
        if not only_run:
            _ , mem_del_fwd , _ = self.memUsage.measure(fct_del_fwd)
            self.ret["mem_del_fwd"] = minus_mem(mem_del_fwd)
            _ , _ , _ = self.memUsage.measure(fct_run_fwd)

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret["mem_fgt_fwd"] = minus_mem(mem_fgt_fwd)
            self.ret["time_run_fwd"] = time_run_fwd
        gc.enable()
    # ===============

    # === BACKWARD ===

    def fct_run_bwd(self):
        self.code_run_bwd = f"{self.mt}.backward({self.mt}.grad)"
        exec(self.code_run_bwd, self.our_global, self.tmp_local)

    def fct_fgt_bwd(self):
        for req_sn in self.sn.deps.keys():
            if not req_sn.is_artefact:
                for tar in req_sn.tensor_targets:
                    self.tmp_local[tar].grad = None
    def fct_prepare_bwd(self):
        self.code_run_fwd = self.sn.get_code()
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.sn.main_target].grad = generate_val(self.info,device)

    # measure
    def measure_bwd(self):
        #def fct_run_fwd():
        #    self.code_run_fwd = self.n.get_code() 
        #    exec(self.code_run_fwd, self.our_global, self.tmp_local)
        if self.info.requires_grad:
            #self.tmp_local[self.mt].data = generate_val(self.info,device)
            #self.tmp_local[self.mt].grad = generate_val(self.info,device)
            gc.disable()
            self.fct_prepare_bwd()
            _ , mem_run_bwd , peak_bwd = self.memUsage.measure(self.fct_run_bwd)
            overhead_bwd = peak_bwd - mem_run_bwd
            _ , mem_fgt_bwd , _ = self.memUsage.measure(self.fct_fgt_bwd)
            #fct_run_fwd()
            #self.timer.measure_median(fct_run_fwd)

            #self.tmp_local[self.n.main_target].grad = generate_val(self.info,device)
            self.fct_prepare_bwd()
            time_run_bwd = self.measure_time(self.fct_run_bwd, self.fct_prepare_bwd)
            # overhead_bwd contains n.target.data now /!\
            gc.enable()
            self.ret["overhead_bwd"] = overhead_bwd
            self.ret["mem_run_bwd"]  = mem_run_bwd
            self.ret["mem_fgt_bwd"]  = minus_mem(mem_fgt_bwd)
            self.ret["time_run_bwd"] = time_run_bwd
    # # ===============
