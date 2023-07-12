import inspect

class DictInputs():
    def __init__(self,model,model_args,model_kwargs):
        # -- load params list --
        sign = inspect.signature(model.forward)
        params = list(sign.parameters.items())
        # -- build model_kwargs --
        if model_kwargs is None: model_kwargs = dict()
        elif not isinstance(model_kwargs,dict): raise Exception(
            f"model_kwargs must be a dict not {type(model_kwargs)}")
        # -- positional params --
        not_kw_params = [
            p[0] for p in params
            if p[0] not in model_kwargs]
        pos_params = [
            p[0] for p in params
            if (p[1].default is inspect._empty
            and p[0] not in model_kwargs)]
        # -- build positional inputs --
        if isinstance(model_args,dict):
            dict_inputs = model_args.copy()
            st_given = set(dict_inputs.keys())
            st_asked = set(pos_params)
            st_missing = st_asked - st_given
            nb_missing = len(st_missing)
            if nb_missing>0: raise Exception(
                f"Missing {nb_missing} inputs for the model: {st_missing}")
        else:
            if (isinstance(model_args,set)
            or  isinstance(model_args,list)
            or  isinstance(model_args,tuple)):
                inputs = list(model_args)
            else:
                inputs = [model_args]
            nb_given = len(inputs)
            nb_asked_pos = len(pos_params)
            nb_asked_tot = len(not_kw_params)
            if nb_given < nb_asked_pos: raise Exception(
                f"To few values given for the model inputs "\
                f"({nb_asked_pos - nb_given} missing).")
            if nb_given > nb_asked_tot: raise Exception(
                f"To much values given for the model inputs "\
                f"({nb_given - nb_asked_tot} too many, including kwargs).")
            dict_inputs = dict(zip(not_kw_params,inputs))

        dict_inputs.update(model_kwargs)
        self.dict = dict_inputs

    def to_list_args(self,model):
        """
        So that we can give all arguments without keywords 
        Only *args no **kwargs. Useful in particular to call jit.
        """
        sign = inspect.signature(model.forward)
        params = list(sign.parameters.keys())
        return [self.dict[p] for p in params if p in self.dict]

