import inspect

class ExampleInputs():
    def __init__(self,original_mod,original_mod_args,original_mod_kwargs=None):
        self.args = tuple()
        self.kwargs = dict()
        self.dict = dict()
        if original_mod_args is ExampleInputs:
            self.dict = original_mod_args.dict
            self.args = original_mod_args.args
            self.kwargs = original_mod_args.kwargs
        # -- load params list --
        sign = inspect.signature(original_mod.forward)
        params = list(sign.parameters.items())
        # -- build original_mod_kwargs --
        if original_mod_kwargs is None: original_mod_kwargs = dict()
        elif not isinstance(original_mod_kwargs,dict): raise Exception(
            f"original_mod_kwargs must be a dict not {type(original_mod_kwargs)}")
        # -- positional params --
        not_kw_params = [
            p[0] for p in params
            if p[0] not in original_mod_kwargs]
        pos_params = [
            p[0] for p in params
            if (p[1].default is inspect._empty
            and "*" not in repr(p[1])
            and p[0] not in original_mod_kwargs)]
        
        # -- kwargs --
        self.kwargs = original_mod_kwargs
        self.dict.update(original_mod_kwargs)

        # -- build positional inputs --
        if isinstance(original_mod_args,dict):
            dict_example_inputs = original_mod_args.copy()
            st_given = set(dict_example_inputs.keys())
            st_asked = set(pos_params)
            st_missing = st_asked - st_given
            nb_missing = len(st_missing)
            if nb_missing>0: raise Exception(
                f"Missing {nb_missing} inputs for the original_mod: {st_missing}")
        else:
            if (isinstance(original_mod_args,set)
            or  isinstance(original_mod_args,list)
            or  isinstance(original_mod_args,tuple)):
                inputs = list(original_mod_args)
            else:
                inputs = [original_mod_args]
            nb_given = len(inputs)
            nb_asked_pos = len(pos_params)
            nb_asked_tot = len(not_kw_params)
            if nb_given < nb_asked_pos or nb_given > nb_asked_tot:
                raise Exception(f"Incorrect number of values given for the original_mod inputs "\
                                f"({nb_given} given, should be between {nb_asked_pos} and {nb_asked_tot}).")
            dict_example_inputs = dict(zip(not_kw_params,inputs))
            self.args = tuple(inputs)
            self.dict.update(dict_example_inputs)

    def to_list_args(self,original_mod):
        """
        So that we can give all arguments without keywords 
        Only *args no **kwargs. Useful in particular to call jit.
        """
        sign = inspect.signature(original_mod.forward)
        params = list(sign.parameters.keys())
        return tuple(self.dict[p] for p in params if p in self.dict)

