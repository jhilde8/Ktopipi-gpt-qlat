import numpy as np
import qlat as q


class Operator:
    def __init__(self, name, p=0, r=None, dagger=False, role='unknown'):
        self.name = name
        self.p = p
        self.r = r
        self.is_dagger = dagger
        self.role = role #src, snk, or ins. 

    #string representation of the operator
    def __repr__(self):
        dag = "^dag" if self.is_dagger else dag=""
        sm = f"sm_({self.r})" if self.r is not None else sm = ""

        return f"wf_({p}) * {sm} * {self.name}{dag}_{self.role}"

    #defines equality between operators
    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return (self.name == other.name and
                self.p == other.p and
                self.r == other.r and
                self.is_dagger == other.is_dagger and
                self.role == other.role)

    def __hash__(self):
        #allows an operator object to be used as a dict key. This is the whole point
        return hash((self.name, self.p, self.r, self.is_dagger,self.role))

    

