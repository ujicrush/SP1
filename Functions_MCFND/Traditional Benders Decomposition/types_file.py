import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

class Config:
    def __init__(self, data=None):
        self._data = data if data is not None else {}
    
    def __getattr__(self, name):
        if name == '_data':
            return super().__getattribute__(name)
        return self._data.get(name)
    
    def __setattr__(self, name, value):
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __dir__(self):
        return list(self._data.keys()) + ['_data']
    
class cMCMDProblem:
    def __init__(self, 
                 n_nodes: int,
                 n_edges: int,
                 n_commodities: int,
                 n_days: int,
                 k: int,  # Sparsity param
                 A: np.ndarray,  # n_nodes x n_edges matrix
                 b: np.ndarray,  # n_commodities x n_nodes vector
                 c: np.ndarray,  # n_edges cost vector
                 d: List[float],  # n_edges cost vector
                 u: List[float],  # n_edges capacity vector  
                 lb: List[float],  # n_edges lower bound / feasible start for z
                 ub: List[float],  # n_edges upper bound for z
                 gamma: float,  # regularization
                 sampling_rate: float,  # divide days by s_rate each iteration
                 edge_map: Dict[int, Tuple[int, int, int]],  # TRACK edges before and after sparsifying
                 outgoing_edges: Dict[int, List[int]],  # For e in outgoing_edges[i], e is an outgoing edge from node i
                 incoming_edges: Dict[int, List[int]],  # For e in incoming_edges[i], e is an incoming edge to node i
                 old_to_new_map: Dict[int, int]  # old_to_new_map[i] = j, means that edge i is now edge j
                ):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_commodities = n_commodities
        self.n_days = n_days
        self.k = k
        self.A = A
        self.b = b
        self.c = c
        self.d = d
        self.u = u
        self.lb = lb
        self.ub = ub
        self.gamma = gamma
        self.sampling_rate = sampling_rate
        self.edge_map = edge_map
        self.outgoing_edges = outgoing_edges
        self.incoming_edges = incoming_edges
        self.old_to_new_map = old_to_new_map

class Cut:
    def __init__(self, obj: float, grad_obj: np.ndarray, status: str):
        self.obj = obj
        self.grad_obj = grad_obj  # ∇obj in Julia
        self.status = status

class Dual:
    def __init__(self, 
                 alpha: np.ndarray,
                 lam: float,
                 betal: np.ndarray,
                 betau: np.ndarray,
                 rho: np.ndarray,
                 w: np.ndarray,
                 ofv: float,
                 status: str
                 ):
        self.alpha = alpha
        self.gamma = gamma
        self.betal = betal
        self.betau = betau
        self.rho = rho
        self.w = w
        self.ofv = ofv
        self.status = status

class PrimalSolution:
    def __init__(self,
                 support: List[int],
                 z: List[float],
                 value: float,
                 offset: List[float],
                 slope: np.ndarray,
                 isbinary: bool,
                 method: str,
                 R_sample: List[int],
                 P: Dict[str, List[int]]):
        self.support = support
        self.z = z
        self.value = value
        self.offset = offset
        self.slope = slope
        self.isbinary = isbinary
        self.method = method
        self.R_sample = R_sample
        self.P = P

class InfeasibleCut:
    def __init__(self,
                 z: List[float],
                 R_sample: List[int],
                 p: np.ndarray,
                 b: np.ndarray,
                 infeas_r: List[int]):
        self.z = z
        self.R_sample = R_sample
        self.p = p
        self.b = b
        self.infeas_r = infeas_r

class DataLogger:
    def __init__(self, data: Dict[str, Any] = None):
        self.data = data if data is not None else {}