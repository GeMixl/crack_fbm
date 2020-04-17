import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D

#from numpy.core.multiarray import ndarray


class HexLattice:
    def __init__(self, l: int=1, k: int=1, n: int=3) -> None:
        self.lenght = l
        self.stiffness = k
        self.size = n

        # Number of nodes is N times long lines (with N+2 nodes) plus N+1 time short lines (with N+1 nodes)
        no = (2*N+1) * N + (2*N) * (N+1)

        # Number of struts is N times the basic shape including 6 struts
        # plus one more basic shape at the lower right end of the lattice:
        #    --                      --
        #   /\/\                    /\
        #   ---- ... basic shape is --
        #   \/\/                    \/
        #    --
        st = (6 * N) * (2 * N) - N + 2 * N - 1

        self.Nodes = np.zeros((no, 2))
        self.Bars = np.zeros((st, 2)).astype(int)

        self.FocusNode = no//2

    def idx(k, l):
        return k * 2 * N + k // 2 + l

    def build_lattice(self) -> None:
        pass

    def print_lattice(self) -> None:
        pass


