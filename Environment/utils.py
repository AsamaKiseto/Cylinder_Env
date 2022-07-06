import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot, cm

def get_value_u(u):
    f = u
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()

    w0 = f.compute_vertex_values(mesh)
    nv = mesh.num_vertices()
    # if len(w0) != gdim * nv:
    #     raise AttributeError('Vector length must match geometric dimension.')
    X = mesh.coordinates()
    X = [X[:, i] for i in range(gdim)]
    U = [w0[i * nv: (i + 1) * nv] for i in range(2)]

    # Compute magnitude
   
    args = X + U  
    return copy.deepcopy(args)
