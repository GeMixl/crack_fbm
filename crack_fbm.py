import numpy as np
import math
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D

#from numpy.core.multiarray import ndarray

RNG = scipy.io.loadmat("/home/mixlmay/Code/Matlab/crack_fbm/RNG.mat")
RNG = RNG['RNG'][:,0]

# bond length
length = 1.

# bond stiffness
bond_stiffness = 400.

# lattice size
N = 3

# Number of nodes is N times long lines (with N+2 nodes) plus N+1 time short lines (with N+1 nodes):
no = (2*N+1) * N + (2*N) * (N+1)

    # Number of struts is N times the basic shape (contains 6 struts)
    # plus one more basic shape at the lower right end of the lattice:
    #    --                      --
    #   /\/\                    /\
    #   ---- ... basic shape is --
    #   \/\/                    \/
    #    --
st = (6*N) * (2*N) - N + 2*N - 1

print("l=", length, "\nN = ", N, "\nno = ", no, "\nst = ", st)

# Array with the cartesian coordinates for each node:
No = np.zeros((no, 2))
# Strut array with the starting node and end node indices:
St = np.zeros((st, 2)).astype(int)

#  Boundary nodes are (node enumeration starts in the lower left corner) ...
BdNo = np.sort(np.concatenate(
    #  1. all lower nodes:
    (range(0, 2*N),
     #  2. all left corner nodes of the long lines:
     range(2*N, no, 4*N+1),
     #  3. all right corner nodes of the long lines:
     range(4*N, no, 4*N+1),
     #  4. all left corners of the short lines:
     range(4*N+1, no, 4*N+1),
     #  5. all right corners of the short lines:
     range(6*N, no, 4*N+1),
     #  6. all upper nodes:
     range(no-2*N+1, no-1))))

# Center nodes are all non-boundary nodes:
CtNo = np.setdiff1d(np.arange(0, no), BdNo)

# Focus node is the middle node:
FcNo = no//2

# Set node coordinates in the No array:
for i in range(0, N*2+1):
    for j in range(0, 2*N + i%2):
        Nr = i * (2*N) + i//2 + j
        x = (i+1)%2 * 0.5 + j*length
        y = i * 3**0.5 / 2 * length
        No[Nr] = [x, y]


def idx(k, l):
    # Helper function to evaluate the node index (row, column) from its position in the grid:
    return k * 2 * N + k//2 + l


# Set strut indices in the Bo-array:
# Iterate over all the lines:
for i in range(0, 2*N+1):
    # Is it a short horizontal line (i is even)?
    if i%2 == 0:
        # Set all the horizontal struts:
        for j in range(0, 2*N-1):
            A = idx(i,j)
            B = idx(i, j+1)
            St[(12*N - 1) * i//2 + j] = [A, B]
    else :
        for j in range(0, 2*N):
            # Set all the diagonal struts below this line:
            A = idx(i, j)
            B = idx(i-1, j)
            C = idx(i, j+1)
            St[(12*N - 1) * (i//2) + 2*N - 1 + 2*j] = [A, B]
            St[(12*N - 1) * (i//2) + 2*N - 1 + 2*j + 1] = [B, C]
        for j in range(0, 2*N):
            # Set all the horizontal struts:
            A = idx(i, j)
            B = idx(i, j+1)
            St[(12*N - 1) * (i//2) + 6*N - 1 + j] = [A, B]
        for j in range(0, 2*N):
            # Set all the diagonal struts above this line:
            A = idx(i, j)
            B = idx(i+1, j)
            C = idx(i, j+1)
            St[(12*N - 1) * (i//2) + 8*N - 1 + 2*j] = [A, B]
            St[(12*N - 1) * (i//2) + 8*N - 1 + 2*j + 1] = [B, C]

# Array holding the focus struts (struts around the focus node):
FcSt = []
# Array holding the intact struts:
IntactSt = []

# Set the six focus struts, all the others are declared intact:
for i in range(0,st):
    if FcNo in St[i,:]:
        FcSt.append(i)
    else:
        IntactSt.append(i)

FcSt= np.array(FcSt)
IntactSt = np.array(IntactSt)

#Define some auxilary vectors:
e = np.ones(st)
# Strut lengths:
l = np.zeros(st)
# X-coordinate, unified
c = np.zeros(st)
# Y-coordinate, unified
s = np.zeros(st)

for i in range(0, len(l)):
    l[i] = math.sqrt((No[St[i,1],0] - No[St[i,0],0])**2 + (No[St[i,1],1] - No[St[i,0],1])**2)
    c[i] = (No[St[i, 1], 0] - No[St[i, 0], 0]) / l[i]
    s[i] = (No[St[i, 1], 1] - No[St[i, 0], 1]) / l[i]

# Hydrostatic pressure (constant to be multiplied with the unit stress):
pressure = 1
# Imposed stresses:
sig_imp = np.zeros((st*4, 1))
# Prepare some arrays for building up the stiffness matrix:
Ku = np.zeros((st*4, st*4))
Kp = np.zeros((st*2, st*2))
T = np.zeros((st*2, st*4))

for i in range(0, st):
    # Fabric tensor, translates the local strut coordinate systems:
    T[i*2:i*2+2, i*4:i*4+4] = [[c[i], s[i], 0, 0], [0, 0, c[i], s[i]]]
    # Unconstrained, unrelated, stiffness matrix in local coordinates:
    Kp[i*2:i*2+2, i*2:i*2+2] = np.multiply([[1, -1], [-1, 1]], bond_stiffness)

# Global, unconstrained, unrelated stiffness matrix:
# Ku = T' * Kp * T
Ku = np.dot(np.dot(np.transpose(T), Kp), T)

# Relation matrix, is set to unity where two struts interact (i.e. touch the same node):
R = np.zeros((st*4, no*2))
for i in range(0, st):
    R[i*4:i*4+2, St[i,0]*2:St[i,0]*2+2] = np.eye(2)
    R[i*4+2:i*4+4, St[i,1]*2:St[i,1]*2+2] = np.zeros((2,2))
    #R(i*4-3:i*4-2, St(i,1)*2-1:St(i,1)*2-0)=eye(2)
    #R(i*4-1:i*4-0, St(i,2)*2-1:St(i,2)*2-0)=eye(2)

# Global, related, unconstrainded stiffness matrix:
# Ks = R' * Ku * R
Ks = np.dot(np.dot(np.transpose(R), Ku), R)

# Homogeneous boundary conditions (entries in the stiffness matrix are set to unity):
BcHom = np.sort(np.concatenate((BdNo*2, BdNo*2+1)))  # type: ndarray
for i in BcHom:
    Ks[i, :] = 0
    Ks[:, i] = 0
    Ks[i, i] = 1

# Global, related and constrained stiffness matrix. Should be ready for inversion:
Ki = Ks

# Degrees-of-freedom is reduced by the number of boundary conditions:
Dof =  np.setdiff1d(np.arange(0, 2*no), BcHom)

# Rupture thresholds are set randomly:
sig_thr = RNG[0:st]

#
r = np.zeros((2*st, 1))

# Maximum stress condition for the loading process:
sig_tot = 10.

# Stress steps for the loading process:
delta_sig=0.05

# Start the loading process with zero stress:
sig_fc = 0.

# Flag for plotting of the stress steps:
plt_cnt = 1

# Loading starts with breaking the focus nodes:
BrokenSt = FcSt

# Result arrays:
# Global stress vs. total strain energy:
StrainEnergy = np.array([0., 0.])
# Rupture events:
RuptureEvents = np.array((0., 0., 0.))

# Loading loop:
while sig_fc < sig_tot:
    # Increment the stress on the focus node by one step
    sig_fc = sig_fc + delta_sig
    sig_nod = pressure * sig_fc

    # Update the StrainEnergy table with the new stress and strain energy
    np.append(StrainEnergy, [sig_nod, 0.5*bond_stiffness * np.sum(r[IntactSt*2])**2])

    # Check for bonds that break in the current load step
    BreakingSt = [i for i in IntactSt if sig_thr[i] < abs(r[i*2])]

#    np.append((RuptureEvents,
#        np.concatenate((np.full((len(BreakingSt),1), sig_nod), np.transpose(BreakingSt),
    #        np.transpose(sig_thr[BreakingSt])))))

    #for i in range(0, len(BreakingSt)):
    #    print("Strut rupture: " + np.array_str(BreakingSt[i]))

    if len(BreakingSt) != 0:
        BrokenSt = np.append(BrokenSt, BreakingSt)

    IntactSt = np.setdiff1d(IntactSt, BrokenSt)

    # Impose loading on all broken struts by using a local-to-global coordinate transformation:
    for i in BrokenSt:
        sig_imp[4*i:4*i+4] = np.dot([[c[i], 0], [s[i], 0], [0, c[i]], [0, s[i]]], [[-1*sig_nod], [sig_nod]])

    # Set stresses on the nodes and node boundary conditions:
    sig_cur = np.dot(np.transpose(R), sig_imp)
    sig_cur[BcHom] = 0.

    # Solve the system of linear equations:
    # sig = Ki * v
    v = np.linalg.solve(Ki, sig_cur)

    # Get node forces
    p = np.dot(Ks, v)
    print(p)
    # Get the strut displacement
    u = np.dot(R, v)
    # Get the strut forces
    r = np.dot(Kp, np.dot(T, u))


print(BrokenSt)

fig = plt.figure(1, figsize=(8,8))
ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(-1, 2*N+1)
ax.set_ylim(-1, 2*N+1)

for i in range(0, no):
    if i is FcNo:
        nd_color = "r"
    elif i in BdNo:
        nd_color = "b"
    else:
        nd_color = "c"

    node_plot = patches.Circle((No[i, 0] + v[i*2], No[i,1] + v[i*2+1]), 0.1, color=nd_color)
    ax.add_patch(node_plot)
    t = plt.text(No[i,0], No[i,1], str(i))

for i in range(0, len(St)):
    if i in BrokenSt:
        st_color = "y"
    else:
        st_color = "b"

    strut_plot = Line2D((No[St[i, 0], 0] + v[St[i, 0] * 2], No[St[i, 1], 0] + v[St[i, 1] * 2]),
                        (No[St[i, 0], 1] + v[St[i, 0] * 2 + 1], No[St[i, 1], 1] + v[St[i, 1] * 2 + 1]),
                        color=st_color)
    ax.add_line(strut_plot)

    t = plt.text(
        No[St[i, 0], 0]+(No[St[i, 1], 0] - No[St[i, 0], 0])/2,
        No[St[i, 0], 1]+(No[St[i, 1], 1] - No[St[i, 0], 1])/2, str(i), color="b")

plt.show()
