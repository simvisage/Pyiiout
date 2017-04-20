import numpy as np
import sympy as sp
from conjugate_gradient import cg
from constraint import constraint

# generate shape functions with sympy
xi_ = sp.symbols('xi')
N_i_xi = sp.Matrix([(1. - xi_) / 2.0, (1 + xi_) / 2.0, ], dtype=np.float_)
dN_i_xi = N_i_xi.diff('xi')
# numerical integration points (IP) and weights
xi_m = np.array([[-1], [1]])
w_m = np.array([1, 1])
# the values of the shape functions and their derivatives at the IPs
N_mi = np.array([N_i_xi.subs(xi_, xi) for xi in xi_m], dtype=np.float_)
dN_mei = np.array([[dN_i_xi.subs(xi_, xi)]for xi in xi_m], dtype=np.float_)
# element specifications
n_dim = 1  # dimension of the element
n_c = 2  # number of the composite components
n_e_n = 2  # number of nodes per element
n_e_dof = n_e_n * n_c * n_dim  # number of DOFs per element
# mesh specifications
n_e = 5  # number of element used
n_n_tot = n_e + 1  # total number of nodes
n_dof_tot = n_c * n_n_tot * n_dim  # total number of DOFs
# array handling the interaction of the material components
DELTA_cd = np.identity(n_c)
c = np.arange(n_c)
c1 = c + 1
SWITCH_cd = np.power(-1.0, c1[np.newaxis, :] + c1[:, np.newaxis])
# geometrical and material parameters
A_c = [1, 1]  # the cross-sectional area of each material component
E_c = [1, 1.]  # the modulus of each material component
G = 0.1  # the shear modulus of the bond interface
p = 1.  # the perimeter of the interface
L_x, L_y, L_z = 10, 1, 1  # dimension of the structure
# Geomerty approximation
I = np.arange(n_n_tot)
x_grid = np.array(np.mgrid[0:L_x:complex(0, n_n_tot), ])
x_Id = np.einsum('d...->...d', x_grid).reshape(-1, 1)
I_Ei = np.c_[I[:-1], I[1:]]
x_Eid = x_Id[I_Ei]
J_Emde = np.einsum('mei,Eid->Emed', dN_mei, x_Eid)
J_det_Em = np.linalg.det(J_Emde)
J_inv_Emed = np.linalg.inv(J_Emde)
# Quadratic forms
dN_Eimd = np.einsum('mei,Eide->Eimd', dN_mei, J_inv_Emed)
BB_ECidDjf = np.einsum('m, CD, C, C, Eimd,Ejmf,Em->ECidDjf',
                       w_m, DELTA_cd, A_c, E_c, dN_Eimd, dN_Eimd, J_det_Em)
NN_ECidDjf = np.einsum('m, CD,mi,mj,,,Em->ECiDj',
                       w_m, SWITCH_cd, N_mi, N_mi, p, G, J_det_Em)
BB_Eij = BB_ECidDjf.reshape(-1, n_e_dof, n_e_dof)
NN_Eij = NN_ECidDjf.reshape(-1, n_e_dof, n_e_dof)
K_Eij = BB_Eij + NN_Eij
# Multilayer expansion
C = np.arange(n_c) * n_n_tot
I_C = I[np.newaxis, :] + C[:, np.newaxis]
I_ECi = np.vstack([[I_C[:, :-1], I_C[:, 1:]]]).T.reshape(-1, n_e_dof)
# apply constraints and solve
F_ext = np.zeros((n_dof_tot,), np.float_)
K_Eij, F_ext = constraint(n_dof_tot / 2 - 1, 0., K_Eij, I_ECi, F_ext)
K_Eij, F_ext = constraint(n_dof_tot - 1, 0.01, K_Eij, I_ECi, F_ext)
d_I = cg(K_Eij, F_ext, I_ECi)
print'd_I', d_I
# post processing
import matplotlib.pyplot as plt
plt.plot(x_Id[:, 0], d_I.reshape(2, -1).T)
plt.xlabel('x')
plt.ylabel('displacement')
plt.show()
