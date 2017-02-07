'''
Created on 07.02.2017

@author: Yingxiong
'''
import numpy as np


def constraint(i, u_i, Ke_array, dof_map, R):
    ''' the stacked stiffness matrix Ke_array and RHS R are modified
    in such a way that the displacement constraint u_i is applied to the 
    i-th DOF
    '''
    el_arr, row_arr = np.where(dof_map == i)
    for el, i_dof in zip(el_arr, row_arr):
        rows = dof_map[el]
        R[rows] += -u_i * Ke_array[el, :, i_dof]
        Ke_ii = Ke_array[el, i_dof, i_dof]
        Ke_array[el, i_dof, :] = 0.
        Ke_array[el, :, i_dof] = 0.
        Ke_array[el, i_dof, i_dof] = -Ke_ii

    return Ke_array, R
