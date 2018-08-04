#!/usr/bin/env python3
import numpy as np
import torch
import torch.autograd as ag
from constants import *
from operator import mul
import IPython
import time

#TODO: make this a class?


def nd_range(stop, dims = None):
    if dims == None:
        dims = len(stop)
    if not dims:
        yield ()
        return
    for outer in nd_range(stop, dims - 1):
        for inner in range(stop[dims - 1]):
            yield outer + (inner,)
            
            
            
            
def skew_symmetric(vector):
    ss = ag.Variable(torch.zeros(3, 3), requires_grad = False)
    #Creates a tensorflow matrix which is a skew-symmetric version of the input vector    
    ss[0, 1] = -vector[2]
    ss[0, 2] = vector[1]
    ss[1, 0] = vector[2]
    ss[1, 2] = -vector[0]
    ss[2, 0] = -vector[1]
    ss[2, 1] = vector[0]    
    return ss
    
def full_jacobian(f, wrt):

    if isinstance(wrt, list):  
        is_list = True      
        wrt_shape = [len(wrt)]
    else:        
        is_list = False
        wrt_shape = list(wrt.size())
        
    f_shape = list(f.size())
    fs = []
    
    #fj = ag.Variable(torch.zeros(f_shape + wrt_shape))
    
    
    f_range = nd_range(f_shape)
    wrt_range = nd_range(wrt_shape)
    
    
    
    for f_ind in f_range:
        
        try:
            
            grad = ag.grad(f[tuple(f_ind)], wrt, retain_graph=True, create_graph=True)
            if not is_list:
                grad = grad[0]
            else:
                grad = torch.cat(grad)
        except:
            grad = torch.zeros_like(wrt)
            
        for i in range(len(f_shape)):
            grad = grad.unsqueeze(0)
        fs.append(grad)

    fj = torch.cat(fs, dim=0)    
    fj = fj.view(f_shape + wrt_shape)
    #fj = fj.view(tuple(f_shape + wrt_shape))
    return fj
    

def nth_derivative(f, wrt, n, return_all=True):
    '''
    if n == 1:
        return full_jacobian(f, wrt)
    else:        
        deriv = nth_derivative(f, wrt, n-1)
        return full_jacobian(deriv, wrt)
    '''
    next_val = f
    return_list = []
    for i in range(n):
        return_list.append(full_jacobian(next_val, wrt))
        next_val = return_list[-1]
    if return_all:
        return return_list
    else:
        return return_list[-1]
        

def full_hessian(f, wrt, return_jacobian = False):
    fj = full_jacobian(f, wrt)
    fh = full_jacobian(fj, wrt)
    if return_jacobian:
        return fj, fh
    else:
        return fh

    


def p_inv(var):
    if len(list(var.size())) != 2:
        raise Exception('Pseudoinverse can only be of 2D tensor')
        
    [U, S, V] = torch.svd(var)
    s_size = S.size()
    for i in range(min(s_size)):
        S[i] = 1.0 / S[i]
        
    S_pinv = ag.Variable(torch.zeros(V.size()[1], U.size()[1]))
    
    for i in range(len(S)):
        S_pinv[i, i] = S[i]
        

    return torch.mm(torch.mm(V, S_pinv), U.t())

if __name__ == '__main__':
    gen = nd_range((2, 3, 4))
    
    #for i in gen:
    #    print(i)
    start = time.time()
    s = ag.Variable(torch.from_numpy(np.array([1.0, 2.0, 3.0])).type(dtype), requires_grad=True)
    op = torch.ger(s, s)
    #full_hessian(op, s)
    lots_of_tensors = nth_derivative(op, s, 2)
    print(lots_of_tensors)
    end = time.time()
    print(end - start)
    #print(fj)
    #print(fh)
