import numpy as np
import math

def first_eq(x):
    f = x * np.log(x + 1) - 0.3
    return f

def first_eq_deriv(x):
    d = np.log(x + 1) + (x / (x + 1))
    return d

def locate_root(x_l, x_r, dx_l, dx_r):
    while first_eq(x_l) * first_eq(x_r) > 0:
        x_l += dx_l
        x_r += dx_r
    return x_l, x_r

def solver_1(x_0):
    solution = x_0 - first_eq(x_0) / first_eq_deriv(x_0)
    while abs(x_0 - solution) > 1e-4:
        x_0 = solution
        solution = x_0 - first_eq(x_0) / first_eq_deriv(x_0) 
    return solution

def jacob(X):
    return np.array([
        [2, math.cos(X[1])],
        [-math.sin(X[0] - 1), 1]
    ])

def nonlinear_system(X):
    return np.array([
        math.sin(X[1]) + 2 * X[0] - 2,
        X[1] + math.cos(X[0] - 1) - 0.7
    ])

def solver_2(X):
    j_inv = np.linalg.inv(jacob(X))
    f = nonlinear_system(X)
    solution = X - np.matmul(j_inv, f)
    while np.linalg.norm(X - solution) > 1e-4:
        X = solution
        solution = X - np.matmul(np.linalg.inv(jacob(X)), nonlinear_system(X)) 
    return solution

def main():
    _, x_1_r = locate_root(first_eq(0), 0, 0, 0.15) #location of the first root
    x_2_l, _ = locate_root(first_eq(0), 0, -0.15, 0) #location of the second root
    root_1 = solver_1(x_1_r)
    root_2 = solver_1(x_2_l)
    if abs(root_1) > abs(root_2):
        print(root_2)
    else:
        print(root_1)
    
    print(solver_2(np.zeros((2, 1))))
    return

if __name__ == '__main__':
    main()