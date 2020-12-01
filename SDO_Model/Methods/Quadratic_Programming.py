#!/usr/bin/env python
# coding: utf-8
# Piotrek Sliwa
import os
import sys
import cvxopt
import matplotlib.pyplot as plt
import numpy as np

def give_flattened(X_dot, theta_matrix):
    """
    Flattens X_dot and theta_matrix according to Fortran 
    maintaining correct ordering of the data (so later multiplication makes sense)
    """
    return X_dot.reshape((-1), order="F"), theta_matrix.reshape((-1, theta_matrix.shape[-1]), order="F")

def give_qp_input(X_dot, theta_matrix, lasso_t):
    """
    Formats the data so, that it is accepted by cvxopt
    H, f correspond to notation from Daniel's report
    """
    #X_dot_, theta_matrix_ = give_flattened(X_dot, theta_matrix)
    X_dot_ = X_dot
    theta_matrix_ = theta_matrix

    T = float(X_dot.shape[0])
    H = theta_matrix_.T.dot(theta_matrix_) / T
    ft = -X_dot_.T.dot(theta_matrix_) / T  # remember about the minus

    num_ansatz = theta_matrix_.shape[-1]
    non_negative_LHS = -np.identity(num_ansatz)
    lasso_LHS = np.ones(num_ansatz)

    G = np.vstack([non_negative_LHS, lasso_LHS])
    non_negative_RHS = np.zeros(num_ansatz)
    h = np.hstack([non_negative_RHS, lasso_t])

    return H, ft.T, G, h

def run_qp(X_dot, theta_matrix, lasso_t):
    """
    Run quadratic program for X_dot, theta_matrix at given lasso_t constraint
    Returns sol - dictionary, whose most important keys are: 
    "x" - cvxopt.base.matrix, can be changed into python list by list(sol["x"])
    "status" - string information whether optimal state was reached
    "primal objective" - float with primal objective function value (mind the transformation,
    gives value proportional to the error [differing by a constant X_dot.T * X_dot / 2T]) 
    """
    P, q, G, h = give_qp_input(X_dot, theta_matrix, lasso_t)
    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc='d')
    h = cvxopt.matrix(h, tc='d')
    sol = cvxopt.solvers.qp(P, q, G, h)
    return sol

if __name__ == "__main__":
    path_to_data = sys.argv[1]
    X_dot_filename = sys.argv[2]
    theta_matrix_filename = sys.argv[3] # design matrix
    threshold = float(sys.argv[4])
    result_filename = sys.argv[5]

    X_dot = np.load(os.path.join(path_to_data, X_dot_filename))
    theta_matrix = np.load(os.path.join(path_to_data, theta_matrix_filename))

    sol = run_qp(X_dot, theta_matrix, threshold)
    print(sol["status"])
    np.save(result_filename, np.array(sol["x"]))

