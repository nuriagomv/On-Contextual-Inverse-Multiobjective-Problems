import gurobipy as gp
import numpy as np
import math
import time
from itertools import accumulate
from scipy.stats import spearmanr, kendalltau, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
import ot
    

def forward_problem(x_n,w,theta,
                    A,b,gamma,
                    d_z,K):

    C = function_C(x_n,w,theta)
    
    #criteria-wise optimal solutions
    v_criteriawiseopt = {}
    for k in range(K):
        model_k = gp.Model()
        model_k.setParam('OutputFlag', 0)
        var_v = model_k.addMVar(shape=d_z, lb=-float('inf'), ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="v")
        model_k.addConstr(A@var_v<=b)
        model_k.setObjective(C[k]@var_v, sense=gp.GRB.MINIMIZE)
        model_k.optimize()
        if model_k.status==2:
            v_criteriawiseopt[k] = var_v.X
        else:
            print("FAIL IN CRITERIA-WISE OPTIMAL SOLUTIONS")
            return None

    #forward model
    model = gp.Model()
    model.setParam('OutputFlag', 0) #disabling solver output
    z = model.addMVar(shape=d_z, lb=-float('inf'), ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="z")
    model.addConstr( A @ z <= b, name='z in Z') 

    if gamma == "l_1":
        model.setObjective(expr = gp.quicksum(C[k]@z for k in range(K)), sense=gp.GRB.MINIMIZE)
    else:
        t = model.addMVar(shape=1, lb = 0, ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="obj_fun_value")
        model.setObjective(expr = t, sense=gp.GRB.MINIMIZE)
        """
        if gamma == "l_inf":
            model.addConstrs((C[j]@(z-v[:,j]) <= t
                              for j in range(k)), name="aux")
        """
        if gamma == "l_2":
            model.addConstr(gp.quicksum((C[k]@(z-v_criteriawiseopt[k]))*(C[k]@(z-v_criteriawiseopt[k])) for k in range(K)) <=t**2, name="aux")
    
    model.update()
    model.optimize()

    if model.status == 2: # could solve
        return z.X, v_criteriawiseopt, model
    else:
        print("Not solved, model.status=",model.status)
        return None, None, model


def function_C(x_n,w,B):

    C = np.array([w[k]*x_n[k]@B[k] for k in range(len(B))])
    #print("LINEAR CRITERIA C=\n",C)
    return C


def inverse_problem(N, x, optimal_z, list_A, b, 
                    d_x, d_x_k, d_z, m, K,
                    gamma,
                    objfun, timelimit, solve_to, 
                    general_case,
                    prior = None,
                    check_feasibility_orig = None,
                    tol = 0.,
                    pista = False):

    model = gp.Model()
    model.setParam("NonConvex",2)
    model.setParam(gp.GRB.Param.TimeLimit, timelimit)
    if solve_to == "feasibility":
        #https://support.gurobi.com/hc/en-us/community/posts/6841043701265-How-Gurobi-solves-a-optimisation-problem-that-doesn-t-have-an-objective
        model.setParam(gp.GRB.Param.SolutionLimit, 1) #To find a feasible solution quickly, Gurobi executes additional feasible point heuristics when the solution limit is set to exactly 1.

    #inverse matrices to find
    theta_tilde = {}
    for k in range(K):
        if check_feasibility_orig is None:
            lowerb, upperb = -1, 1
        else:
            model.setParam('OutputFlag', 0) #disabling solver output
            (w_orig,theta_orig) = check_feasibility_orig
            bound = w_orig[k]*theta_orig[k] # https://support.gurobi.com/hc/en-us/community/posts/4405152437777-Use-Gurobi-to-check-a-solution-feasibility
            lowerb, upperb = bound, bound
        theta_tilde[k] = model.addMVar(shape=(d_x_k[k],d_z), 
                                       lb=lowerb, ub=upperb, 
                                       vtype=gp.GRB.CONTINUOUS, name="theta_tilde_"+str(k))
    C_nk = np.array([[x[n][k]@theta_tilde[k] for k in range(K)] for n in range(N)]) #pedir como C_nk[n][k] y ese elemento es un vector fila de dimension d_z

    #AYUDA PARA MI PROBLEMA DE PORTFOLIO
    if pista:
        model.addConstr(theta_tilde[0][:,-1] == np.zeros(d_x_k[0]))
        model.addConstr(theta_tilde[1][:,:-1] == np.zeros(d_z-1))


    #NORMALIZATION CONSTRAINT
    norms = model.addMVar(shape=K, lb = 0., ub=1., vtype=gp.GRB.CONTINUOUS, name="norms")
    if tol == 0.:
        model.addConstr(gp.quicksum(norms[k] for k in range(K)) == 1, name="unit_SumNorm")
    else:
        model.addConstr(gp.quicksum(norms[k] for k in range(K)) <= 1+tol, name="unit_SumNorm_U")
        model.addConstr(gp.quicksum(norms[k] for k in range(K)) >= 1-tol, name="unit_SumNorm_L")
    model.addConstrs((norms[k]**2 == gp.quicksum(theta_tilde[k][j][q]**2 for j in range(d_x_k[k]) for q in range(d_z))
                        for k in range(K)), name="define_norm**2")

    """
    #criteria-wise optimal solutions
    # VER CUÁNDO ES OMITIBLE
    if not ((gamma == "l_1") and (objfun == "min_nonzeros")): #solo omitible para el caso l_1 a maximizar sparsity
        v = model.addMVar(shape=(d,k,R), lb=-float('inf'), ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="v")
        v.PoolIgnore = 1
        #dual of the criteria-wise optimal solutions
        u = model.addMVar(shape=(m,k,R), lb=-float('inf'), ub=0, vtype=gp.GRB.CONTINUOUS, name="u")
        u.PoolIgnore = 1
        #primal fesibility:
        model.addConstrs((A@v[:,j,r]<=b
                            for j in range(k) for r in range(R)), name='primalfeas-v')
        #dual feasibility:
        if tol==0.:
            model.addConstrs((np.transpose(A) @ u[:,j,r] == C_rj[r][j]
                                for j in range(k) for r in range(R)), name="dualfeas-u")
        else:
            model.addConstrs((np.transpose(A) @ u[:,j,r] <= C_rj[r][j] +tol
                                for j in range(k) for r in range(R)), name="Udualfeas-u")
            model.addConstrs((np.transpose(A) @ u[:,j,r] >= C_rj[r][j] -tol
                                for j in range(k) for r in range(R)), name="Ldualfeas-u")
        #weak and strong duality
        model.addConstrs((C_rj[r][j]@v[:,j,r] <= u[:,j,r]@b
                            for j in range(k) for r in range(R)), name="equal_objval")
    """
    #consistency of each optimal solution in the sample r=1,...,R
    mu = model.addMVar(shape=(m,N), 
                       lb=0, ub=float('inf'), 
                       vtype=gp.GRB.CONTINUOUS, name="mu")
    for n in range(N):
        if tol == 0.:
            model.addConstrs((mu[i,n]*(list_A[n][i,:]@optimal_z[n]-b[i])==0 for i in range(m)), name='consistency-l1_a_'+str(n))
        else:
            model.addConstrs((mu[i,n]*(list_A[n][i,:]@optimal_z[n]-b[i])<=0+tol for i in range(m)), name='U_consistency-l1_a_'+str(n))
            model.addConstrs((mu[i,n]*(list_A[n][i,:]@optimal_z[n]-b[i])>=0-tol for i in range(m)), name='L_consistency-l1_a_'+str(n))
    
    if gamma=="l_1":
        for n in range(N):
            aux_vect_C = np.array([gp.quicksum(C_nk[n][k][q] for k in range(K)) for q in range(d_z)])
            model.addConstrs((-aux_vect_C[q] == (np.transpose(list_A[n])@mu[:,n])[q]
                              for q in range(d_z)), name='consistency-l1_b_'+str(n))
    if gamma == "l_2":
        y = model.addMVar(shape=(m,K,N), 
                          lb = -float('inf'), ub = 0.,
                          vtype=gp.GRB.CONTINUOUS, name="y_optcriteriawise")
        for n in range(N):
            if tol ==0.:
                model.addConstrs((np.transpose(list_A[n])@y[:,k,n] == x[n][k]@theta_tilde[k]
                                for k in range(K)))
            else:
                model.addConstrs((np.transpose(list_A[n])@y[:,k,n] <= x[n][k]@theta_tilde[k] +tol
                                for k in range(K)))
                model.addConstrs((np.transpose(list_A[n])@y[:,k,n] >= x[n][k]@theta_tilde[k] -tol
                                for k in range(K)))
            aux_vect_C_l2 = np.array([gp.quicksum(C_nk[n][k][q] * (x[n][k]@theta_tilde[k]@optimal_z[n] - y[:,k,n]@b)
                                                  for k in range(K))
                                      for q in range(d_z)])
            if tol ==0.:
                model.addConstrs((-aux_vect_C_l2[q] == (np.transpose(list_A[n])@mu[:,n])[q]
                                for q in range(d_z)), name='consistency-l1_b_'+str(n))
            else:
                model.addConstrs((-aux_vect_C_l2[q] <= (np.transpose(list_A[n])@mu[:,n])[q]+tol
                                for q in range(d_z)), name='consistency-l1_b_'+str(n))
                model.addConstrs((-aux_vect_C_l2[q] >= (np.transpose(list_A[n])@mu[:,n])[q]-tol
                                for q in range(d_z)), name='consistency-l1_b_'+str(n))
        
    """
    else:
        mu = model.addMVar(shape=(m,R), lb=0, ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="mu")
        mu.PoolIgnore = 1
        d_optval = model.addMVar(shape=(k,R), lb=-float('inf'), ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="aux_d")
        d_optval.PoolIgnore = 1
        model.addConstrs((d_optval[j,r] == -(C_rj[r][j])@v[:,j,r]
                          for r in range(R) for j in range(k)), name="aux_d")
        if gamma=="l_inf":
            lambd = model.addMVar(shape=(k,R), lb=0, ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name="lambda")
            lambd.PoolIgnore = 1
            model.addConstrs((-gp.quicksum(lambd[j,r]*C_rj[r][j] for j in range(k)) == gp.quicksum(mu[i,r]*A[i,:] for i in range(m))
                              for r in range(R)), name='consistency-linf_a')
            model.addConstrs((gp.quicksum(lambd[j,r] for j in range(k)) == 1
                              for r in range(R)), name='consistency-linf_b')
            model.addConstrs((lambd[j,r]*((C_rj[r][j])@optimal_z[r] + d_optval[j][r] - (C_rj[r][l])@optimal_z[r] - d_optval[l][r]) >= 0
                              for r in range(R) for j in range(k) for l in range(k)), name='consistency-linf_c')
            model.addConstrs((mu[i,r]*(A[i,:]@optimal_z[r] -b[i]) == 0
                              for r in range(R) for i in range(m)), name='consistency-linf_d')

        if gamma=="l_2":
            model.addConstrs((-gp.quicksum(C_rj[r][j][q]*((C_rj[r][j])@optimal_z[r] + d_optval[j,r]) for j in range(k)) == gp.quicksum(mu[i,r]*A[i,:] for i in range(m))[q]
                              for r in range(R) for q in range(d)), name='consistency-l2_a')
            if tol==0.:
                model.addConstrs((mu[i,r]*(A[i,:]@optimal_z[r] -b[i]) == 0
                                  for r in range(R) for i in range(m)), name="consistency-l2_b")
            else:
                model.addConstrs((mu[i,r]*(A[i,:]@optimal_z[r] -b[i]) <= 0 +tol
                                  for r in range(R) for i in range(m)), name="Uconsistency-l2_b")
                model.addConstrs((mu[i,r]*(A[i,:]@optimal_z[r] -b[i]) >= 0 -tol
                                  for r in range(R) for i in range(m)), name="Lconsistency-l2_b")
    """
    """
    #general case
    if general_case:
        eta = model.addMVar(shape = (n_j[0],k), vtype=gp.GRB.BINARY, name="eta")
        model.addConstrs((B_tilde[j][p][q] <= eta[p,j]
                            for j in range(k) for p in range(n_j[j]) for q in range(d)), name='Uabs_assoc')
        model.addConstrs((B_tilde[j][p][q] >= -eta[p,j]
                            for j in range(k) for p in range(n_j[j]) for q in range(d)), name='Labs_assoc')
        model.addConstrs((gp.quicksum(eta[s,j] for j in range(k)) <=1
                          for s in range(n_j[0])), name="assoc_most_to1")    
        
        #SYMMETRY BREAKING CONSTRAINTS
        #variable contador binaria
        phi = model.addMVar(shape = (n_j[0],k), vtype=gp.GRB.BINARY, name="phi")
        model.addConstrs((gp.quicksum(phi[s,j] for j in range(k)) <= 1 
                          for s in range(n_j[0])), name="exactly_one_number")
        #prohibited
        model.addConstrs((gp.quicksum(eta[s,j] for j in range(j_star+1,k)) <= 1 - phi[s,j_star]
                          for s in range(n_j[0]) for j_star in range(k-1)), name="prohibited")

        #variable contador 'al menos' auxiliar
        varphi = model.addMVar(shape=(n_j[0],k), vtype=gp.GRB.BINARY, name="varphi")
        model.addConstrs((varphi[0,j] == eta[0,j] 
                          for j in range(k)), name="def_1")
        #model.addConstrs((varphi[s,j] == gp.max_(varphi[s-1,j] , eta[s,j])
         #                 for s in range(1,n_j[0]) for j in range(k)), name="def_s")
        #equivalente al maximo:
        model.addConstrs((varphi[s,j] >= varphi[s-1,j]
                          for s in range(1,n_j[0]) for j in range(k)), name="def_s1")
        model.addConstrs((varphi[s,j] >= eta[s,j]
                          for s in range(1,n_j[0]) for j in range(k)), name="def_s1")
        model.addConstrs((varphi[s,j] <= varphi[s-1,j] + eta[s,j]
                          for s in range(1,n_j[0]) for j in range(k)), name="def_s1")

        model.addConstrs((gp.quicksum((j+1)*phi[s,j] for j in range(k)) == gp.quicksum(varphi[s,j] for j in range(k)) 
                          for s in range(n_j[0])), name="equiv")
        
        model.addConstr(gp.quicksum(varphi[0,j] for j in range(k)) <= 1, name="conteo_1_sup")
        model.addConstrs((gp.quicksum(varphi[s-1,j] for j in range(k)) <= gp.quicksum(varphi[s,j] for j in range(k))
                          for s in range(1,n_j[0])), name="conteo_s_inf")
        model.addConstrs((gp.quicksum(varphi[s,j] for j in range(k)) <= 1 + gp.quicksum(varphi[s-1,j] for j in range(k))
                          for s in range(1,n_j[0])), name="conteo_s_sup")
    """    

    #OBJECTIVE FUNCTION
    expression = 0.
    for (lamb, obj) in objfun.items():
        if obj == "min_prior" and lamb>0.:
            if prior is not None:
                #expression = gp.quicksum((theta_tilde[k][j,q]-prior[k][j,q])**2 for k in range(K) for j in range(d_x_k[k]) 
                expression += lamb*gp.quicksum((theta_tilde[k][j,q]- norms[k]*prior[k][j,q])**2 for k in range(K) for j in range(d_x_k[k]) 
                                               for q in range(d_z))
        if obj == "max_sparsity" and lamb>0.:
            delta = {}
            K=1#SOLO SPARSITY PRIMERA MATRIZ
            for k in range(K):
                delta[k] = model.addMVar(shape=(d_x_k[k],d_z), vtype=gp.GRB.BINARY, name="sparse_entries")
            #NO CUENTO SPARSITY ÚLTIMA COLUMNA
            model.addConstrs((theta_tilde[k][j,q] <= delta[k][j,q] for k in range(K) for j in range(d_x_k[k]) for q in range(d_z-1)))#range(d_z)))
            model.addConstrs((-theta_tilde[k][j,q] <= delta[k][j,q] for k in range(K) for j in range(d_x_k[k]) for q in range(d_z-1)))#range(d_z)))
            expression += lamb*gp.quicksum(delta[k][j,q] for k in range(K) for j in range(d_x_k[k]) for q in range(d_z))

    model.setObjective(expr = expression, sense=gp.GRB.MINIMIZE)
    """
    if objfun == "min_distideal":
        if gamma == "l_1":
            model.setObjective(expr = 1/R * gp.quicksum(C_rj[r][j]@(optimal_z[r]-v[:,j,r]) for j in range(k) for r in range(R)), sense=gp.GRB.MINIMIZE)
        else:
            each_obj = model.addMVar(shape=(R), lb=0., ub=float('inf'), name="each_obj_sample")
            model.setObjective(expr = 1/R * gp.quicksum(each_obj[r] for r in range(R)), sense=gp.GRB.MINIMIZE)
            if gamma == "l_inf":
                model.addConstrs((C_rj[r][j]@(optimal_z[r]-v[:,j,r]) <= each_obj[r]
                                    for r in range(R) for j in range(k)), name="each_obj")
            if gamma == "l_2":
                model.addConstrs((gp.quicksum( (C_rj[r][j]@optimal_z[r] + d_optval[j,r])*(C_rj[r][j]@optimal_z[r] + d_optval[j,r]) for j in range(k)) <= each_obj[r]**2
                                    for r in range(R)), name="each_obj")
    
    if objfun == "min_nonzeros" or sparsity_constraints is not None:
        l, c, delta = {}, {}, {}
        for j in range(k):
            l[j] = model.addMVar(shape=(n_j[j]), vtype=gp.GRB.BINARY, name="l_indicator_row_"+str(j))
            l[j].PoolIgnore = 1
            c[j] = model.addMVar(shape=(d), vtype=gp.GRB.BINARY, name="c_indicator_column_"+str(j))
            c[j].PoolIgnore = 1
            delta[j] = model.addMVar(shape=(n_j[j],d), vtype=gp.GRB.BINARY, name="delta_indicator_element_"+str(j))
            model.addConstrs((delta[j][p,q] == l[j][p]*c[j][q]
                                for p in range(n_j[j]) for q in range(d)), name="aux_delta_j")
        
        model.addConstrs((B_tilde[j][p][q] <= delta[j][p,q]
                            for j in range(k) for p in range(n_j[j]) for q in range(d)), name='Uabs')
        model.addConstrs((B_tilde[j][p][q] >= -delta[j][p,q]
                            for j in range(k) for p in range(n_j[j]) for q in range(d)), name='Labs')
        
        if objfun == "min_nonzeros":
            model.setObjective(expr = gp.quicksum(delta[j][p,q]
                                                  for j in range(k) for p in range(n_j[j]) for q in range(d)),
                               sense = gp.GRB.MINIMIZE)
        else:
            #parameters (total nonzero matrices, total nonzero rows per matrix, total nonzero columns per matrix)
            (N_E, N_T, N_A, N_V) = sparsity_constraints
            print("SPARSITY CONSTRAINTS CONSIDERED WITH PARAMETERS (N_E, N_T, N_A, N_V)=",(N_E, N_T, N_A, N_V))
            
            if N_E is not None:
                model.addConstr(gp.quicksum(delta[j][p,q] for j in range(k) for p in range(n_j[j]) for q in range(d)) <= N_E, name='29a')
            
            if N_T is not None:
                tau = model.addMVar(shape=(k), vtype=gp.GRB.BINARY, name="tau")
                model.addConstrs((gp.quicksum(delta[j][p,q] for p in range(n_j[j]) for q in range(d)) >= tau[j]
                                  for j in range(k)), name="total_matrices2")
                model.addConstrs((gp.quicksum(delta[j][p,q] for p in range(n_j[j]) for q in range(d))* (1- tau[j]) == 0
                                  for j in range(k)), name="total_matrices3")
                model.addConstr(gp.quicksum(tau[j] for j in range(k)) <= N_T, name="total_matrices")
            if N_A is not None:
                model.addConstrs((gp.quicksum(l[j]) <= N_A[j] for j in range(k)), name='total_rows')
            if N_V is not None:
                model.addConstrs((gp.quicksum(c[j]) <= N_V[j] for j in range(k)), name='total_columns')

    if objfun == "min_prior":
        model.setObjective(expr = gp.quicksum((B_tilde[j][p,q]-prior[j][p,q])**2 for j in range(k) for p in range(n_j[j]) for q in range(d)), 
                           sense=gp.GRB.MINIMIZE)
    """

    model.update()
    model.optimize()
    
    if check_feasibility_orig is not None:
        if model.status == 2:
            print('THE ORIGINAL (w,B) IS A FEASIBLE SOLUTION OF MY INVERSE PROBLEM')
        else: #https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html#sec:StatusCodes
            print('THE ORIGINAL (w,B) IS NOT A FEASIBLE SOLUTION OF MY INVERSE PROBLEM')
            orignumvars = model.NumVars
            # https://www.gurobi.com/documentation/current/refman/py_model_feasrelaxs.html
            model_relax = model.copy()
            model_relax.feasRelax(relaxobjtype=1, minrelax=False, vars=None, lbpen=None, ubpen=None,
                                  constrs=[c for c in model_relax.getConstrs() if (("consistency" in c.ConstrName) or ("dualfeas" in c.ConstrName) or ("unit_SumNorm"==c.ConstrName))],
                                  rhspen=None)
            model_relax.optimize()

            if model_relax.status==2:
                # print the values of the artificial variables of the relaxation
                print('\nSlack values:')
                slacks = model_relax.getVars()[orignumvars:]
                for sv in slacks:
                    if sv.X > 1e-9:
                        change_type = sv.VarName[:4] #https://support.gurobi.com/hc/en-us/articles/8444209982737-How-do-I-change-variable-and-or-constraint-bounds-to-make-an-infeasible-model-feasible-using-feasRelax
                        if change_type == 'ArtU':
                            change_type = 'Increase upper bound of the variable '
                        if change_type == 'ArtL':
                            change_type = 'Decrease lower bound of the variable '
                        if change_type == 'ArtP':
                            change_type = 'Decrease RHS (right-hand side) of the constraint ' 
                        if change_type == 'ArtN':
                            change_type = 'Increase RHS (right-hand side) of the constraint '

                        print(change_type+sv.VarName[5:]+f" by: {sv.X:.9f}")
            else:
                print('THE ORIGINAL (w,B) IS NOT A FEASIBLE SOLUTION OF MY INVERSE PROBLEM EVEN IF RELAXING IT')

    return model, theta_tilde


def return_estimates(theta_estimates):
    
    print("\n PARAMETROS BUSCADOS:")
    theta_inv = {}
    w_inv = []
    k = 0
    for theta in theta_estimates:
        w_inv_k = np.sqrt(np.sum([t**2 for t in theta])) #por defecto uso matrix_norm==l2
        w_inv.append(w_inv_k)
        theta_inv[k] = theta/w_inv_k if w_inv_k !=0 else theta
        print("theta_inv_",k,"=\n", theta_inv[k])
        k+=1
    print("w_inv=", w_inv)
    return np.array(w_inv), theta_inv


def eval_estimates(w_inv, theta_inv, w_orig, theta_orig, x_test,N,N_test):

    #preferences
    if len(w_orig) ==2:
        bins = range(len(w_orig))
        #https://medium.com/@srivastava.abh/earth-mover-distance-88dfa03ae9cb
        emd = wasserstein_distance(u_values = bins, v_values = bins, u_weights = w_orig, v_weights = w_inv)
        #emd = wasserstein_distance(w_orig, w_inv)
    else:
        M = 1 - np.eye(len(w_orig))
        emd = ot.emd2(w_orig, w_inv, M)

    #objectives x*theta
    cos_sims = {}
    for k in range(len(theta_orig)):
        cos_sim = []
        for n in range(N,N+N_test):
            cos_sim.append( cosine_similarity((x_test[n][k]@theta_orig[k]).reshape(1, -1),
                                         (x_test[n][k]@theta_inv[k]).reshape(1, -1))[0][0]
            )
        cos_sims[k] = (np.mean(cos_sim),np.std(cos_sim),np.median(cos_sim))
    
    return emd, cos_sims
    


##########################




def val_fobj_forward(x_r,k,w,B,gamma,z, vect_vs):

    C = function_C(x_r,w,B)

    if gamma == "l_1":
        val = np.sum([C[j,:]@(z - vect_vs[j]) for j in range(k)])
    if gamma == "l_inf":
        val = np.max([C[j,:]@(z - vect_vs[j]) for j in range(k)])
    if gamma == "l_2":
        val = np.sqrt(np.sum([(C[j,:]@(z - vect_vs[j]))**2 for j in range(k)]))

    return val


def values_for_gap_or_consistency(gamma, k,
                                  w, B,
                                  x, z_inv, z_orig, vect_v):
    """
    if (w, B, vect_v) = (w_orig, B_orig, vect_v_orig) --> optimalitygap
    if (w, B, vect_v) = (w_orig_inv, B_orig_inv, vect_v_orig_inv) --> consistency
    """
    val_orig = val_fobj_forward(x,k,w,B,gamma,z_orig, vect_v)
    val_inv = val_fobj_forward(x,k,w,B,gamma,z_inv, vect_v)
    return (val_orig, val_inv)


def checkeo_sols(R,R_test, gamma, d, m, n, n_j, k, 
                 list_A,b,
                 w_inv, B_inv,#vars,
                 w_orig,B_orig,
                 x, optimal_z,opt_criteriawise_v,
                 x_test, optimal_z_test,opt_criteriawise_v_test):

    equal_sol = 0
    consistencys = []
    suboptimalitygaps_in = []
    for r in range(R):
        x_r = [x[r][j] for j in range(k)]
        z_r_inv, opt_criteriawise_v_inv, model_inv = forward_problem(x_r,w_inv,B_inv,list_A[r],b,gamma,d,k)
        if z_r_inv is None:
            print("For r=",r," couldn't solve the forward problem with the inverse parameters")
            continue
        #print("\n\nr=",r, "contexto x_r=",x_r,"\n original_optimal_z=", np.round(optimal_z[r],2),"z_r_segun_inv=",z_r_segun_inv)
        #print("and opt_criteriawise_v_segun_inv:",opt_criteriawise_v_segun_inv)   
        equal_sol += all(z_r_inv == optimal_z[r])
             
        #MIDO SI TIENEN MISMO VALOR DE LA FUNCIÓN OBJETIVO CON LOS PARÁMETROS ESTIMADOS POR EL PROBLEMA INVERSO
        consistency_val_orig, consistency_val_inv = values_for_gap_or_consistency(gamma, k,
                                                                                  w_inv, B_inv,
                                                                                  x_r, z_r_inv, optimal_z[r],
                                                                                  list(opt_criteriawise_v_inv.values()))
                                                                                  #[vars['v'][:,j,r].X for j in range(k)])
        # LAS SOLUCIONES optimal_z[r] Y z_r_segun_inv DEBEN TENER MISMO VALOR OBJETIVO DEL PROBLEMA FORWARD CON MIS ESTIMACIONES INVERSAS
        consistencys.append( round(consistency_val_orig - consistency_val_inv, 5) )

        # EL SUBOPTIMALITY GAP VA MEDIDO EVALUANDO BAJO LOS PARÁMETROS ORIGINALES
        gap_val_orig, gap_val_inv = values_for_gap_or_consistency(gamma, k,
                                                                  w_orig, B_orig,
                                                                  x_r, z_r_inv, optimal_z[r],
                                                                  list(opt_criteriawise_v[r].values()))
        suboptimalitygaps_in.append( (gap_val_orig, gap_val_inv) )
    

    # EL SUBOPTIMALITY GAP VA MEDIDO EVALUANDO BAJO LOS PARÁMETROS ORIGINALES
    suboptimalitygaps_out = []
    for r in range(R,R+R_test):
        x_r = [x_test[r][j] for j in range(k)]
        z_r_inv, opt_criteriawise_v_inv, _ = forward_problem(x_r,w_inv,B_inv,list_A[r],b,gamma,d,k)
        if z_r_inv is None:
            print("For r_test=",r," couldn't solve the forward problem with the inverse parameters")
            continue
        gap_val_orig, gap_val_inv = values_for_gap_or_consistency(gamma, k,
                                                                  w_orig, B_orig,
                                                                  x_r, z_r_inv, optimal_z_test[r],
                                                                  list(opt_criteriawise_v_test[r].values()))
        suboptimalitygaps_out.append( (gap_val_orig, gap_val_inv) )

    #summary
    print("equal_decisionvec==R?",equal_sol,"==",R,":",equal_sol==R)
    consistency_achieved = math.isclose(np.mean(consistencys) , 0. , abs_tol=1e-5)
    print("consistency_achieved? len(consistencys):",len(consistencys),"==",R,"? achieved:",consistency_achieved)
    if not consistency_achieved:
        print("Error in consistency")
    
    med_in = np.nanmedian([compute_gap(a,b) for (a,b) in suboptimalitygaps_in])
    med_out = np.nanmedian([compute_gap(a,b) for (a,b) in suboptimalitygaps_out])
    print("median fractional suboptimality gap in sample: ",med_in)
    print("median fractional suboptimality gap out sample: ",med_out)
    

    return consistency_achieved, suboptimalitygaps_in,suboptimalitygaps_out


def compute_gap(orig,inv):
    if orig!=0:
        gap = inv/orig 
    else:
        if inv==0:
            gap = 1. 
        else:
            gap = np.inf
    return gap
