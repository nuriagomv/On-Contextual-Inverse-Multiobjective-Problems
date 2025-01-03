import os
#os.chdir(r'/Users/nuriagomezvargas/Library/CloudStorage/OneDrive-UNIVERSIDADDESEVILLA/Académico/PhDiva/Inverse Multiobjective Optimization')

import numpy as np
import auxfuncs as aux
from instances import all_components_of_instances
import pickle
import pandas as pd
import gurobipy as gp
import math

def executeIO_with_parameters(seed, R,R_test,gamma, timelimit,objfun, solve_to,sparsity_constraints, general_case):

    print('\n\n\nEXPERIMENT: ',(seed, R,R_test,gamma))#,solve_to,sparsity))

    #INSTANCES
    instances = all_components_of_instances(R,R_test,gamma, seed)
    n_j,n,k,m,d,A,b,w_orig,B_orig, x_total,optimal_z_total,opt_criteriawise_v_total, x,x_test, optimal_z, optimal_z_test, opt_criteriawise_v, opt_criteriawise_v_test = instances
    
    if general_case:
        x_general = []
        for r in range(R):
            x_complete = np.concatenate([x[j][r] for j in range(k)],axis=0)
            x_general.append(x_complete)
        x_general_test = []
        for r in range(R_test):
            x_complete = np.concatenate([x_test[j][r] for j in range(k)],axis=0)
            x_general_test.append(x_complete)
        for j in range(k):
            x[j] = np.array(x_general)
            x_test[j] = np.array(x_general_test)
            B_orig[j] = np.concatenate([np.zeros((n_j[l],d)) if l!=j else B_orig[l] for l in range(k)], axis=0)    
        n_j = np.repeat(np.sum(n_j), len(n_j))
        #ME FALTA HACERLO CON EL TEST 
        instances = n_j,n,k,m,d,A,b,w_orig,B_orig, x_total,optimal_z_total,opt_criteriawise_v_total, x,x_test, optimal_z, optimal_z_test, opt_criteriawise_v, opt_criteriawise_v_test

    # INVERSE PROBLEM
    print('\n -----------------------------------------------------\nINVERSE PROBLEM')
    #check feasibility of original (w,B)
    _, _ = aux.inverse_problem(x,optimal_z, A,b,d,k,m,n_j,R, gamma, objfun=None,
                               check_feasibility_orig = (w_orig, B_orig), tol = 1e-2 if gamma=="l_2" else 0.)
    
    #solve to find inverse estimates
    model, B_tilde = aux.inverse_problem(x,optimal_z, A,b,d,k,m,n_j,R, gamma,
                                         objfun = objfun,
                                         timelimit = timelimit,
                                         solve_to = solve_to,
                                         sparsity_constraints = sparsity_constraints,
                                         general_case = general_case)

    return instances, model, B_tilde


def retrieve_solutions(instances, model, B_tilde, dict_execution):

    n_j,n,k,m,d,A,b,w_orig,B_orig, x_total,optimal_z_total,opt_criteriawise_v_total, x,x_test, optimal_z, optimal_z_test, opt_criteriawise_v, opt_criteriawise_v_test = instances
    unique_solutions = set()
    #https://www.gurobi.com/documentation/current/refman/retrieving_solutions.html
    # Loop through the solutions and print them
    for i in range(model.SolCount):
        model.setParam(gp.GRB.Param.SolutionNumber, i)
        solution = tuple(np.concatenate([B.Xn.flatten() for B in B_tilde.values()]))#tuple(v.Xn for v in model.getVars())
        if solution not in unique_solutions:
            unique_solutions.add(solution)
            print(f"\nSolution {i}:", solution)
            time_sol_i = model._solution_times[i] - model._start_time
            print(f"Solution {i} found at {time_sol_i:.2f} seconds with objval={model.PoolObjVal:.0f}")
            B_estimates = [B.Xn for B in B_tilde.values()]
            #print(B_estimates)
            w_inv, B_inv = aux.return_estimates(B_estimates)

            #veo cual ha sido la sparsity efectiva
            N_T_efect = np.sum(w_inv>0)
            N_A_efect = [np.sum([any(B_inv[j][p,:] !=0) for p in range(n_j[j])]) for j in range(k)]
            N_V_efect = [np.sum([any(B_inv[j][:,q] !=0) for q in range(d)]) for j in range(k)]
            sparsity_effective =  (N_T_efect, N_A_efect, N_V_efect)

            consistency_achieved, suboptimalitygaps_in, suboptimalitygaps_out = aux.checkeo_sols(R,R_test, gamma, d, m, n, n_j, k, A,b,
                                                                                                    w_inv, B_inv,
                                                                                                    w_orig,B_orig,
                                                                                                    x, optimal_z,opt_criteriawise_v,
                                                                                                    x_test, optimal_z_test,opt_criteriawise_v_test)
            dict_execution['solutions'][i] = {'obj_val': model.PoolObjVal,
                                                'time_found': time_sol_i,
                                                'w_inv': w_inv, 'B_inv': B_inv,
                                                'consistecy_achieved': consistency_achieved,
                                                'mean_suboptimalitygap_out': np.mean([b-a for (a,b) in suboptimalitygaps_out]),
                                                'median_suboptimalitygap_out': np.median([b-a for (a,b) in suboptimalitygaps_out]),
                                                'mean_suboptimalitygap_out_frac': np.nanmean([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_out]),
                                                'median_suboptimalitygap_out_frac': np.nanmedian([aux.compute_gap(a,b)  for (a,b) in suboptimalitygaps_out]),
                                                'mean_suboptimalitygap_in_frac': np.nanmean([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_in]),
                                                'median_suboptimalitygap_in_frac': np.nanmedian([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_in]),
                                                'suboptimalitygaps_in': suboptimalitygaps_in,
                                                'suboptimalitygaps_out': suboptimalitygaps_out}
    
    return dict_execution


executions = []
for seed in range(10):#5):
    for gamma in ["l_1"]:
        for R in [50]:#[1,5,10,20,30,50,100]:#5,10,15,20,25,30]:
             
            R_test = 100
            timelimit = 1200#600
            solve_to = "feasibility" # "feasibility" or "optimality"
            objfun = "min_distideal" # "min_distideal" or "min_nonzeros"
            general_case = True
            sparsity_constraints = None

            nombre= str(R)+"-"+gamma+"-"+solve_to+"-"+objfun+"-"+str(sparsity_constraints)+"-"+str(general_case)

            instances, model, B_tilde = executeIO_with_parameters(seed, R,R_test,gamma, timelimit,objfun, solve_to,sparsity_constraints, general_case)
            
            dict_execution = {'seed': seed, 'R': R, 
                              'R_test': R_test, 'timelimit':timelimit,
                              'instances': instances,
                              'model_status': model.status, 'total_time': model.Runtime,
                              'model_solcount': model.SolCount, 'solutions': {}}
            n_j,n,k=instances[:3]
            caract_instances = 'n='+str(n_j)+'_k='+str(k)
            # Retrieve the solutions found
            if model.SolCount > 0: #puede ser solved to optimality o time limit habiendo encontrado alguna sol factible
                dict_execution  = retrieve_solutions(instances, model, B_tilde, dict_execution)


                #NUEVO
                if general_case:
                    eta_obtained = np.zeros((n,k))
                    for i in range(n):
                      for j in range(k):
                        eta_obtained[i,j] = model.getVarByName('eta[%d,%d]'%(i,j)).X
                    eta_obtained
                    clusters = {}
                    for j in range(k):
                        clusters['B_'+str(j+1)] = ['x_'+str(i+1) for i in range(n) if eta_obtained[i,j]==1 ]

                    f = open("CLUSTERS-"+caract_instances+'__'+nombre+".txt", "a")
                    f.write(str(clusters)+'\n')
                    f.close()


                dict_execution['solutions_dataframe'] = pd.DataFrame(dict_execution['solutions'].values()).drop(['suboptimalitygaps_in','suboptimalitygaps_out'],axis=1)
            else:
                dict_execution['solutions_dataframe'] = pd.DataFrame(dict_execution['solutions'].values())
            executions.append(dict_execution)
            
            
            with open(caract_instances+'__'+nombre+'.pkl', 'wb') as handle:
                pickle.dump(executions, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


print('\n\n-------------------\n\n')
"""
executions = []
for element in os.listdir():
    if ("executions.pkl" in element):
        print(element)
        with open(element, 'rb') as handle:
            archivo = pickle.load(handle)
            executions+=archivo
"""
ex_dataframe = pd.DataFrame(executions)

for i in range(ex_dataframe.shape[0]):
    general_experiment = ex_dataframe.drop(['solutions','instances','solutions_dataframe'],axis=1).loc[i,:]
    print("\nGENERAL EXPERIMENT: \n", general_experiment)
    try:
        solutions = ex_dataframe.loc[i,'solutions_dataframe'].round(2)
        print("\nSOLUTIONS: \n", solutions)
    except:
        print("No solution found in this experiment")
