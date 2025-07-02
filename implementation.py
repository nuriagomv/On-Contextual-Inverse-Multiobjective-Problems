import os
#os.chdir(r'/Users/nuriagomezvargas/Library/CloudStorage/OneDrive-UNIVERSIDADDESEVILLA/Académico/PhDiva/Inverse Multiobjective Optimization')
import numpy as np
np.set_printoptions(suppress=True,precision=3)
import auxfuncs as aux
from data_stocks import get_real_portfolio_instances
import pickle
import pandas as pd
#import gurobipy as gp
#from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
#import math
import random
import time




def executeIO(seed, N, N_test, n_stocks, n_market,
              timelimit = 1800, objfun = {1.:"min_prior"}, solve_to = "optimality",
              start = "2024-01-01", end = "2024-12-31", 
              pista = True,
              what_in_context = "past_t-1", volume_in_objective = False,
              gamma = "l_1", tol = 0.,
              noise = False, general_case = False,
              sparse_model = True):
    
    np.random.seed(seed)
    random.seed(seed)
    
    #INSTANCES
    instances = get_real_portfolio_instances(seed, n_stocks, n_market,
                                             N, N_test, gamma, 
                                             noise,
                                             start, end,
                                             what_in_context, volume_in_objective,
                                             sparse_model)
    (data_final, stocks, market_features), (d_x,d_x_k,d_z,m,K), (w_orig,theta_orig), (list_A,b), (x,x_test), (optimal_z,optimal_z_test), (opt_criteriawise_v,opt_criteriawise_v_test) = instances.values()
    

    print("w_orig:", w_orig)
    print("theta_orig_0:\n", theta_orig[0])
    percentage_zeros_orig = (np.sum(theta_orig[0][:,:-1]==0))/np.prod((theta_orig[0][:,:-1]).shape)
    #VARIABILIDAD DE SOLUCIONES
    unique_z = list(set(tuple(sublist) for sublist in optimal_z.values()))
    #for z in unique_z:
     #   print(np.array(z).round(2))
    n_unique_z = len(unique_z)
    
    unique_x = list(set(tuple(sublist[0]) for sublist in x.values()))
    #for x_1 in unique_x:
     #  print(np.array(x_1).round(2))
    n_unique_x = len(unique_x)



    # INVERSE PROBLEM
    print('\n -----------------------------------------------------\nINVERSE PROBLEM')
    #check feasibility of original (w,theta)
    _, _ = aux.inverse_problem(N, x, optimal_z, list_A, b, d_x, d_x_k, d_z, m, K,
                               gamma, {1:0.}, timelimit, solve_to,
                               general_case,
                               check_feasibility_orig = (w_orig, theta_orig),
                               tol=tol)
    
    #solve to find inverse estimates
    #objfun = {0.9: "min_prior", .1:"max_sparsity"}
    prior = {}
    for k in range(K):
        #prior[k] = w_orig[k]*theta_orig[k] + np.random.uniform(-0.05,0.05,size = theta_orig[k].shape)
        prior[k] = theta_orig[k] + np.random.uniform(-0.25,0.25,size = theta_orig[k].shape)
        prior[k] *= np.random.randint(0,2,size=prior[k].shape)
    
    if noise:
        L = 5
        #DEBERIA HACER CLUSTER SOLO DE  X_1
        x_all = [np.concatenate(tuple(x[n])) for n in range(N)]
        kmedoids = KMedoids(n_clusters = L, random_state=seed)
        kmedoids.fit(np.array(x_all))
        # Get centroids
        centroids = kmedoids.cluster_centers_
        print("Centroids:\n", centroids) 
        """
        labels = kmedoids.labels_
        clusters = np.zeros((R,L))
        for r in range(R):
            clusters[r,labels[r]] = 1.
        """
        subset = []
        for c in centroids:
            subset+=[n for n in range(N) if (x_all[n] ==c).all()]
        x_meds,z_opt_meds, n_meds = {},{},0
        for n in range(N):
            if n in subset:
                x_meds[n_meds],z_opt_meds[n_meds] = x[n],optimal_z[n]
                n_meds+=1
        model, theta_tilde = aux.inverse_problem(n_meds, x_meds, z_opt_meds,
                                                 A, b, d_x, d_x_k, d_z, m, K,
                                                 gamma, objfun, timelimit, solve_to,
                                                 general_case,
                                                 prior=prior,
                                                 tol=0.0005)

    inicio = time.time()
    model, theta_tilde = aux.inverse_problem(N, x, optimal_z, list_A, b, d_x, d_x_k, d_z, m, K,
                                             gamma, objfun, timelimit, solve_to,
                                             general_case,
                                             prior=prior,
                                             tol=tol,
                                             pista = pista)
    tiempo = time.time()-inicio
    MIP_gap = model.MIPGap

    w_inv, theta_inv = aux.return_estimates([theta.X for theta in theta_tilde.values()])

    emd, cos_sims = aux.eval_estimates(w_inv, theta_inv, w_orig, theta_orig, 
                                       x_test,N,N_test) #YA CORREGIDO 
    #COS_SIMS ES (MEDIA,STD,MEDIAN)
    
    #n_sparse_coefs = np.sum([t == 0 for t in theta_inv[0]]) - d_x_k[0]

    consistency_achieved, suboptimalitygaps_in, suboptimalitygaps_out = aux.checkeo_sols(N, N_test, gamma, d_z, m, d_x, d_x_k, K, 
                                                                                         list_A,b,
                                                                                                    w_inv, theta_inv,
                                                                                                    w_orig, theta_orig,
                                                                                                    x, optimal_z,opt_criteriawise_v,
                                                                                                    x_test, optimal_z_test,opt_criteriawise_v_test)                                                                                  

    return tiempo, MIP_gap, instances, percentage_zeros_orig, (n_unique_z,n_unique_x), prior, w_inv, theta_inv, emd, cos_sims, consistency_achieved, suboptimalitygaps_in, suboptimalitygaps_out


N_test = 100
n_market = 6
pista=True
sparse_model = True
what_in_context = "past_t-1"

gamma="l_2"
tol=0.005

results = []
dicts = []
for seed in [123,456,789]:#[123,456,789]:
    for N in [1]:#[25,50,100,150,200]:
        for n_stocks in [5]:
            for objfun in [{1.: "min_prior"},
                           {0.9: "min_prior", .1:"max_sparsity"}]:#,,
                           #{0.8: "min_prior", .2:"max_sparsity"}]:
                
            
                with open('results_dicts_SPARSITY.pkl', 'wb') as f:
                    pickle.dump((results,dicts), f, protocol=pickle.HIGHEST_PROTOCOL)


                print("SETING: ",seed,N_test,n_market,N,pista,n_stocks)

                outputs = executeIO(seed, N, N_test, n_stocks, n_market, 
                                    objfun = objfun, 
                                    gamma=gamma, tol=tol,
                                    what_in_context=what_in_context, pista = pista,
                                    start = "2023-01-01", sparse_model=sparse_model)
                results.append(outputs)
                tiempo, MIP_gap, instances, percentage_zeros_orig, (n_unique_z,n_unique_x), prior, w_inv, theta_inv, emd, cos_sims, consistency_achieved, suboptimalitygaps_in, suboptimalitygaps_out = outputs
                (data_final, stocks, market_features), (d_x,d_x_k,d_z,m,K), (w_orig,theta_orig), (list_A,b), (x,x_test), (optimal_z,optimal_z_test), (opt_criteriawise_v,opt_criteriawise_v_test) = instances.values()    
                
                med_in = np.nanmedian([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_in])
                med_out = np.nanmedian([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_out])
                mean_in = np.nanmean([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_in])
                mean_out = np.nanmean([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_out])
                std_in = np.std([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_in])
                std_out = np.std([aux.compute_gap(a,b) for (a,b) in suboptimalitygaps_out])
     

                zeros_orig = (np.sum(w_inv[0]*theta_orig[0]==0)-d_z)#/np.prod(theta_orig[0].shape)
                zeros_results = (np.sum(w_inv[0]*theta_inv[0]==0)-d_z)#/np.prod(theta_orig[0].shape)

                print("RESULTS: ", emd, cos_sims, consistency_achieved, suboptimalitygaps_in, suboptimalitygaps_out)

                dict = {'seed': seed, 'objfun':objfun, 'N':N, 'n_stocks':n_stocks, 'tiempo':tiempo, 'MIP_gap':MIP_gap, 'percentage_zeros_orig':percentage_zeros_orig, 'n_unique':(n_unique_z,n_unique_x), 
                        'emd':emd, 'cos_sims':cos_sims, #cos_sims = (mean,std,median)
                        'consistency':consistency_achieved,
                        'suboptimalitygaps_in':(mean_in,std_in,med_in), 
                        'suboptimalitygaps_out':(mean_out,std_out,med_out),
                        'zeros_orig': zeros_orig, 'zeros_results':zeros_results}
                dicts.append(dict)

            

########################################


pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)


with open('results_dicts_l2.pkl', 'rb') as f:
   (results,dicts) = pickle.load(f)





d = pd.DataFrame(dicts)
d['cos_sims'] = d['cos_sims'].apply(lambda x: x[0])

d.to_excel('dicts_results.xlsx')
