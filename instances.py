import numpy as np
np.set_printoptions(suppress=True,precision=3)
#import auxfuncs as aux
import auxfuncs as aux


def get_instances(R, R_test, gamma, seed,
                  n_j = np.array([2,2,2])):
    """
    Attributes:
    R: int
        sample size.
    gamma: str
        string indicating which norm to use.
    n_j: np.array
        size of each partition of the context,
        length(n_j) == k number of criterias.
    """

    print('\n\n---------------------------\n\n GENERATING INSTANCES FOR NEW EXPERIMENT')

    #experiments are reproducible
    np.random.seed(seed)


    # From (Hwang et al.,1993) Computers&OR
    # CONSTRAINTS A·z <= b
    A = np.array([[720,107,7080,0,134,1000],
                   [0.2,10.1,13.2,0.75,0.15,1.2],
                   [344,1460,1040,75,17.4,240],
                   [18,151,78,2.5,0.2,4]])
    A = - np.concatenate((A, -np.identity(6), np.identity(6)), axis=0)
    b = np.array([5000,12.5,2500,63,-6,-1,-0.25,-10,-10,-4])
    b = - np.concatenate((b,np.zeros(6)),axis=0)

    d = A.shape[1] #number of decision variables in z
    m = A.shape[0] #number of constraints (rows in A)
    print("In total we have m=",m," constraints for d=",d," decision variables. Defined by A=\n",A, "\n and b=\n",b)


    # MODEL PARAMETERS
    k = len(n_j) #number of criteria
    n = np.sum(n_j) #total size of the context vectors x_r  
    print("\n sample size, R=",R, " and test size R_test=",R_test,
          "\n number of different criterias in the objective function, k=",k,
          "\n size of each partition of the context, n_j=",n_j,
          "\n checkeo len(n_j) == k = ",len(n_j) == k,
          "\n total context size, n=",n,
          "\n distance gamma=",gamma)


    # CRITERIA
    # weights from (Hwang et al.,1993) Computers&OR
    w_orig = np.array([0.3,0.5,0.2])
    if len(w_orig) != k:
        raise Exception("Error length(w_orig) == k")
    print("w_orig=",w_orig)

    # matrices
    B_0 = np.random.exponential(size=(n_j[0],d), scale=1)#np.random.uniform(-10,-1, size=(n_j[0],d))
    B_1 = -np.random.exponential(size=(n_j[1],d), scale=7)
    B_2 = np.random.exponential(size=(n_j[2],d), scale=16)#np.random.uniform(1,10, size=(n_j[2],d))
    #normalized matrices
    B_orig = {0: B_0/np.linalg.norm(B_0.flatten(), ord=2), #B_0/np.abs(B_0).max(), #(B_0 - B_0.min())/(B_0.max() - B_0.min()),
              1: B_1/np.linalg.norm(B_1.flatten(), ord=2), #B_1/np.abs(B_1).max(), #(B_1 - B_1.min())/(B_1.max() - B_1.min()),
              2: B_2/np.linalg.norm(B_2.flatten(), ord=2)  #B_2/np.abs(B_2).max() #(B_2 - B_2.min())/(B_2.max() - B_2.min())
              }
    for j in range(k):
        print("B_",j,"orig=\n",np.round(B_orig[j],4))
        print("Norma 1?", np.linalg.norm(B_orig[j].flatten(),ord=2), "== 1?",np.linalg.norm(B_orig[j].flatten(),ord=2)==1)#np.inf) == 1)
    #context
    megaR=500
    x_0 = np.round(np.random.uniform(-20,50, size=(megaR,n_j[0])),2)
    x_1 = 10*np.random.randint(low=-2,high=3, size=(megaR,n_j[1]))#np.round(np.random.uniform(-50,20, size=(megaR,n_j[0])),2)#
    x_2 = np.round(np.random.uniform(-30,30, size=(megaR,n_j[2])),2)
    x = {0: np.concatenate((x_0[:R,:], x_0[-R_test:,:]),axis=0),
         1: np.concatenate((x_1[:R,:], x_1[-R_test:,:]),axis=0),
         2: np.concatenate((x_2[:R,:], x_2[-R_test:,:]),axis=0)}
    

    # FORWARD PROBLEM
    print('\n -----------------------------------------------------\n')
    print("computing FORWARD PROBLEM...")

    optimal_z = {}
    opt_criteriawise_v = {}
    for r in range(R+R_test):
        x_r = [x[j][r] for j in range(k)]
        z_r, opt_criteriawise, _ = aux.forward_problem(x_r,w_orig,B_orig,A,b,gamma,d,k,m)
        optimal_z[r] = z_r
        opt_criteriawise_v[r] = opt_criteriawise
        if r<R: #NO HAGO PRINT PARA EL CONJUNTO TEST AUNQUE LOS GENERO TODOS A LA VEZ
            print("\n\nFor context x_r=\n",x_r,"\n optimal solution is z_r=",np.round(z_r,2))
            if opt_criteriawise is not None:
                for j in range(k):
                    print(f"and optimal criteriawise v_{j}:",opt_criteriawise[j])

    dict_instances = {'n_j': n_j,
                      'n': n,
                      'k': k,
                      'm': m,
                      'd': d,
                      'A': A,
                      'b': b,
                      'w_orig': w_orig,
                      'B_orig': B_orig,
                      'x': x,
                      'optimal_z': optimal_z,
                      'opt_criteriawise_v': opt_criteriawise_v}
    return dict_instances

    

def all_components_of_instances(R, R_test, gamma, seed):

    dict_instances = get_instances(R, R_test, gamma, seed)
    n_j,n,k,m,d,A,b,w_orig,B_orig,x_total,optimal_z_total,opt_criteriawise_v_total = dict_instances.values()

    x,x_test = {},{}
    optimal_z, optimal_z_test = {},{}
    opt_criteriawise_v, opt_criteriawise_v_test = {},{}
    for r in range(R):
        optimal_z[r] = optimal_z_total[r]
        opt_criteriawise_v[r] = opt_criteriawise_v_total[r]
    reind = 0 
    for r in range(R,R+R_test):
        optimal_z_test[reind] = optimal_z_total[r]
        opt_criteriawise_v_test[reind] = opt_criteriawise_v_total[r]
        reind+=1
    for j in range(k):
        x[j] = x_total[j][:R]
        x_test[j] = x_total[j][R:]
    
    return (n_j, n, k, m, d, A, b, w_orig, B_orig,\
            x_total, optimal_z_total, opt_criteriawise_v_total,\
            x, x_test, optimal_z, optimal_z_test, opt_criteriawise_v, opt_criteriawise_v_test)