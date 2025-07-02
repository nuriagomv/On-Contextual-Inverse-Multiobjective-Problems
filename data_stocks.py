import yfinance as yf
import pandas as pd 
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import auxfuncs as aux


def data_yfinance(seed, n_stocks, n_market,
                  start, end):
    
    ref = str((seed, n_stocks, n_market, start, end))

    try:
        with open('stock_save'+ref+'.pkl', 'rb') as f:
            (data_final, stocks, market_features) = pickle.load(f)
    except:
        major_us_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "BRK-B", "JNJ", "V", "JPM", "WMT", 
                        "NVDA", "HD", "PG", "UNH", "MA", "DIS", "PYPL", "NFLX", "PEP", "KO"]
        
        np.random.seed(seed)
        random.seed(seed)
        stocks = random.sample(major_us_stocks, n_stocks)

        # Downloading data
        data = yf.download(stocks, start = start, end = end).reset_index()

        if data.shape[0]!=0:
            #Open	  Price at which the stock opened on a given trading day
            #High	  Highest price reached during the trading day
            #Low	  Lowest price reached during the trading day
            #Close	  Price at which the stock closed on that trading day
            #Volume	  Number of shares traded (i.e., trading volume) during the day
            data['DATE'] = data.loc[:,('Date', '')].astype('datetime64[ns]')
            data.set_index('DATE', inplace=True)

            #data.isna().sum().sum()

            ###################################################################################
            #RETURNS
            data_returns = data.loc[:,('Close', slice(None))]
            data_returns.columns = [b+"_Return" for (_,b) in data_returns.columns]
            # pct_change() Fractional change between the current and a prior element.
            # Computes the fractional change (CUANDO DICE CHANGE ES COMO EN NUMEROS INDICES, <1 O >1)
            # from the immediately previous row by default. 
            # This is useful in comparing the fraction of change in a time series of elements.
            data_returns = data_returns.pct_change()
            data_returns = data_returns.iloc[1:,:] #  quito la primera fila porque todo es NaN al no poder dividir entre nada anterior
            data_returns = data_returns*100 #percentage return
            data_returns.dropna(inplace = True)

            #VOLATILITY PROXY relative to the mean price that day
            data_high, data_low = data.loc[:,('High', slice(None))], data.loc[:,('Low', slice(None))]
            data_volatility = pd.DataFrame((np.array(data_high) - np.array(data_low))/((np.array(data_high) + np.array(data_low))/2))
            data_volatility.index, data_volatility.columns = data_high.index, [b+"_Volatility" for (_,b) in data_high.columns]
        
            #TRADING VOLUME
            data_volume = data.loc[:,('Volume', slice(None))]
            data_volume.columns = [b+"_Volume" for (_,b) in data_volume.columns]
        
            data_final = pd.merge(data_volatility, data_returns, on='DATE', how = 'inner')
            data_final = pd.merge(data_volume, data_final, on='DATE', how = 'inner')

            ###################################################################################
            #CONTEXTUAL INFORMATION
            market_features = ['^DJI', '^VIX', '^GSPC', 'CL=F', 'GC=F', '^TNX']
            market_features = random.sample(market_features, n_market)
            data_m = yf.download(market_features, start = start, end = end).reset_index()
            data_m['DATE'] = data_m.loc[:,('Date', '')].astype('datetime64[ns]')
            data_m.set_index('DATE', inplace=True)
            data_market = data_m.loc[:,('Open', slice(None))]
            data_market.columns = [b for (_,b) in data_market.columns]
            data_market.dropna(inplace = True)

            ###################################################################################

            #COMPLETE DATASET
            data_final = pd.merge(data_market, data_final, on='DATE', how = 'inner')

            #data_final.isna().sum().sum()

            with open('stock_save'+ref+'.pkl', 'wb') as f:
                pickle.dump((data_final, stocks, market_features), f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("ERROR IN DOWNLOAD")

    return data_final, stocks, market_features


def linear_surrogates(data_final, stocks, market_features,
                      what_in_context, volume_in_objective, sparse_model):
    #what_in_context in ["market", "past_t-1", "all"]
    #volume_in_objective in [True, False]

    d_z = len(stocks)
    d_x = len(market_features)

    #LINEAR OBJECTIVES
    theta_orig = {}
    x = {}
    objs = ["_Return","_Volume"] if volume_in_objective else ["_Return"]
    k = 0
    for obj in objs:
        #+1 in shape for intercept
        if what_in_context == "all":
            shape = (d_x + d_z + 1) 
        elif what_in_context == "past_t-1":
            shape = (d_z+1)
        elif what_in_context == "market":
            shape = (d_x+1)
        theta_orig[k] = np.zeros((shape, d_z+1))
        q=0
        for s in stocks:
            if what_in_context == "all":
                sub = data_final.loc[:,market_features + [(stock+obj) for stock in stocks]]
                sub_for_sparsity = data_final.loc[:,market_features + [(s+obj)]]
            elif what_in_context == "past_t-1":
                sub = data_final.loc[:,[(stock+obj) for stock in stocks]]
                sub_for_sparsity = data_final.loc[:,[(s+obj)]]
            elif what_in_context == "market":
                sub = data_final.loc[:,market_features]
                sub_for_sparsity = sub
            X = np.array(sub.iloc[:-1,:])
            X = np.hstack((X,np.ones((X.shape[0],1))))
            X_for_sparsity = np.array(sub_for_sparsity.iloc[:-1,:])
            X_for_sparsity = np.hstack((X_for_sparsity,np.ones((X_for_sparsity.shape[0],1))))
            y = np.array(data_final.loc[:,(s+obj)])[1:]
            x[k] = X

            # Create model
            model = LinearRegression(fit_intercept=False)
            # Fit model
            if (sparse_model) and (what_in_context != "market"):
                model.fit(X_for_sparsity,y)
                if what_in_context == "all":
                    for j in range(d_x):
                        theta_orig[k][j,q] = model.coef_[j]
                    theta_orig[k][d_x+q,q] = model.coef_[d_x]
                    theta_orig[k][-1,q] = model.coef_[-1]
                elif what_in_context == "past_t-1":
                    theta_orig[k][q,q] = model.coef_[0]
                    theta_orig[k][-1,q] = model.coef_[1]
            else:
                model.fit(X, y)
                theta_orig[k][:,q] = model.coef_
            #model.score(X, y)
            q+=1
            """
            # Make predictions
            y_pred = model.predict(X)
            # Coefficients
            print("Intercept:", model.intercept_)
            print("Slope:", model.coef_)
            
            # Plot results
            plt.scatter(y, y_pred, color='red')
            plt.plot(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), label='x = y', color='blue')
            plt.show()
            """
        theta_orig[k] = - theta_orig[k]/np.linalg.norm(theta_orig[k], 'fro')
        k+=1
    #VOLATILITY
    theta_orig[k] = np.hstack((np.zeros(d_z), np.ones(1))).reshape((1,d_z+1))
    x[k] = np.ones((x[k-1].shape[0],1))
    #K=k

    """
    np.array(data_final.loc[:,[stock+"_Volatility" for stock in stocks]]).argsort()
    a = np.array([x[0][n]@theta_orig[0] for n in range(N)]).argsort()
    len(set([str(tuple(elem)) for elem in list(a)]))
    """

    #CONSTRAINTS
    list_A = []
    for ind in data_final.index:
        A = np.vstack((np.hstack((-np.eye(d_z),
                                np.zeros((d_z,1)))),
                    np.hstack((np.diag(data_final.loc[ind,[stock+"_Volatility" for stock in stocks]]#.mean()
                                       ),
                                -np.ones((d_z,1)))),
                    np.hstack((np.ones((1,d_z)),
                                np.zeros((1,1,)))),
                    np.hstack((-np.ones((1,d_z)),
                                np.zeros((1,1,)))) ))
        list_A.append(A)
    b = np.hstack((np.zeros(d_z),
                   np.zeros(d_z),
                   np.ones(1),
                   -np.ones(1)))
    
    return theta_orig, x, list_A, b


def get_real_portfolio_instances(seed, n_stocks, n_market,
                                 N, N_test, gamma, 
                                 noise,
                                 start, end,
                                 what_in_context, volume_in_objective,
                                 sparse_model):

    data_final, stocks, market_features = data_yfinance(seed, n_stocks, n_market, start, end)

    theta_orig, x_total, list_A, b = linear_surrogates(data_final, stocks, market_features,
                                                  what_in_context, volume_in_objective,
                                                  sparse_model)
    K = len(theta_orig.keys())
    m,d_z = list_A[0].shape
    d_x_k = np.array([x_k.shape[1] for x_k in x_total.values()])
    d_x = np.sum(d_x_k) #total size of the context vectors x^n 
    
    #PREFERENCES
    w_orig = np.array([0.3,0.7])
    #w_orig = np.array([random.random() for _ in range(K)])
    #w_orig = w_orig/np.linalg.norm(w_orig,ord=1)
    
    # FORWARD PROBLEM
    print('\n -----------------------------------------------------\n')
    print("computing FORWARD PROBLEM...")

    optimal_z, optimal_z_test = {}, {}
    opt_criteriawise_v, opt_criteriawise_v_test = {},{} 
    x, x_test = {},{}
    #if noise:
     #   noisy_z = {}
    #Cs = []
    for n in range(N+N_test):
        x_n = [x_total[k][n,:] for k in range(K)]
        #C= aux.function_C(x_n, w_orig, theta_orig)
        #Cs.append(C[0,:])
        #C[0,:].argsort()
        #len(set([str(tuple(c.argsort())) for c in Cs]))
        z_n, opt_criteriawise, _ = aux.forward_problem(x_n,w_orig,theta_orig,
                                                       list_A[n],b,gamma,
                                                       d_z,K)
        #Noise de los papers de Chaosheng Dong
        #if noise:
         #   z_r_noise = z_r + [random.uniform(-0.25,0.25) for _ in range(d)]
          #  noisy_z[r] = z_r_noise
        if n<N: #NO HAGO PRINT PARA EL CONJUNTO TEST AUNQUE LOS GENERO TODOS A LA VEZ
            optimal_z[n] = z_n
            opt_criteriawise_v[n] = opt_criteriawise
            x[n] = x_n
            print("\n\nFor context x_n=\n",x_n,"\n optimal solution is z_r=",np.round(z_n,2))
            #if noise:
             #   print("\n\nand noisy solution is z_r_noise=",np.round(z_r_noise,2))
            if opt_criteriawise is not None:
                for k in range(K):
                    print(f"and optimal criteriawise v_{k}:",opt_criteriawise[k])
        else:
            optimal_z_test[n] = z_n
            opt_criteriawise_v_test[n] = opt_criteriawise
            x_test[n] = x_n

    #list(set(tuple(sublist) for sublist in optimal_z.values()))

    dict_instances = {'variables': (data_final, stocks, market_features),
                      'params': (d_x,d_x_k,d_z,m,K),
                      '(w,theta)_orig': (w_orig,theta_orig),
                      '(A,b)': (list_A,b),
                      'x': (x,x_test),
                      'optimal_z': (optimal_z,optimal_z_test),
                      'opt_criteriawise': (opt_criteriawise_v,opt_criteriawise_v_test)
                      }
    #if noise:
     #   dict_instances['noisy_z'] = noisy_z
    return dict_instances


 