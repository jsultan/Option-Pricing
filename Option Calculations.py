from VanillaOptions import BSM_Option, get_year_deltas, GBM, graph_payoff, graph_payoff_nova
import datetime as dt
import pandas as pd
%matplotlib inline

S0 = 1.43
strike = 1.55
t = get_year_deltas([dt.datetime(2018,7,31), dt.datetime(2018,11,30)])[1]    
r = 0.026701
sigma = 0.17131500
flavor = 'Call'

call = BSM_Option(S0, strike, t, r, sigma , flavor)  
call.value()
call.delta()
call.vega()
call.theta()
call.gamma()
call.rho()
call.imp_vol(.174, sigma_est=0.15)


strike = [50, 60, 65, 75]
premium = [1,0,0,1]
lots = [-100, 100, 100, -100]
flavor = ['Call','Call', 'Call', 'Call']
lotsize = 1

graph_payoff(strike, premium ,flavor, lots , lotsize)




option = pd.read_csv('C:\Transfer\Options.csv')
%matplotlib inline
graph_payoff_nova(option)
