from VanillaOptions import BSM_Option, get_year_deltas, GBM, graph_payoff, graph_payoff_nova
import datetime as dt
import pandas as pd
%matplotlib inline

S0 = 4.11
strike = 4.3
t = get_year_deltas([dt.datetime(2018,5,9), dt.datetime(2018,8,24)])[1]    
r = 0.025733
sigma = 0.243957
flavor = 'c'

I = 500
M = 150

call = BSM_Option(S0, strike, t, r, sigma , flavor)   
call.value()
call.delta()
call.vega()
call.theta()
call.gamma()
call.rho()



strike = []
premium = []
lots = []
lotsize = 1000
graph_payoff(strike, premium ,flavor, lots , lotsize)




option = pd.read_csv('C:\Transfer\Options.csv')

cols = ['Principal', 'Unrealized', 'Market', 'Gamma', 'Vega', 'Theta', 'Imp. Volatility']

for col in cols:
    if col == 'Imp. Volatility':
        data[col] = data[col].str.replace('%','').astype(float)
    else:
        data[col] = data[col].str.replace(',','').astype(float)
    
agg_fun = ['sum', 'sum', 'sum', 'sum', 'sum', 'sum', 'mean']
agg_fun = dict(zip(cols, agg_fun))

temp = option.groupby('Underlying')['Principal', 'Unrealized', 'Market', 'Gamma', 'Vega', 'Theta'].agg(agg_fun).round(4)

graph_payoff_nova(option)


data.groupby(['Underlying', 'Flavor','Strike'])['Qty'].sum()

