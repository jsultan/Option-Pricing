from VanillaOptions import BSM_Option, get_year_deltas, GBM, graph_payoff, graph_payoff_nova, get_year_deltas
import datetime as dt
import pandas as pd
%matplotlib inline

S0 = .44
strike = .25
t = get_year_deltas([dt.datetime(2018,8,2), dt.datetime(2018,12,18)])[1]    
r = 0.026815
sigma = 07
flavor = 'Put'

futures_adjustment = np.exp(-r*t)

call = BSM_Option(S0, strike, t, r, sigma , flavor)  
call.value()
call.delta()
call.vega()
call.theta()
call.gamma()
call.rho()
call.imp_vol(C0 = .174)

payoff = []
for i in np.linspace(sigma * .9, sigma *1.1, 100):
    payoff.append(BSM_Option(S0, strike, t, r, i , flavor).value() )

paths = GBM(S0, r, sigma, t, )
CallPayoffAverage = np.average(np.maximum(0, paths[-1] - K))


call = BSM_Option(S0, strike, t, r, sigma , flavor)  


strike = [50, 60, 65, 75]
premium = [1,0,0,1]
lots = [-100, 100, 100, -100]
flavor = ['Call','Call', 'Call', 'Call']
lotsize = 1

graph_payoff(strike, premium ,flavor, lots , lotsize)




option = pd.read_csv('C:\Transfer\Options.csv')
%matplotlib inline
graph_payoff_nova(option)



import QuantLib as ql
import math

calendar = ql.UnitedStates()
bussiness_convention = ql.ModifiedFollowing
settlement_days = 0
day_count = ql.ActualActual()

interest_rate = 0.026815
calc_date = ql.Date(2,8,2018)
yield_curve = ql.FlatForward(calc_date, 
                             interest_rate,
                             day_count,
                             ql.Compounded,
                             ql.Continuous)

ql.Settings.instance().evaluationDate = calc_date
option_maturity_date = ql.Date(18,12,2018)
strike = .25
spot = 0.44
volatility = 4 
flavor = ql.Option.Put

discount = yield_curve.discount(option_maturity_date)
strikepayoff = ql.PlainVanillaPayoff(flavor, strike)
T = yield_curve.dayCounter().yearFraction(calc_date, 
                                          option_maturity_date)
stddev = volatility*math.sqrt(T)

black = ql.BlackCalculator(strikepayoff, 
                           spot, 
                           stddev, 
                           discount)

print( "%-20s: %4.4f" %("Option Price", black.value() ))
print( "%-20s: %4.4f" %("Delta", black.delta(spot) ))
print( "%-20s: %4.4f" %("Gamma", black.gamma(spot) ))
print( "%-20s: %4.4f" %("Theta", black.theta(spot, T) ))
print( "%-20s: %4.4f" %("Vega", black.vega(T) ))
print( "%-20s: %4.4f" %("Rho", black.rho( T) ))


dat = pd.read_clipboard()
dat['Date'] = pd.to_datetime(dat['Date'])
dat.set_index(['Date'], inplace = True)

ret = np.log(dat/dat.shift(-1))
ret.dropna(inplace = True)

np.sqrt(np.dot(np.dot([1,1],ret.rolling(window = 20).cov()), [1, 1]))








