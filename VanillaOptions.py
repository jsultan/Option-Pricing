from math import log, sqrt, exp
from scipy import stats
import numpy as np
import datetime as dt

class BSM_Option(object):
    '''
    Attributes
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
    '''
    
    def __init__(self, S0, K, T, r, sigma, flavor):
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.flavor = flavor

    def value(self):
            if self.flavor == 'Call':
                ''' Returns option value. '''
                d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
                d2 = d1 - self.sigma*sqrt(self.T) 
                value = self.S0 * stats.norm.cdf(d1, 0, 1) - self.K * stats.norm.cdf(d2, 0 , 1)*exp(-self.r * self.T)
                return value
            elif self.flavor == 'Put':
                d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
                d2 = d1 - self.sigma*sqrt(self.T) 
                value = -self.S0 * stats.norm.cdf(-d1, 0, 1) + self.K * stats.norm.cdf(-d2, 0 , 1)*exp(-self.r * self.T)
                return value
            else:
                return "Incorrect Flavor"
            
    def delta(self):
        ''' Returns delta of an option'''
        if self.flavor == 'Call':
            d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
            return stats.norm.cdf(d1, 0, 1)
        elif self.flavor == 'Put':
            d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
            return -stats.norm.cdf(-d1, 0, 1)
        else:
            return  "Incorrect Flavor"
    
    def gamma(self):
            d1 = ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
            gamma = stats.norm.pdf(d1)/(self.S0 * self.sigma * sqrt(self.T))
            return gamma
        
        
    def vega(self):
        ''' Returns Vega of option. '''
        d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
        vega = self.S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(self.T)
        return vega/100
    
    
    def theta(self):
        ''' Returns Theta of option. '''
        if self.flavor == 'Call':
            d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))   
            d2 = d1 - self.sigma*sqrt(self.T)
            p1 = - self.S0*stats.norm.pdf(d1, 0 ,1)*self.sigma/(2*sqrt(self.T))
            p2 = - self.r * self.K  * stats.norm.cdf(d2, 0 ,1) * exp(-self.r * self.T)
            theta = p1 + p2
            return theta/365
        elif self.flavor == 'Put':
            d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
            d2 = d1 - self.sigma*sqrt(self.T)
            p1 = - self.S0*stats.norm.pdf(d1, 0 ,1)*self.sigma/(2*sqrt(self.T))
            p2 = self.r * self.K  * stats.norm.cdf(-d2, 0 ,1) * exp(-self.r * self.T)
            theta = p1 + p2
            return theta/365    
        else:
            return "Incorrect Flavor"
    
    def rho(self):
        if self.flavor == 'Call':
            d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
            d2 = d1 - self.sigma*sqrt(self.T)
            rho =  self.K * self.T * exp(-self.r * self.T) * stats.norm.cdf(d2, 0, 1)
            return rho/100
        elif self.flavor == 'Put':
            d1 =  ((log(self.S0/self.K) +(self.r + .5*(self.sigma**2))*self.T))/(self.sigma*sqrt(self.T))
            d2 = d1 - self.sigma*sqrt(self.T)
            rho =  - self.K * self.T * exp(-self.r * self.T) * stats.norm.cdf(-d2, 0, 1)
            return rho/100
        else:
            return "Incorrect Flavor"

    def imp_vol(self, C0, sigma_est=0.2, it=1000):
        ''' Returns implied volatility given option price. '''
        option = BSM_Option(self.S0, self.K, self.T, self.r, sigma_est, self.flavor)
        for i in range(it):
            option.sigma -= (option.value() - C0) / option.vega()
        return option.sigma


def get_year_deltas(date_list, day_count=365.):
    start = date_list[0]
    delta_list = [(date - start).days / day_count for date in date_list]
    return np.array(delta_list)  


def spread_option(s1, s2, k , t, sigma1, sigma2, cor, r):
    a =  s2 + k
    b = s2/a
    sigma = sqrt(sigma1**2 - 2*b*cor*sigma1*sigma2 + (b**2)*(sigma2**2))
    d1 = (log(s1/a) + (.5 * sigma1 ** 2 - b*cor*sigma1*sigma2 + .5*(b**2)*(sigma**2)))*sqrt(t)/sigma
    d2 = (log(s1/a) + (-.5 * sigma1 ** 2 + cor*sigma1*sigma2 + .5*(b**2)*(sigma**2) - b*sigma2**2))*sqrt(t)/sigma
    d3 = (log(s1/a) + (-.5*sigma1**2 + .5*(b**2)*(sigma2**2)))*sqrt(t)/sigma
    value = exp(-r*t)*(s1*stats.norm.cdf(d1, 0, 1) - s2*stats.norm.cdf(d2, 0, 1) - k*stats.norm.cdf(d3, 0 ,1))
    return value
    
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import pyodbc 


def graph_payoff(strike, premium, flavor, lots, lotsize):
    if type(strike) is pd.Series:
        strike = strike.tolist()
    if type(premium) is pd.Series:
        premium = premium.tolist()
    if type(lots) is pd.Series:
        lots = lots.tolist()
    if type(flavor) is pd.Series:
        flavor = flavor.tolist()
    
    margin = .5*min(strike)
    lb = min(strike) - margin
    ub = max(strike) + margin
    x = np.linspace(lb, ub, 500)
    pay_off = np.zeros(len(x))
    
    
    for i in range(len(strike)):
        if (lots[i] > 0) & (flavor[i] == 'Call'):
            pay_off += (np.maximum((x - strike[i]), 0)-premium[i])*np.abs(lots[i])*lotsize
            
        elif (lots[i] < 0) & (flavor[i] == 'Call'):
            pay_off += (premium[i]-np.maximum(x - strike[i] , 0))*np.abs(lots[i])*lotsize
            
        elif (lots[i] > 0) & (flavor[i] == 'Put'):
            pay_off += (np.maximum(strike[i] - x, 0)-premium[i])*np.abs(lots[i])*lotsize
            
        elif (lots[i] < 0) & (flavor[i] == 'Put'):
            pay_off += (premium[i]-np.maximum(strike[i] - x, 0))*np.abs(lots[i])*lotsize
    
    
    
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.plot(x, pay_off)
    plt.grid(color='black', linestyle='-', linewidth=.15)
    plt.axhline(y=0, color='k') 
    plt.title('Payoff Diagram')
    plt.xlabel('Underlying Price')
    plt.ylabel('Payoff')
    fmt = '${x:,.0f}'
    tick = mtick.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick) 
    
    
    
def graph_payoff_nova(data):
    underlyings = data.Underlying.unique().tolist()
    underlyings.sort(key=lambda x: x.lower())
        
    nova = pyodbc.connect('driver={SQL Server};server=vm-hou-sql01;database=nova;trusted_connection=true')
    
    
    data['Imp. Volatility'] = data['Imp. Volatility'].str.rstrip('%').astype('float') / 100.0
    data.Expiry = pd.to_datetime(data.Expiry)
    
    #Seperate each option by underlyings
    for idx, types in enumerate(underlyings):
        
        query = '''
        select prs_symb from price_symbols where prs_description = '{}' '''.format(types)
        fwd = pd.read_sql(query, nova)
        pos = fwd.prs_symb.tolist()[0]
        
        # Get lotsize per lot from db
        query = '''
        select prs_ctr_size from price_symbols where prs_description = '{}' '''.format(types)
        lot = pd.read_sql(query, nova)
        lotsize = lot.prs_ctr_size.tolist()[0]
        
        temp = data.loc[data.Underlying == types, :]
        
        strikes =  temp.Strike.unique().tolist()
        for strike in strikes:
            for flav in ['Call', 'Put']:
                if temp.loc[(temp.Strike == strike) & (temp.Flavor == flav), 'Qty'].sum() == 0:
                    temp = temp.loc[~((temp.Strike == strike) & (temp.Flavor == flav)), :]
        
        if temp.shape[0] == 0:
            continue

         
        strike = temp.Strike
        premium = temp.Premium
        lots = temp.Qty
        flavor = temp.Flavor
        vol = temp['Imp. Volatility']
        date = temp.Expiry
        if type(strike) is pd.Series:
            strike = strike.tolist()
        if type(premium) is pd.Series:
            premium = premium.tolist()
        if type(lots) is pd.Series:
            lots = lots.tolist()
        if type(flavor) is pd.Series:
            flavor = flavor.tolist()
        if type(vol) is pd.Series:
            vol = vol.tolist()
        if type(date) is pd.Series:
            date = date.tolist()
            
        # Set bounds for graph
        if 'CORN' in types:
            margin = .5*min(strike)
            lb = min(strike) - margin
            ub = max(strike) + margin
        elif ('WTI' in types)|('Brent' in types):
            if min(strike) > 50:
                margin = .5*min(strike)
                lb = min(strike) - margin
                ub = max(strike) + margin
            elif min(strike) > 0:
                margin = .5*min(strike)
                lb = min(strike) - margin
                ub = max(strike) + margin
            elif min(strike) < 0:
                margin = .5*min(strike)
                lb = min(strike) - margin
                ub = max(strike) + margin
            elif min(strike) == 0:
                lb = -.5
                ub = .5
        elif 'HH' in types:
            margin = .15*min(strike)
            lb = min(strike) - margin
            ub = max(strike) + margin
        elif 'Ethanol' in types:
            margin = .25*min(strike)
            lb = min(strike) - margin
            ub = max(strike) + margin
        elif 'LCFS' in types:
            lb = min(strike) - .25*min(strike)
            ub = max(strike) + .25*max(strike)
        elif 'Soybean' in types:
            margin = .25*min(strike)
            lb = min(strike) - margin
            ub = max(strike) + margin
        else:
            print('Missing Bounds')
            pass
        
        x = np.linspace(lb, ub, 1000)
        r = .0266
        
        #Calculate option payoff
        pay_off = np.zeros(len(x))
        pay_off2 = np.zeros(len(x))
        for i in range(len(strike)):
            if (lots[i] > 0) & (flavor[i] == 'Call'):
                pay_off += (np.maximum((x - strike[i]), 0)-premium[i])*np.abs(lots[i])*lotsize
                try:
                    pay_off2+= np.asarray([(BSM_Option(s, strike[i], get_year_deltas([dt.datetime.now(), \
                                           date[i]])[1] , r, vol[i], 'Call').value()-premium[i])*np.abs(lots[i])*lotsize for s in x])
                except:
                    pass
            elif (lots[i] < 0) & (flavor[i] == 'Call'):
                pay_off += (premium[i]-np.maximum(x - strike[i] , 0))*np.abs(lots[i])*lotsize
                try:
                    pay_off2+= np.asarray([(premium[i]-BSM_Option(s, strike[i], get_year_deltas([dt.datetime.now(), \
                           date[i]])[1] , r, vol[i], 'Call').value())*np.abs(lots[i])*lotsize for s in x])
                except:
                    pass                
            elif (lots[i] > 0) & (flavor[i] == 'Put'):
                pay_off += (np.maximum(strike[i] - x, 0)-premium[i])*np.abs(lots[i])*lotsize
                try:
                    pay_off2+= np.asarray([(BSM_Option(s, strike[i], get_year_deltas([dt.datetime.now(), \
                                           date[i]])[1] , r, vol[i], 'Put').value()-premium[i])*np.abs(lots[i])*lotsize for s in x])
                except:
                    pass
                
            elif (lots[i] < 0) & (flavor[i] == 'Put'):
                pay_off += (premium[i]-np.maximum(strike[i] - x, 0))*np.abs(lots[i])*lotsize
                try:
                    pay_off2+= np.asarray([(premium[i]-BSM_Option(s, strike[i], get_year_deltas([dt.datetime.now(), \
                           date[i]])[1] , r, vol[i], 'Put').value())*np.abs(lots[i])*lotsize for s in x])     
                except:
                    pass
        
        # Add intersection point of value 
        query = '''
        select top 1 prq_value from price_quotations where
        prs_symb = {} order by prq_date_value desc'''.format("'"+pos+"'")
        
        price = float(pd.read_sql(query, nova).iloc[0,0])
        print("{} : {}".format(types, price))
        
        val = 0
        for i in range(len(strike)):
            if (lots[i] > 0) & (flavor[i] == 'Call'):
                val += (np.maximum((price - strike[i]), 0)-premium[i])*np.abs(lots[i])*lotsize
                
            elif (lots[i] < 0) & (flavor[i] == 'Call'):
                val += (premium[i]-np.maximum(price - strike[i] , 0))*np.abs(lots[i])*lotsize
                
            elif (lots[i] > 0) & (flavor[i] == 'Put'):
                val += (np.maximum(strike[i] - price, 0)-premium[i])*np.abs(lots[i])*lotsize
                
            elif (lots[i] < 0) & (flavor[i] == 'Put'):
                val += (premium[i]-np.maximum(strike[i] - price, 0))*np.abs(lots[i])*lotsize        
        
      
        
        #Create Payoff grapgh
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(figsize = (10, 5))
        plt.plot(x, pay_off, linewidth= 2, label = "Intrinsic Value")
        if 'LCFS' not in types:
            plt.plot(x, pay_off2, linewidth= 2, label = "Option Value")
        plt.grid(color='black', linestyle=':', linewidth=.5)
        plt.axhline(y=0, color='k') 
        plt.title('Payoff Diagram --- {}'.format(types))
        plt.xlabel('Underlying Price')
        plt.ylabel('Payoff')
        fmt = '${x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)    
        plt.axvline(price, color = "orange", linewidth = 2, label = 'Underlying Price : ${:,.3f}'.format(price))
        plt.plot(price, int(val),'ro', markersize  = 10, label= "Current Value : ${:,.0f}".format(int(val)))
        plt.plot([], [], ' ', label="Expiry : {:%m-%d-%Y}".format(date[0]))
        plt.legend(loc = 0, frameon = True)
        


def GBM(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
        sigma * np.sqrt(dt) * rand)
    return paths
    



































    
    
    