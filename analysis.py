#%%
import math
import numpy as np
from scipy.stats import lognorm
from scipy.special import erfinv
import plotnine as pn
from pandas import DataFrame

#%%
eu_market_value = 40.8547 # = exp(mu + sigma^2/2)
realm_market_value = 45.2019
eu_sale_price = 34.1367 # = exp(mu - sigma^2)
eu_sigma = math.sqrt(2/3*(math.log(eu_market_value) - math.log(eu_sale_price)))
eu_mu = math.log(eu_sale_price) + (eu_sigma ** 2)
realm_mu = math.log(realm_market_value) - (eu_sigma ** 2) / 2

#%%
a = lognorm(eu_sigma)

#%%
xs = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 1., 0.1),
                     np.arange(1., 3, .5)])
ys = np.apply_along_axis(a.pdf, 0, xs)

#%%
df = DataFrame({'x': xs * np.exp(realm_mu), 'y': ys})
(pn.ggplot(df, pn.aes(x='x', y='y')) + pn.geom_line()).draw()


#%%
def log_approx(x, cst=1e10):
    return cst * (x)**(1/cst) - cst
values = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
logs = np.log(values)
taylors = np.apply_along_axis(log_approx, 0, values)
df = DataFrame({'x': values, 'log': logs, 'taylor': taylors})
(pn.ggplot(df)
    + pn.geom_line(pn.aes(x='x', y='log'), color='blue')
    + pn.geom_line(pn.aes(x='x', y='taylor'), color='red')).draw()

#%%
df['log'] - df['taylor']

#%%
def n_sigmas(pop_size):
    return (3.29
            - 5.58 * np.power(log_approx(1+pop_size), 0.5)
            + 2.444 * np.power(log_approx(1+pop_size), 1)
            - 0.653 * np.power(log_approx(1+pop_size), 1.5)
            + 0.0652 * np.power(log_approx(1+pop_size), 2))

#%%
def quantile2(p):
    return np.sqrt(2 * math.pi) * (p-0.5
                                   + math.pi/3 * (p-0.5)**3
                                   + 7*math.pi**2/30 * (p-0.5)**5
                                   + 127*math.pi**3/630 * (p-0.5)**7)

def approx_erfinv(p):
    a = 8*(math.pi-3)/(3*math.pi*(4-math.pi))
    return np.sign(p) * np.sqrt(
        np.sqrt((2/(math.pi*a) + log_approx(1-p**2)/2)**2
                - log_approx(1-p**2)/a)
        - (2/(math.pi*a) + log_approx(1-p**2)/2)
    )

def quantile(p):
    return np.sqrt(2)*approx_erfinv(2*p-1)

values = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5,
                   0.8, 0.9, 0.97, 0.99, 0.997, 0.999])
quantiles = np.apply_along_axis(quantile, 0, values)
quantiles2 = np.apply_along_axis(quantile2, 0, values)
real_qtles = np.sqrt(2)*np.apply_along_axis(erfinv, 0, 2 * values - 1)
df = DataFrame({'x': values, 'y': quantiles, 'y2': quantiles2, 'yr': real_qtles})
(pn.ggplot(df, pn.aes(x='x', y='y'))
    + pn.geom_line(color='blue')
    + pn.geom_line(pn.aes(y='y2'), color='red')
    + pn.geom_line(pn.aes(y='yr'), color='green')
 ).draw()


#%%
eu_min_buyout = 17.0084
eu_market_value = 20.1892
eu_historical = 19.3073
realm_min_buyout = 12
realm_market_value = 30.1279
realm_historical = 24.7006
eu_current_volume = 164
realm_current_volume = 79
eu_n_sale = 377
population_ratio = 0.62
net_volume_ratio = .5
stack_size = 1
realm_n_sale = eu_n_sale*population_ratio
eu_volume = eu_n_sale + eu_current_volume
realm_volume = realm_n_sale + realm_current_volume
eu_net_volume = eu_n_sale + net_volume_ratio*eu_current_volume
# realm_net_volume = realm_n_sale + net_volume_ratio*realm_current_volume
realm_net_volume = eu_net_volume * population_ratio #estimated
if eu_net_volume:
    eu_sale_rate = eu_n_sale / eu_net_volume
else:
    eu_sale_rate = 0
if realm_net_volume:
    realm_sale_rate = realm_n_sale / realm_net_volume
else:
    realm_sale_rate = eu_sale_rate
default_sigma_ratio = 0.2
min_risk_ratio = .5
normal_risk_ratio = .9
max_risk_ratio = 1.4
snipe_risk_ratio = 0.1

#eu_net_volume = eu_current_volume

if realm_market_value:
    realm_safe_market = realm_market_value
elif realm_historical:
    realm_safe_market = realm_historical
elif eu_market_value:
    realm_safe_market = eu_market_value
else:
    realm_safe_market = eu_historical
if eu_market_value:
    eu_safe_market = eu_market_value
else:
    eu_safe_market = eu_historical
realm_weight = (max(1,realm_net_volume))**(-math.log(10)/math.log(5))
realm_my_market = ((0.5*realm_weight) * max(eu_safe_market, realm_safe_market)
                   + (1 - 0.5*realm_weight) * realm_safe_market)

if (realm_net_volume >= 300
    and realm_min_buyout
    and realm_market_value
    and realm_min_buyout < .99 * realm_market_value):
    realm_sigma = -((realm_market_value - realm_min_buyout)
                    / n_sigmas(realm_net_volume/stack_size))
    print('Z-score of Realm min buyout:',
          n_sigmas(realm_net_volume/stack_size))
else:
    eu_default_sigma = default_sigma_ratio * eu_safe_market
    if eu_min_buyout and eu_min_buyout < 0.99 * eu_safe_market:
        eu_market_sigma = (-(eu_safe_market-eu_min_buyout)
                           / n_sigmas(eu_net_volume/stack_size))
        eu_weight = (max(1, eu_net_volume))**(-math.log(10)/math.log(5))
        eu_sigma = eu_weight*eu_default_sigma + (1-eu_weight)*eu_market_sigma
    else:
        eu_sigma = eu_default_sigma
    print('Z-score of EU min buyout:', n_sigmas(eu_net_volume/stack_size))
    print('EU:', eu_safe_market, '+/-', eu_sigma)
    realm_sigma = eu_sigma * realm_my_market / eu_safe_market

print('Realm:', realm_my_market, '+/-', realm_sigma)

min_sale_rate = min(0.97, max(0.05, realm_sale_rate*min_risk_ratio))
normal_sale_rate = min(0.97, max(0.05, realm_sale_rate*normal_risk_ratio))
max_sale_rate = min(0.97, max(0.05, realm_sale_rate*max_risk_ratio))
snipe_sale_rate = min(0.97, max(0.02, realm_sale_rate*snipe_risk_ratio))
print(min_sale_rate, normal_sale_rate, max_sale_rate)
print(snipe_sale_rate)
min_selling_price = realm_my_market + quantile(min_sale_rate)*realm_sigma
normal_selling_price = (realm_my_market
                        + quantile(normal_sale_rate)*realm_sigma)
max_selling_price = realm_my_market + quantile(max_sale_rate)*realm_sigma
snipe_selling_price = realm_my_market + 3*quantile(snipe_sale_rate)*realm_sigma
print(min_selling_price, normal_selling_price, max_selling_price)
print(snipe_selling_price)
