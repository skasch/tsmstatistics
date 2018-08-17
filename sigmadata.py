#%%
import numpy as np
from pandas import DataFrame
import plotnine as pn

#%%
n_sellers = np.array([
    1,
    2,
    3,
    4,
    5,
    7,
    10,
    14,
    20,
    28,
    38,
    50,
    65,
    85,
    110,
    140,
    175,
    215,
    260,
    310,
    370,
    440,
    520,
    620,
    740,
    880,
    1040,
    1240,
    1480,
    1760,
    2080,
    2400,
    2800,
    3300,
    3800,
    4400,
    5000,
    5750,
    6500,
    7250,
    8000,
    9000,
    10000,
    12000,
    14000,
    17000,
    20000,
    24000,
    28000,
    33000,
    38000,
    44000,
    50000,
])

delta_sigma = np.array([
    0,
    -0.56419,
    -0.846284,
    -1.02938,
    -1.16296,
    -1.35218,
    -1.53875,
    -1.70338,
    -1.86748,
    -2.01371,
    -2.14009,
    -2.24907,
    -2.34958,
    -2.44894,
    -2.54147,
    -2.62559,
    -2.70148,
    -2.76993,
    -2.83187,
    -2.88819,
    -2.94391,
    -2.99761,
    -3.0486,
    -3.1015,
    -3.15393,
    -3.20454,
    -3.25268,
    -3.30269,
    -3.35233,
    -3.40032,
    -3.44602,
    -3.48474,
    -3.52602,
    -3.56955,
    -3.60655,
    -3.64464,
    -3.67756,
    -3.71325,
    -3.7443,
    -3.77177,
    -3.79637,
    -3.82562,
    -3.85162,
    -3.89622,
    -3.93358,
    -3.98017,
    -4.01879,
    -4.06171,
    -4.09769,
    -4.13571,
    -4.1681,
    -4.20152,
    -4.23046,
])

#%%
n_iterations = 100000
n_pulls = 110
mins = np.empty(n_iterations)
for i in range(n_iterations):
    mins[i] = np.random.normal(size=n_pulls).min()
print(mins.mean())

#%%
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
pow_model = 0.5
model_sellers = np.power(np.log(1+n_sellers), pow_model)
predictions = {}
models = {}
for degree in [1, 2, 3, 4, 5]:
    models[degree] = make_pipeline(PolynomialFeatures(degree),
                                   LinearRegression())
    models[degree].fit(model_sellers.reshape(-1, 1), delta_sigma)
    predictions[degree] = models[degree].predict(model_sellers.reshape(-1, 1))
    print(degree, np.log(mean_squared_error(delta_sigma, predictions[degree])))

#%%
end_slope = ((delta_sigma[-2] - delta_sigma[-1])
             / (np.power(np.log(1+n_sellers[-2]), pow_model) 
                - np.power(np.log(1+n_sellers[-1]), pow_model)))
end_constant = (delta_sigma[-1]
                - end_slope * np.power(np.log(1+n_sellers[-1]), pow_model))
end_tangent = np.power(np.log(1+n_sellers), pow_model) * end_slope + end_constant
print(end_slope, end_tangent[-1])

start_slope = ((delta_sigma[1] - delta_sigma[0])
               / (np.power(np.log(1+n_sellers[1]), pow_model)
                  - np.power(np.log(1+n_sellers[0]), pow_model)))
start_constant = (delta_sigma[0]
                  - start_slope * np.power(np.log(1+n_sellers[0]), pow_model))
start_tangent = (np.power(np.log(1+n_sellers), pow_model) * start_slope
                 + start_constant)
print(start_slope, start_constant)

#%%
coefs = models[4]._final_estimator.coef_
coefs[0] = models[4]._final_estimator.intercept_
test_predictions = (coefs[0]
                    + coefs[1] * model_sellers
                    + coefs[2] * np.power(model_sellers, 2)
                    + coefs[3] * np.power(model_sellers, 3)
                    + coefs[4] * np.power(model_sellers, 4))
coefs

#%%
def n_sigmas(pop_size):
    return (3.29
            - 5.58 * np.power(np.log(1+pop_size), 0.5)
            + 2.444 * np.power(np.log(1+pop_size), 1)
            - 0.653 * np.power(np.log(1+pop_size), 1.5)
            + 0.0652 * np.power(np.log(1+pop_size), 2))
poly_model = np.apply_along_axis(n_sigmas, 0, n_sellers)

#%%
df = DataFrame({'n_sellers': np.log(n_sellers),
                'delta_sigma': delta_sigma, 'end_tangent': end_tangent,
                'start_tangent': start_tangent, 'poly_model': poly_model})
(pn.ggplot(df, pn.aes(x='n_sellers', y='delta_sigma'))
    + pn.geom_line()
    + pn.geom_line(pn.aes(y='end_tangent'), color='blue')
    #+ pn.geom_line(pn.aes(y='start_tangent'), color='orange')
    + pn.geom_line(pn.aes(y='poly_model'), color='green')
    ).draw()
