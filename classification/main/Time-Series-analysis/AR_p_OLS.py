import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def ARpOLS(y, p, CI = 0.95):

    y = np.array(y)
    T = len(y)

    #step 1 : constructing p lag matrix

    # y_0= y_-1 = ... = y_-p : this tells us that data before y_1 is not important so first row of design matrix with:
    #  y_p+1 = (y_p.a_0 + y_p-1.a_1 + ... +  y_1.a_p)   => first func AR(p)
    #                                 ...
    #  y_T = (y_T-1.a_0 + y_T-2.a_1 + ... +  y_T-p.a_p)   => last func AR(p) so:

    Y_AR = y[p:]
    X = np.column_stack([y[p - i - 1: T - i - 1] for i in range(p)])
    XtX_inv = np.linalg.inv(X.T @ X)
    print(XtX_inv.shape ,X.T.shape , Y_AR.shape)
    a_hat = XtX_inv @ X.T @ Y_AR
    print(f"{a_hat.shape}___")


    # White noise :
    residuals = Y_AR - X@a_hat
    wn_variance =(residuals@residuals)/T-p

    # Variance-covariance matrix
    cov_a_hat = wn_variance * XtX_inv

    std_cov = np.sqrt(np.diag(cov_a_hat))

    # Compute z-score from confidence level
    z = norm.ppf((1 + CI) / 2)

    # Confidence intervals
    ci_bounds = np.column_stack([
        a_hat - z * std_cov,
        a_hat + z * std_cov
    ])
    print(f"__ OLS estimation ___"
          f"a_hat: {a_hat},"
          f"white noise sigma^2: {wn_variance},"
          f"std_cov = {std_cov},"
          f"ci_bounds: {ci_bounds}")

    return a_hat


def ARp_forecast(y, a_hat, p,h):
    y = list(y)
    forecast= []

    for _ in range(h):
        lags = y[-p:] # we want y_T,... y_T-p+1
        y_hat = np.dot(a_hat, lags[::-1]) #sequence[start : stop : step] reverses the order
        forecast.append(y_hat)
        y.append(y_hat)

    return np.array(y) , np.array(forecast)


y = np.array([
    0.2, -0.5, 0.1, 0.8, -0.3, 0.6, -0.2, 0.4,
    -0.1, 0.9, -0.4, 0.3, 0.2, -0.6, 0.5
])
h = 8
p = 2

y_train = y[:-h]
y_test = y[-h:]
a_hat_ols = ARpOLS(y_train, p)
y_f , forecast_ols = ARp_forecast(y_train, a_hat_ols, p, h)

table = np.column_stack([
    y_test,
    forecast_ols,
    y_test - forecast_ols
])
print(" real_y     forecast     error")
for row in table:
    print(f"{row[0]:8.3f}  {row[1]:10.3f}  {row[2]:10.3f}")


t = np.arange(len(y))
t_forecast = np.arange(len(y_train), len(y))

plt.figure(figsize=(10, 5))

plt.plot(t, y, label="Observed data", linewidth=2)
plt.plot(t_forecast, forecast_ols, "o-", label="OLS AR forecast")

plt.axvline(len(y_train)-1, linestyle="--", color="black", label="Forecast origin")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("8-step Forecast: OLS AR")
plt.legend()
plt.grid(True)
plt.show()

