import numpy as np
import pandas as pd

def build_H(T, m=4, method ='cosine',p=0):
    if method == 'dummy':
        H = np.zeros((T,m-1)) # m-1 dummies as we consider last part of division as baseline
        for t in range(T):
            j = ((t+1)- int(t/m)*m)
            if j<m:
                H[t,j-1] = 1
    else:
        H = np.column_stack([
        np.sin(2*np.pi*np.arange(1, T+1)/p),
        np.cos(2*np.pi*np.arange(1, T+1)/p)
         ])

    return H


def calc_coef(y,G,H):

    y = np.asarray(y).reshape(-1,1) #here -1 demonstrates as many rows as needed which in our case is "T" to create a matrix [T,1]
    G_H = np.hstack([G,H]) # hstack stands for horizontal stack joins the two matrix side by side

    """ Ready to use code for ols
    coef, _, _, _ = np.linalg.lstsq(G_H, y, rcond=None)
    coef = coef.ravel()  #makes it easier to slice 
    """
    #####           from scratch :  θ =([G,H]′[G,H])−1 [G,H]′x  :            ####
    G_HtG_H_inv = np.linalg.inv(G_H.T @ G_H)
    theta_hat = G_HtG_H_inv @ (G_H.T @ y)

    y_hat = G_H @ theta_hat
    u_hat = y - y_hat

    k = G.shape[1]  # number of trend regressors
    beta_hat = theta_hat[:k]  # trend part
    gamma_hat = theta_hat[k:]

    d_hat = G @ beta_hat
    s_hat = H @ gamma_hat
    y_hat = d_hat + s_hat
    u_hat = y.ravel() - y_hat

    return beta_hat, gamma_hat, d_hat, s_hat, y_hat, u_hat



#fetching data ##################################################################################
import requests
from io import StringIO
import matplotlib.pyplot as plt

# --- Download quarterly GDP from ECB ---------------------------------
base_url = "https://data-api.ecb.europa.eu/service/data"
data_flow = "MNA"
data_id = "Q.N.I9.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.N"

url = f"{base_url}/{data_flow}/{data_id}"
params = {
    "format": "csvdata",
    "startPeriod": "1995-Q1",
    "endPeriod": "2024-Q4",
    "detail": "dataonly"
}

r = requests.get(url, params=params)
r.raise_for_status()
df = pd.read_csv(StringIO(r.text))

# Keep relevant columns and clean
df = df.rename(columns={"TIME_PERIOD": "Date", "OBS_VALUE": "GDP"})
df = df[["Date", "GDP"]]
df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce")
df["Date"] = pd.PeriodIndex(df["Date"], freq="Q").to_timestamp("Q")

df.head()


df["log_GDP"] = np.log(df["GDP"])
T = len(df)
y = df["log_GDP"].values
m=4
H = build_H(T,m)
G = np.column_stack([np.ones(T), np.arange(1, T + 1)]) #we want two column for G one for beta0 the other for t's

beta_hat, gamma_hat, d_hat, s_hat, y_hat, u_hat = calc_coef(y,G,H)

plt.figure(figsize=(10,6))
plt.plot(df["Date"], y, label="log(GDP)", color="gray")
plt.plot(df["Date"], d_hat, label="Estimated trend d̂ₜ", color="blue")
plt.plot(df["Date"], d_hat + s_hat, label="Trend + season d̂ₜ + ŝₜ", color="orange")
plt.legend()
plt.title("OLS Trend + Seasonal Decomposition of log(GDP)___DUMMY")
plt.show()


p = 4
method="cosine"
H_harm = build_H(T,m,method,p)
beta_hat_HRM, gamma_hat_HRM, d_hat_HRM, s_hat_HRM, y_hat_HRM, u_hat_HRM = calc_coef(y, G, H_harm)

plt.figure(figsize=(10,6))
plt.plot(df["Date"], y, label="log(GDP)", color="gray")
plt.plot(df["Date"], d_hat_HRM, label="Estimated trend d̂ₜ ", color="blue")
plt.plot(df["Date"], d_hat_HRM + s_hat_HRM, label="Trend + season d̂ₜ + ŝₜ", color="orange")
plt.legend()
plt.title("OLS Trend + Seasonal Decomposition of log(GDP)___ Fourier-harmonic")
plt.show()



##################### part c

k = 4  # forecast horizon (4 quarters ahead)

def k_step_naive_forecasting(T,k,beta,gamma):

    t_future = np.arange(T + 1, T + k + 1)  # T+1 ... T+k
    G_future = np.column_stack([np.ones(k), t_future])

    # same seasonal pattern
    m = 4
    method="cosine"
    p=4
    H_future = build_H(k,m,method,p)

    y_pred_future = (G_future @ beta) + (H_future @ gamma)

    GDP_pred_future = np.exp(y_pred_future)

    last_date = df["Date"].iloc[-1]

    # create new quarterly dates
    future_dates = pd.period_range(start=last_date.to_period("Q") + 1, periods=k, freq="Q").to_timestamp("Q")

    df_forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted_log_GDP": y_pred_future.ravel(),
        "Predicted_GDP": GDP_pred_future.ravel()
    })

    print(df_forecast)

    plt.figure(figsize=(10, 6))
    plt.plot(df_forecast["Date"], df_forecast["Predicted_log_GDP"], label="Forecast (k steps ahead)", color="red",
             marker="o")
    plt.legend()
    plt.title("OLS-based Forecasts of log(GDP)")
    plt.show()

k =8
k_step_naive_forecasting(T,k,beta_hat_HRM,gamma_hat_HRM)