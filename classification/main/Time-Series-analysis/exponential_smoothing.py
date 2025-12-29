import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#  part b
def g_exponential_smoothing(A, M_B , f_0 , y , theta_hat_0, k):
    y = np.asarray(y)
    T = len(y)
    m = f_0.shape[0] # dimension

    M_B_inv = np.linalg.inv(M_B) #inverse of matrix M_B

    f_N_lst = []
    f_1 = A @ f_0

    # K step forecast based on A :
    for _ in range(1, k+1):
        v = f_1
        f_N_lst.append(v.copy()) # copy() because we want the stored object to be independent of initial
        v = A @ v


    thetas =[theta_hat_0.copy()]
    forecasts ={} # this captures every array of y_hats with different thetas that obtained every time

    thetas_t = thetas[0]
    y_hat = np.array([f_N_lst[N-1].T @ thetas_t for N in range(1, k+1)])  #given in the question y_T+N,T = f(N)' theta_T
    forecasts[0] = y_hat

    # recursion
    for t in range(1, T+1):
        theta_best = thetas[-1]
        y_hat_theta_best = f_1.T @ theta_best
        pred_error = y[t-1] - y_hat_theta_best
        new_theta_best = A.T @ theta_best + M_B_inv @ (f_0 * pred_error)
        thetas.append(new_theta_best)

        #compute new y_hats based on the last most optimal theta
        y_hat = np.array([f_N_lst[N-1].T @ thetas_t for N in range(1, k+1)])
        forecasts[t] = y_hat

    return thetas, forecasts



"""computing lim(F' omega.T F)^-1 = f0 f0' + beta A^{-1} M A'^{-1}"""

def M_computation(A, f_0, beta):
    m = A.shape[0]
    Ainv = np.linalg.inv(A)

    # Kronecker product A_inv ⊗ A_inv
    K = np.kron(Ainv, Ainv)
    I = np.eye(m * m)

    # vec(f0 f0')
    FF = np.outer(f_0, f_0)
    vec_FF = FF.reshape(m * m)

    # Solve for vec(M)
    vec_M = np.linalg.solve(I - beta * K, vec_FF)

    M_B = vec_M.reshape((m, m))
    return M_B




# part c

##############################  fetching data ################################
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
##############################################################

df["log_GDP"] = np.log(df["GDP"])
y = df["log_GDP"].to_numpy()
# description in the uploaded notes:

A = np.array([[1.0,0.0,0.0,0.0,0.0],
             [1.0,1.0,0.0,0.0,0.0],
             [1.0,0.0,0.0,1.0,0.0],
             [1.0,0.0,-1.0,0.0,0.0],
             [1.0,0.0,0.0,0.0,-1.0]])

f_0 = np.array([1.0, 0.0, 0.0, 1.0, 1.0])

m = len(f_0)

theta_hat_0 = np.zeros(m)

betas = [0.9 , 0.8 , 0.5 , 0.3]

k= 8

mean_sqr_forecast_err= {}

for beta in betas:
    M_B = M_computation(A, f_0, beta)
    #print(f"+_+_+_+_+_{M_B}")
    thetas, forecasts = g_exponential_smoothing(A, M_B , f_0 , y , theta_hat_0, k=8)
    print(f'for {beta}  -- >  {thetas[-1]}')
    T = len(y)
    msfe_t = []

    for t_index in range(0, T - k):
        t = t_index + 1  # just for interpretation

        # forecasts from origin t (after seeing y_1..y_t):
        # our recursion stored θ̃_t at index t in `thetas` and forecasts[t]
        yhat_future = forecasts[t_index + 1]  # shape (k,) = [ŷ_{t+1|t},...,ŷ_{t+8|t}]

        # actual future values y_{t+1},...,y_{t+8}
        y_future = y[t_index + 1: t_index + 1 + k]  # 8 values

        errors = y_future - yhat_future
        msfe = np.mean(errors ** 2)
        msfe_t.append(msfe)

    mean_sqr_forecast_err[beta] = np.array(msfe_t)


for beta in betas:
    avg_msfe = mean_sqr_forecast_err[beta].mean()
    print(f"β={beta:.2f}, avg MSFE={avg_msfe:.6f}")

# ---------- Visualization ----------
betas = list(mean_sqr_forecast_err.keys())
msfe_matrix = np.column_stack([mean_sqr_forecast_err[b] for b in betas])

time = np.arange(msfe_matrix.shape[0]) + 1  # 1..T-k

plt.figure(figsize=(9,5))

# plot ALL series with a single command
plt.plot(time, msfe_matrix)

# optional: add labels using legend
plt.legend([f"β={b}" for b in betas])

plt.xlabel("Forecast origin t")
plt.ylabel("8-step MSFE")
plt.title("8-step MSFE for different β values")
plt.grid(True)
plt.tight_layout()
plt.show()





