import numpy as np
import math
import pandas as pd
import itertools
import matplotlib.pyplot as plt


#part a

def holt_winter(n_season=4,y=np.array([]), alpha=0.1 , delta=0.1, gamma=0.1, k=8 ):
    T = len(y)
    j =n_season

    a = [None] * (T + 1)
    b = [None] * (T + 1)
    s = [None] * (T + 1)

    # (11)–(17)
    a[3] = (1 / 8) * y[0] + (1 / 4) * y[1] + (1 / 4) * y[2] + (1 / 4) * y[3] + (1 / 8) * y[4]
    a[4] = (1 / 8) * y[1] + (1 / 4) * y[2] + (1 / 4) * y[3] + (1 / 4) * y[4] + (1 / 8) * y[5]
    b[4] = a[4] - a[3]

    s[4] = y[3] - a[4]
    s[3] = y[2] - a[3]
    s[2] = y[1] - a[3] + b[4]
    s[1] = y[0] - a[3] + 2 * b[4]

    # (7)–(9) recursive formula for t = 5..T -----
    for t in range(5, T + 1):
        a[t] = (1 - alpha) * (y[t - 1] - s[t - j]) + alpha * (a[t - 1] + b[t - 1])
        b[t] = (1 - gamma) * (a[t] - a[t - 1]) + gamma * b[t - 1]
        s[t] = (1 - delta) * (y[t - 1] - a[t]) + delta * s[t - j]

    forecasts = []
    # last seasonal cycle
    cycle = [s[T - 3], s[T - 2], s[T - 1], s[T]]

    for h in range(1, k + 1):
        seasonal = cycle[(h - 1) % j]  # cycles through 0..3
        yhat = a[T] + h * b[T] + seasonal
        forecasts.append(yhat)

    return forecasts


#part b

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
##############################################################

df["log_GDP"] = np.log(df["GDP"])
y = df["log_GDP"].to_numpy()
T = len(y)

alpha_grid = [0.2, 0.5, 0.8]
gamma_grid = [0.2, 0.5, 0.8]
delta_grid = [0.2, 0.5, 0.8]

results = []

for alpha, gamma, delta in itertools.product(alpha_grid, gamma_grid, delta_grid):

    msfes = []

    # forecast origins t = 5,...,T-8
    for t in range(5, T-8):

        y_train = y[:t+1]              # data up to t
        forecasts = holt_winter(y=y_train, alpha=alpha, delta=delta, gamma=gamma, k=8)

        actual = y[t : t+8]
        error = actual - forecasts
        msfe_t = np.mean(error**2)
        msfes.append(msfe_t)

    avg_msfe = np.mean(msfes)

    results.append({
        'alpha': alpha,
        'gamma': gamma,
        'delta': delta,
        'MSFE': avg_msfe
    })

results_df = pd.DataFrame(results)


pivot = results_df[results_df['delta'] == 0.5].pivot(
    index='alpha', columns='gamma', values='MSFE'
)

plt.imshow(pivot, origin='lower', aspect='auto')
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.colorbar(label='MSFE')
plt.title('MSFE for δ = 0.5')
plt.xlabel('gamma')
plt.ylabel('alpha')
plt.show()

