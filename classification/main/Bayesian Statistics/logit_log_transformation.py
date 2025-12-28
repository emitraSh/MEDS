import numpy as np
from scipy.stats import beta,gamma
import matplotlib.pyplot as plt
from scipy.special import beta as beta_func
from scipy.special import gamma as gamma_func


# initial random variable "theta" is theta ~ Beta(a,b)

a = 0.5
b = 0.5


theta = np.linspace(0.0001, 0.9999, 300)
beta_pdf= beta.pdf(theta, a, b)

# applying logit transformation : z = log(theta/(1-theta))

z = np.linspace(-6, 6, 200)
ez = np.exp(z)
theta_transformed = ez / (1 + ez)
logit_beta_pdf =(1 / beta_func(a, b)) * theta_transformed**a * (1 - theta_transformed)**b


fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Beta distribution
axes[0].plot(theta, beta_pdf)
axes[0].set_title("Beta PDF (θ space)")
axes[0].set_xlabel("θ")
axes[0].set_ylabel("density")

# Right: Logit-Beta transformed distribution
axes[1].plot(z, logit_beta_pdf)
axes[1].set_title("Logit-Beta PDF (z space)")
axes[1].set_xlabel("z")

plt.figtext(0.5, -0.05,
            "Figure 1. The Beta(0.5,0.5) is an extreme probability concentrating on 1 and 0, and transformation makes another visual representation of the same distribution by showcasing uncertainty. "
            "theta = 0.01 -> z = -4.6 -> close to 0"
            "theta = 0.5 -> z = 0 -> middle"
            "theta = 0.99 -> z = 4.6 -> close to 1",

            wrap=True, ha='center', fontsize=10)

plt.tight_layout()
plt.show()


a, b = 1, 1  # shape=a, rate=b

theta = np.linspace(0.0001, 10, 300)
gamma_pdf = (b**a / gamma_func(a)) * theta**(a - 1) * np.exp(-b * theta)


z = np.linspace(-6, 4, 300)
theta_from_z = np.exp(z)              # θ = e^z
dz_theta = theta_from_z               # derivative dθ/dz = e^z

log_gamma_pdf = (b**a / gamma_func(a)) * theta_from_z**(a - 1) * np.exp(-b * theta_from_z) * dz_theta


fig, axes = plt.subplots(1, 2, figsize=(12,4))

axes[0].plot(theta, gamma_pdf)
axes[0].set_title("Gamma PDF (θ space)")
axes[0].set_xlabel("θ")
axes[0].set_ylabel("density")
axes[0].text(0.5, -0.25,
             "Skewed right: long tail toward large θ",
             transform=axes[0].transAxes,
             ha='center', fontsize=9)

axes[1].plot(z, log_gamma_pdf)
axes[1].set_title("Log-Gamma PDF (z space)")
axes[1].set_xlabel("z")
axes[1].text(0.5, -0.25,
             "Log transformation spreads small θ and compresses large θ",
             transform=axes[1].transAxes,
             ha='center', fontsize=9)

plt.tight_layout()
plt.figtext(0.5, -0.05,
            "Comparison: Gamma(1,1) is on θ>0, while log(θ) creates an unconstrained z variable.",
            ha='center', fontsize=10)

plt.show()