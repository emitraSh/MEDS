from scipy.stats import norm,t
import matplotlib.pyplot as plt
import numpy as np

prior_male = 0.45
prior_female = 0.4
prior_nonB = 0.15


x =  np.linspace(6, 16, 400)
likelihood_speed_male = norm.pdf(x,10.5, 1.2)
likelihood_speed_female = norm.pdf(x,9.8, 1.0)

#for shifted t distribution, we need to define(degree of freedom) and scale to define the density df= 1_3 : very heavy tailed, 5_10: heavy tailed
df = 5
scale = np.sqrt(((df-2)/df)*3)
likelihood_speed_nonB = t.pdf((x - 10.1) / scale, df) / scale

#non normalized posterior
posterior_male = likelihood_speed_male * prior_male
posterior_female = likelihood_speed_female * prior_female
posterior_nonB = likelihood_speed_nonB * prior_nonB

plt.figure(figsize=(10,5))
plt.plot(x, likelihood_speed_male, label = 'Speed | male' , color = 'blue')
plt.plot(x, likelihood_speed_female, label = 'Speed | female', color = 'red')
plt.plot(x, likelihood_speed_nonB, label = 'Speed | nonB', color = 'gray')
plt.xlabel('Speed')
plt.ylabel('Density')
plt.title('SPEED | GENDER')
plt.legend()
plt.show()


plt.figure(figsize=(10,5))
plt.plot(x, posterior_male, label = 'Male|Speed', color = 'blue')
plt.plot(x, posterior_female, label = 'Female|Speed', color = 'red')
plt.plot(x, posterior_nonB, label = 'NonB|Speed', color = 'gray')
plt.xlabel('Gender')
plt.ylabel('Density')
plt.title('GENDER | SPEED')
plt.legend()
plt.show()

#normalized posterior
normalization = posterior_male+posterior_female+posterior_nonB
posterior_male_n = likelihood_speed_male * prior_male(normalization)
posterior_female_n = likelihood_speed_female * prior_female(normalization)
posterior_nonB_n = likelihood_speed_nonB * prior_nonB(normalization)
plt.figure(figsize=(10,5))
plt.plot(x, posterior_male_n, label = 'Male|Speed', color = 'blue')
plt.plot(x, posterior_female_n, label = 'Female|Speed', color = 'red')
plt.plot(x, posterior_nonB_n, label = 'NonB|Speed', color = 'gray')
plt.xlabel('Gender')
plt.ylabel('Density')
plt.title('normalized(GENDER | SPEED)')
plt.legend()
plt.show()




#__________personal check_________________
general_posterior_male = posterior_male *posterior_female *posterior_nonB * prior_male
general_posterior_female = posterior_male *posterior_female *posterior_nonB * prior_female
general_posterior_nonB = posterior_male *posterior_female *posterior_nonB * prior_nonB
plt.figure(figsize=(10,5))
plt.plot(x, general_posterior_male, label = 'gender|Speed', color = 'black')
plt.plot(x, general_posterior_female, label = 'gender|Speed', color = 'black')
plt.plot(x, general_posterior_nonB, label = 'gender|Speed', color = 'black')
plt.show()