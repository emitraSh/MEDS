import pandas as pd
import plotly.express as px

df = pd.read_csv('breast-cancer.csv')

print(df.head())
print(df.describe())

hist_diagnosis =px.histogram(data_frame=df, x='diagnosis', color='diagnosis', color_discrete_sequence=px.colors.sequential.Reds)
hist_diagnosis.show()

hist_size_diagnosis = px.histogram(df,x='area_mean', color='diagnosis', color_discrete_sequence=px.colors.sequential.Reds)
hist_size_diagnosis.show()

df.drop('id', axis=1, inplace=True)

df['diagnosis'] = (df['diagnosis'] == 'M').astype(int) # M->1 , B ->0

corr = df.corr() # creates correlation matrix - default is pearson correlation
plt.figure(figsize = (20,20)) 


