import pdb
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

data = pd.read_csv("MI_10000traj.csv")
data.head()
#pdb.set_trace()

No_Up = data["No_Up"]
Single_Up = data["Single_Up"]
Double_Up = data["Double_Up"]


matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)

models = [No_Up, Single_Up, Double_Up]
ax.boxplot(models,showfliers=False) # showfliers = False remove outliers
plt.xticks([1, 2, 3], ['No-I', 'Mono-I', 'Bi-I'])
xmin, xmax = ax.get_xlim()
plt.plot(np.linspace(xmin, xmax, 100), np.percentile(No_Up, 95)*np.ones(100), "--", color = "black", linewidth = 2)  
plt.text(2.25, 1.1*np.percentile(No_Up, 95), "95th percentile (No-I)", fontsize = 15)
plt.ylabel("MI scores", fontsize = 20)

# One way anova https://reneshbedre.github.io/blog/anova.html
import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns F and P-value
fvalue, pvalue = stats.f_oneway(data['No_Up'], data['Single_Up'], data['Double_Up'])
print("One way anova")
print("Fvalue =", fvalue, "pvalue", pvalue)
print("######################################")
#### get an R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols
# reshape the d dataframe suitable for statsmodels package 
d_melt = pd.melt(data.reset_index(), id_vars=['index'], value_vars=['No_Up', 'Single_Up', 'Double_Up'])
# replace column names
d_melt.columns = ['index', 'treatments', 'value']
# Ordinary Least Squares (OLS) model
model = ols('value ~ C(treatments)', data=d_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

# pairwise comparision significance with HSD https://reneshbedre.github.io/blog/anova.html
from pingouin import pairwise_tukey
# perform multiple pairwise comparison (Tukey HSD)
# for unbalanced (unequal sample size) data, pairwise_tukey uses Tukey-Kramer test
print("Pairwise comparison with Tukey HSD")
m_comp = pairwise_tukey(data=d_melt, dv='value', between='treatments')
print(m_comp)

plt.tight_layout()
plt.savefig("MI.pdf")

