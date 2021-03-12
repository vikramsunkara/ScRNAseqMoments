import pdb
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


pdf= PdfPages("Figures/MI_sensitivity"+".pdf")

data0 = pd.read_csv("MI_10000traj_shift30.csv")
data0.head()
No_Up = data0["No_Up"]

sigma_1 = 0.01875
sigma_1_list = sigma_1*np.array([1/2, 2, 2**2, 2**3, 2**4])
    
mes   = ["halved"]+["times %d"%2**i for i in (1, 2, 3, 4)]  
    
for i in range(len(sigma_1_list)):
    data1 = pd.read_csv("MI_10000traj_shift30_%d.csv"%i)
    
    df = {}
    df["No_Up"] = No_Up
    df["Double_Up"] = data1["Double_Up"]
    df["Double_Up_1chng"] = data1["Double_Up_1chng"]
    df["Single_Up"] = data1["Single_Up"]
    
    data = pd.DataFrame(df)
    
    Double_Up = data["Double_Up"]
    Double_Up_1chng = data["Double_Up_1chng"]
    Single_Up = data["Single_Up"]
    
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    
    models = [No_Up, Single_Up, Double_Up, Double_Up_1chng]
    ax.boxplot(models,showfliers=False) # showfliers = False remove outliers
    plt.xticks([1, 2, 3, 4], ['No-I', 'Mono-I', 'Bi-I', 'Bi-I-asym'])
    xmin, xmax = ax.get_xlim()
    plt.plot(np.linspace(xmin, xmax, 100), np.percentile(No_Up, 95)*np.ones(100), "--", color = "black", linewidth = 2)  
    plt.text(2.25, 1.1*np.percentile(No_Up, 95), "95th percentile (No-I)", fontsize = 15)
    plt.ylabel("MI scores", fontsize = 20)
    
    # One way anova https://reneshbedre.github.io/blog/anova.html
    import scipy.stats as stats
    # stats f_oneway functions takes the groups as input and returns F and P-value
    fvalue, pvalue = stats.f_oneway(data0['No_Up'], data['Single_Up'], data['Double_Up'], data['Double_Up_1chng'])
    print("One way anova")
    print("Fvalue =", fvalue, "pvalue", pvalue)
    print("######################################")
    #### get an R like output
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    # reshape the d dataframe suitable for statsmodels package 
    d_melt = pd.melt(data.reset_index(), id_vars=['index'], value_vars=["No_Up", 'Single_Up', 'Double_Up', 'Double_Up_1chng'])
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
    
    plt.title("Regulation: "+mes[i], fontsize = 16)
    plt.tight_layout()
    pdf.savefig(fig)
    
    
pdf.close()

