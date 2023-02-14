import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import warnings

class Import:

    def __init__(self, file, columns, n):

        self.file = file
        self.columns = columns
        self.n = n

    def import_df(self):      
        """
        import_data Import excel file to data framme

        Args:
            filename (str): Name of the excel file
            names (dict): dictionary of current column names and replaced column names (eg. {oldname:newname})

        Returns:
            Dataframe: Dataframe of the cleaned excel file
        """        

        df = pd.read_excel(self.file, header=1)
        df = df.dropna(axis=1)
        df = df.loc[:,~df.columns.str.contains("^metingnr")]
        df = df.rename(columns=self.columns)
        names = df.columns.values.tolist()

        # df = df.T.tail(-1)
        df.columns = names

        df_melt = pd.melt(df.reset_index(), id_vars="index", value_vars=df.columns)
        df_melt.columns = ["index", "treatment", "value"]
        
        mean = []
        std = []
        for col in df.columns:
            means = df[col].values.mean()
            stds = df[col].values.std()
            mean.append(means)
            std.append(stds)
        means

        df_melt["means"] = np.repeat(mean,self.n)

        return df, df_melt, names, mean, std


class Anova:

    def __init__(self, df_melt, mean, alpha, n, a):

        self.df_melt = df_melt
        self.alpha = alpha
        self.mean = mean
        self.n = n
        self.a = a

    def anova(self):
        """
        anova: Do anova analysis to check if there is a significant difference between treatments
               Prints if H0 gets accepted or rejected

        Args:
            df_melt (dataframe): newly formated dataframe
            alpha (_type_): significance

        Returns:
            object: object containing:
                    - df_melt (dataframe): new df_melt containing extra column of residuals
                    - model (function): ols model of the values against treatments
                    - anova_table(dataframe): A anova table containing sum_sq, df, F and pvalue
        """        
        
        model = ols("value~treatment", data=self.df_melt).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        pvalue = anova_table["PR(>F)"]["treatment"]
        if pvalue < self.alpha:
            print("H0 rejected")
        else:
            print("H0 accepted")

        self.df_melt["residual"] = model.resid  

        sse = anova_table["sum_sq"]["Residual"]
        dfe = anova_table["df"]["Residual"]
        mse = sse/(self.a*(self.n-1))
        t_crit = stats.t.ppf(q=1-self.alpha/2, df=dfe)

        lsd = t_crit * np.sqrt(2*mse/self.n)

        i = 0

        while i < len(self.mean)-1:
            for j in range(i+1, len(self.mean)):
                diff= np.abs(self.mean[i]-self.mean[j])
                if diff > lsd:
                    print("%i-%i: SIGNIFICANT DIFFERENCE!" %((i+1),(j+1)))
                else:
                    print("%i-%i: no significant difference " %(i+1,j+1))
            i+=1

        return self.df_melt, anova_table, pvalue, lsd



    def boxplot(self):
        """
        boxplot Creates boxplot of the values in dataframe

        Args:
            df_melt (dataframe): newly formated dataframe
        """        
        plt.figure()
        sns.boxplot(data=self.df_melt, x="treatment", y="value")
        plt.show()

    def norm_plot(self):
        """
        norm_plot Creates a probability plot of residuals

        Args:
            df_melt (df): newly formated dataframe
        """    
        warnings.filterwarnings('ignore')    
        residuals = self.df_melt['residual'].values.tolist()
        residuals.sort()
        residuals = np.round(residuals,2)
        residuals
        ind = [1]

        for i in range(1, len(residuals)):
            if residuals[i] == residuals[i-1]:
                ind.append(ind[i-1])
            else:
                ind.append(i+1)

        ind
        chance = [(i-0.5)/len(residuals) for i in ind]
        z = stats.zscore(chance)
        residuals = np.transpose(residuals)
        residuals = residuals[:,np.newaxis]
        az = np.linalg.lstsq(residuals, z)

        fig, ax1 = plt.subplots(1,1)
        ax1.plot(residuals, az[0]*residuals, 'r')
        ax1.scatter(residuals, z)
        ax1.set_ylabel('z-score')
        ax1.set_xlabel('Residual')
        plt.show()



    def res_plot(self):
        """
        res_plot Creates residualplots

        Args:
            df_melt (dataframe): newly formated dataframe
        """        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,8))
        ax1.stem(self.df_melt["treatment"], self.df_melt["residual"], basefmt='k')
        ax2.stem(self.df_melt["means"], self.df_melt["residual"], basefmt='k')
        ax1.set_xlabel("treatment")
        ax1.set_ylabel("residual")
        ax2.set_xlabel(r"$\bar{y}_i$")
        ax2.set_ylabel(r"residual")

        plt.show()
