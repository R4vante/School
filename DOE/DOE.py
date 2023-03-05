import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import warnings

class Import:

    def __init__(self, file, sheet_name, n, a, alpha):
        self.file = file
        self.sheet_name = sheet_name
        self.n = n
        self.a = a

    def import_df(self):
        """
        import_df Import excel sheet to dataframe

        Args:
            file (str): file name
            sheet_name (str): sheet name
            n (int): number of replicates

        Returns:
            object: returns object containing
                    - df (dataframe): initial datatframe
                    - df_melt (dataframe): newly formatted data_frame for easy calculations
                    - names (list): list of column names of df
                    - mean (list): list containing means of treatments
        """        
        self.df = pd.read_excel(self.file, sheet_name=self.sheet_name, header = 1)
        self.names = self.df['treatment'][:].values.tolist()
        self.df = self.df.T.tail(-1)
        self.df.columns = self.names

        self.df_melt = pd.melt(self.df.reset_index(), id_vars="index", value_vars=self.df.columns)
        self.df_melt.columns = ["index", "treatment", "value"]
        
        self.mean = []
        for col in df.columns:
            means = df[col].values.mean()
            self.mean.append(means)


        self.df_melt["means"] = np.repeat(self.mean,self.n)

        return self.df, self.df_melt, self.names, self.mean

class Anova_NoBlock:

    def __init__(self, filename, alpha, n, a):

        self.filename = filename
        self.alpha = alpha
        self.n = n
        self.a = a
    
    def import_df(self, sheetname):
       
        self.df = pd.read_excel(self.filename, sheet_name=sheetname, header = 1)
        self.names = self.df['treatment'][:].values.tolist()
        self.df = self.df.T.tail(-1)
        self.df.columns = self.names

        self.df_melt = pd.melt(self.df.reset_index(), id_vars="index", value_vars=self.df.columns)
        self.df_melt.columns = ["index", "treatment", "value"]
        
        self.mean = []
        for col in self.df.columns:
            means = self.df[col].values.mean()
            self.mean.append(means)


        self.df_melt["means"] = np.repeat(self.mean,self.n)

        return self.df, self.df_melt, self.names, self.mean

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

        return self.df_melt, anova_table, pvalue, lsd, mse



    def boxplot(self):
        """
        boxplot Creates boxplot of the values in dataframe

        Args:
            df_melt (dataframe): newly formated dataframe
        """        
        plt.figure()
        sns.boxplot(data=self.df_melt, x="treatment", y="value")
        plt.show()

    def meanplot(self, unit=None):
        fig, ax = plt.subplots(1,1)
        ax.grid(True)
        ax.scatter(self.names, self.mean)
        ax.set_xlabel("treatment")
        if unit == None:
            ax.set_ylabel("mean")
        else:
            ax.set_ylabel(f"mean [{unit}]")
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


class Anova_Block:

    def __init__(self, filename, alpha, n, a):

        self.filename = filename
        self.n = n
        self.a = a
        self.alpha = alpha

    def melt(self, sheet_name):

        self.df = pd.read_excel(self.filename, sheet_name=sheet_name, header = 1)
        names = self.df['treatment'][:].values.tolist()
        self.df = self.df.T.tail(-1)
        self.df.columns = names

        self.df_melt = pd.melt(self.df.reset_index(), id_vars="index", value_vars=self.df.columns)
        self.df_melt.columns = ["block", "treatment", "value"]

        self.mean_row = self.df.mean(axis=0).values.tolist()
        self.mean_block = self.df.mean(axis=1).values.tolist()

        self.df_melt["mean_row"] = np.repeat(self.mean_row, self.n)
        self.df_melt["mean_column"] = self.mean_block*self.a


        return self.df, self.df_melt, self.mean_row, self.mean_block
    
    def anova(self):    
    
        model = ols("value ~ treatment+block", data=self.df_melt).fit()
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

        while i < len(self.mean_row)-1:
            for j in range(i+1, len(self.mean_row)):
                diff= np.abs(self.mean_row[i]-self.mean_row[j])
                if diff > lsd:
                    print("%i-%i: SIGNIFICANT DIFFERENCE!" %((i+1),(j+1)))
                else:
                    print("%i-%i: no significant difference " %(i+1,j+1))
            i+=1

        return self.df_melt, anova_table, pvalue, lsd

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
        z = stats.norm.ppf(chance)
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
        y_hat = self.df_melt['mean_column'].values + self.df_melt['mean_row'] - np.mean(self.mean_row + self.mean_block)


        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,8))
        ax1.stem(self.df_melt["treatment"], self.df_melt["residual"], basefmt='k')
        ax2.stem(self.df_melt["block"], self.df_melt["residual"], basefmt='k')
        ax1.set_xlabel("treatment")
        ax1.set_ylabel("residual")
        ax2.set_xlabel(r"$\bar{y}_i$")
        ax2.set_ylabel(r"residual")
        

        fig2, ax3 = plt.subplots(1,1, figsize=(12,8))

        ax3.scatter(y_hat, self.df_melt['residual'])
        ax3.set_xlabel(r"$\hat{y}_i$")
        ax3.set_ylabel(r"residual")

        plt.show()
