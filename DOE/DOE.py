import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

class Import:

    def import_df(file, sheet_name, n):
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
        df = pd.read_excel(file, sheet_name=sheet_name, header = 1)
        names = df['treatment'][:].values.tolist()
        df = df.T.tail(-1)
        df.columns = names

        df_melt = pd.melt(df.reset_index(), id_vars="index", value_vars=df.columns)
        df_melt.columns = ["index", "treatment", "value"]
        
        mean = []
        for col in df.columns:
            means = df[col].values.mean()
            mean.append(means)
        means

        df_melt["means"] = np.repeat(mean,n)

        return df, df_melt, names, mean


class Calc:

    def anova(df_melt, alpha):
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
        
        model = ols("value~treatment", data=df_melt).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        pvalue = anova_table["PR(>F)"]["treatment"]
        if pvalue < alpha:
            print("H0 rejected")
        else:
            print("H0 accepted")

        df_melt["residual"] = model.resid

        return df_melt, model, anova_table 

    def LSD(anova_table, mean, alpha, n, a):
        """
        LSD Doing a Least Significant Difference Analysis. Outputs the influence of every group.

        Args:
            anova_table (dataframe): anova table
            mean (list): means of the treatments
            alpha (float): significance level
            n (int): number of replicates
            a (int): number of treatments

        Returns:
            lsd: float: Least Significant Difference value
        """        

        sse = anova_table["sum_sq"]["Residual"]
        dfe = anova_table["df"]["Residual"]
        mse = sse/(a*(n-1))
        t_crit = stats.t.ppf(q=1-alpha/2, df=dfe)

        lsd = t_crit * np.sqrt(2*mse/n)

        i = 0

        while i < len(mean)-1:
            for j in range(i+1, len(mean)):
                diff= np.abs(mean[i]-mean[j])
                if diff > lsd:
                    print("%i-%i: SIGNIFICANT DIFFERENCE!" %((i+1),(j+1)))
                else:
                    print("%i-%i: no significant difference " %(i+1,j+1))
            i+=1

        return lsd


class Plot:

    def boxplot(df_melt):
        """
        boxplot Creates boxplot of the values in dataframe

        Args:
            df_melt (dataframe): newly formated dataframe
        """        
        plt.figure()
        sns.boxplot(data=df_melt, x="treatment", y="value")
        plt.show()

    def norm_plot(df_melt):
        """
        norm_plot Creates a probability plot of residuals

        Args:
            df_melt (df): newly formated dataframe
        """        
        plt.figure()
        stats.probplot(df_melt["residual"], dist="norm", plot=plt)
        plt.xlabel("z-score")
        plt.ylabel("residual")
        plt.show()

    def res_plot(df_melt):
        """
        res_plot Creates residualplots

        Args:
            df_melt (dataframe): newly formated dataframe
        """        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,8))
        ax1.stem(df_melt["treatment"], df_melt["residual"], basefmt='k')
        ax2.stem(df_melt["means"], df_melt["residual"], basefmt='k')
        ax1.set_xlabel("treatment")
        ax1.set_ylabel("residual")
        ax2.set_xlabel(r"$\bar{y}_i$")
        ax2.set_ylabel(r"residual")

        plt.show()
