import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.linalg import norm
import warnings
import seaborn as sns
import statsmodels.graphics.gofplots as sm

class Import:
    def __init__(self):
        
        
        pass


    
    def import_data(filename, names):
        """
        import_data Import excel file to data framme

        Args:
            filename (str): Name of the excel file
            names (dict): dictionary of current column names and replaced column names (eg. {oldname:newname})

        Returns:
            Dataframe: Dataframe of the cleaned excel file
        """        

        df = pd.read_excel(filename, header=1)
        df = df.dropna(axis=1)
        df = df.loc[:,~df.columns.str.contains("^metingnr")]
        df = df.rename(columns=names)
        column_names = df.columns.values.tolist()

        return df, column_names


class Test:

    def test_normal(x):
        """
        test_normal Test data for normal distribution

        Args:
            x (array): list of values to test
        """        

        x = x / norm(x)

        warnings.filterwarnings("ignore", category=UserWarning)

        stat, p = scipy.stats.normaltest(x)

        print("\n", 100*"-", "\n")
        print("\nH0: The results follow a normal distribution.\n")

        print("\n", 100*"-", "\n")
        print("k^2: %.3f \nP-value: %.3g\n" %(stat, p))

        if p <= 0.05:
            print("Null Hypothesis rejected: The test results don't detect a significant relationship with the normal distribution.")
        
        else:
            print("Null Hypothesis accepted: The test results detect a significant relationship with the normal distribution.")
        print("\n", 100*"-", "\n")

    
class Plot:
    def Boxplot(df, labels):
        """
        Boxplot Create boxplot of given datapoint

        Args:
            df (datafram): dataframe containing the datapoints
            labels (list): list of columnsnames
        """        
        
        fig, ax = plt.subplots(1,1)
        ax.boxplot(df, patch_artist=True, labels=labels)


    def Normplot(x):
        """
        Normplot Hist- and kde-plot of data following a normal distribution. Also make qq (residu) plot for testing normaldistribution

        Args:
            x (array): array of datapoints to test
        """        
        fig, ax = plt.subplots(1,2)
        sns.histplot(x, kde=True, ax=ax[0])
        sm.ProbPlot(x).qqplot(line='s', ax=ax[1])
        ax[0].grid(True)
        ax[1].grid(True)

        "hoihoi"
    