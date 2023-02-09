import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import warnings
import sys

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



    def Test_normal(x):

        warnings.filterwarnings("ignore", category=UserWarning)

        k2, p = scipy.stats.normaltest(x)

        print("\n", 100*"-", "\n")
        print("\nH0: The results follow a normal distribution.\n")

        print("\n", 100*"-", "\n")
        print("k^2: %.3f \nP-value: %.3g\n" %(k2, p))

        if p <= 0.5:
            print("Null Hypothesis rejected: The test results don't detect a significant relationship with the normal distribution.")
        
        else:
            print("Null Hypothesis accepted: The test results detect a significant relationship with the normal distribution.")
        print("\n", 100*"-", "\n")

    
    def Boxplot(df, labels):
        fig, ax = plt.subplots(1,1)
        ax.boxplot(df, patch_artist=True, labels=labels)
        plt.show()

    