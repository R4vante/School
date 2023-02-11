import pandas as pd
import matplotlib.pyplot as plt
from vliegveld import Import, Test, Plot
import seaborn as sns
import statsmodels.graphics.gofplots as sm

names = {"doorlooptijd [s]":"Handbagage", 
                    "doorlooptijd [s].1":"Bodyscan",
                    "doorlooptijd [s].2":"Douane"}



if __name__ == "__main__":
    df, column_names = Import.import_data("Data.xlsx", names)

    # meetnmr = index + 1


    ## Print Tables
    ## Dataframe
    print(df.describe())

    # Test for normal distribution
    print("\nHandbagage\n\n")
    print(Test.test_normal(df['Handbagage'].values))

    print("\nBodyscan\n\n")
    print(Test.test_normal(df['Bodyscan'].values))

    print("\nDouane\n\n")
    print(Test.test_normal(df['Douane'].values))



    # Boxplot
    Plot.Boxplot(df.values, column_names)

    Plot.Normplot(df['Handbagage'])
    plt.show()


