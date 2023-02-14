import pandas as pd
import matplotlib.pyplot as plt
from DOE import Data
import seaborn as sns
import statsmodels.graphics.gofplots as sm

column_names = {"doorlooptijd [s]":"Handbagage", 
                    "doorlooptijd [s].1":"Bodyscan",
                    "doorlooptijd [s].2":"Douane"}



if __name__ == "__main__":
    n = 10
    a = 3

    df, df_melt, names, mean = Data.import_df("Data.xlsx", column_names, n)
    print(df_melt)

    Data.boxplot(df_melt)

    Data.anova(df_melt, 0.05)

    Data.res_plot(df_melt)

    Data.norm_plot(df_melt)


