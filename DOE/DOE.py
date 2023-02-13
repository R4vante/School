import pandas as pd

class Import:

    def import_df(file, sheet_name):
        df = pd.read_excel(file, sheet_name=sheet_name, header = 1)
        names = df['treatment'][:].values.tolist()
        df = df.T.tail(-1)
        df.columns = names

        df_melt = pd.melt(df.reset_index(), id_vars="index", value_vars=df.columns)

        return df, df_melt, names
