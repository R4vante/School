import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Vliegveld:
    def __init__(self, filename, column_names):
        """
        __init__ Initialising function

        Args:
            filename (str): Name of the file
            column_names (list): dictionary containing column names
        """        
        self.filename = filename
        self.names = column_names


    
    def import_data(self.filename):
        df = pd.read_excel(self.filename, header=1)
        df = df.dropna(axis=1)
        df = df.loc[:, ~df.columns.contains("^metingnr")]
        df = df.rename(columns=self.names)

        return df