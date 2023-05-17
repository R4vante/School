import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ResCO2:

    def __init__(self, file):
        
        self.file = file

        xls = pd.ExcelFile(file)

        self.xls_sheetnames = xls.sheet_names
        self.xls_sheetnames.remove(self.xls_sheetnames[0])

        self.df_map = {}

        for sheet_name in self.xls_sheetnames:
            self.df_map[sheet_name] = xls.parse(sheet_name)

    def plot(self):

        self.L = []
        self.counts = []

        for i in range(len(self.xls_sheetnames)):
            self.L.append(self.df_map[self.xls_sheetnames[i]]['L (nm)'])
            self.counts.append(self.df_map[self.xls_sheetnames[i]].counts)

        fig = plt.subplots(2,2, figsize=(15,10))

        for i in range(4):
            plt.subplot(2,2, i+1)
            plt.vlines(self.L[i][np.argmax(self.counts[i])], 0, np.max(self.counts[i]), linestyle='--', color='k', linewidth=1)
            plt.plot(self.L[i], self.counts[i])
            plt.title(f"Meting {i+1}")
            plt.text(1.1*self.L[i][np.argmax(self.counts[i])], 0.5*np.max(self.counts[i]), f'$\lambda$ = {(self.L[i][np.argmax(self.counts[i])]):.2f}\ncounts = {np.max(self.counts[i])}')

        plt.show()

        print(self.L)
