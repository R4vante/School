import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)
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
            plt.grid(True)
            plt.vlines(self.L[i][np.argmax(self.counts[i])], 0, np.max(self.counts[i]), linestyle='--', color='k', linewidth=1)
            plt.plot(self.L[i], self.counts[i])
            plt.title(f"Meting {i+1}")
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel('counts')
            plt.text(1.1*self.L[i][np.argmax(self.counts[i])], 0.85*np.max(self.counts[i]), f'$\lambda$ = {(self.L[i][np.argmax(self.counts[i])]):.2f}\ncounts = {np.max(self.counts[i]):.1f}')

        self.P = [2.1e-1, 2.9e-1, 4.1e-1, 5.6e-1, 8.0e-1, 9.2e-1, 1.2e0, 1.8e0, 2.2e0, 3.3e0, 4.3e0]
        self.voltage = [0.6, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.9, 2.5]
        self.max_counts = [np.max(self.counts[i]) for i in range(len(self.counts))]
    
        fig, ax = plt.subplots()
        plt.yscale("log") 
        ax.scatter(self.P, self.max_counts) 
        ax.set_xlabel("P (mbar)")
        ax.set_ylabel("counts")
        ax.grid()

        fig, ax = plt.subplots()
        ax.scatter(self.voltage, self.P)
        ax.set_xlabel("Breakout voltage")
        ax.set_ylabel("counts")
        ax.grid()
       

class ResN2:

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
            self.L.append(self.df_map[self.xls_sheetnames[i]].L)
            self.counts.append(self.df_map[self.xls_sheetnames[i]].counts)

        self.L_peak = {}
        self.counts_peak = {}

        for i in range(len(self.L)):
            L_new = []
            counts_new = []
            for k, l in enumerate(self.L[0]):
                if l>=300 and l<350:
                    L_new.append(self.L[i][k])
                    counts_new.append(self.counts[i][k])
            self.L_peak[i] = L_new
            self.counts_peak[i] = counts_new


        fig = plt.subplots(2,2, figsize=(15,10))

        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.grid(True)
            plt.vlines(self.L_peak[i][np.argmax(self.counts_peak[i])], 0, np.max(self.counts_peak[i]), linestyle = '--', color='k', linewidth=1)
            plt.plot(self.L[i], self.counts[i])
            plt.title(f"Meting {i+1}")
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel('counts')
            plt.text(1.1*self.L_peak[i][np.argmax(self.counts_peak[i])], 0.85*np.max(self.counts_peak[i]), f'$\lambda$ = {(self.L_peak[i][np.argmax(self.counts_peak[i])]):.2f}\ncounts = {np.max(self.counts[i]):.1f}')



        self.counts_p = [np.max(self.counts_peak[i]) for i in range(len(self.counts_peak))]
        self.P = [2.3e-1, 5.2e-1, 1.0e0, 2.1e0, 3.2e0, 4.1e0, 5.4e0]

        fig, ax = plt.subplots()
        plt.yscale("log")
        ax.scatter(self.P, self.counts_p)
        ax.set_xlabel('P (mbar)')
        ax.set_ylabel("counts")
        ax.grid(True)
