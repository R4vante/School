import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', axisbelow=True)
import pandas as pd

class ResCO2:
    """
     Class to plot the results of C02 measurements
    """
    def __init__(self, file):

        """
        Function: _init__

        Initializing the parameters
        


        Parameters:
            
            -file (str): file path

        """
        
        self.file = file

        xls = pd.ExcelFile(file)

        self.xls_sheetnames = xls.sheet_names
        self.xls_sheetnames.remove(self.xls_sheetnames[0])

        self.df_map = {}

        for sheet_name in self.xls_sheetnames:
            self.df_map[sheet_name] = xls.parse(sheet_name)

    def plot(self):
        """
        Plot: Ploting the results for CO2 measurements. Results plotted:
            - Results spectrometry per measured pressure
            - Highest counts against pressure (logarthmic y-axis)
            - Pressure against breakout voltage
        """

        self.L = []
        self.counts = []
        self.P = [2.1e-1, 2.9e-1, 4.1e-1, 5.6e-1, 8.0e-1, 9.2e-1, 1.2e0, 1.8e0, 2.2e0, 3.3e0, 4.3e0]
        self.voltage = [0.6, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.9, 2.5]

        for i in range(len(self.xls_sheetnames)):
            self.L.append(self.df_map[self.xls_sheetnames[i]]['L (nm)'])
            self.counts.append(self.df_map[self.xls_sheetnames[i]].counts)

        fig = plt.figure("CO2 spectra", figsize=(15,10))

        for i in range(4):
            plt.subplot(2,2, i+1)
            plt.grid(True)
            plt.vlines(self.L[i][np.argmax(self.counts[i])], 0, np.max(self.counts[i]), linestyle='--', color='k', linewidth=1)
            plt.plot(self.L[i], self.counts[i])
            plt.title(f"P = {self.P[i]} mbar")
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel('counts')
            plt.text(1.1*self.L[i][np.argmax(self.counts[i])], 0.85*np.max(self.counts[i]), f'$\lambda$ = {(self.L[i][np.argmax(self.counts[i])]):.2f}\ncounts = {np.max(self.counts[i]):.1f}')

        
        
        self.max_counts = [np.max(self.counts[i]) for i in range(len(self.counts))]
    
        fig = plt.figure("CO2 druk")
        ax = fig.add_subplot()
        plt.yscale("log") 
        ax.scatter(self.P, self.max_counts) 
        ax.set_xlabel("P (mbar)")
        ax.set_ylabel(r"$\log_{10} \left(counts\right)$")
        ax.grid()

        fig = plt.figure("CO2 voltage")
        ax = fig.add_subplot()
        ax.scatter(self.P, self.voltage)
        ax.set_xlabel("P (mbar)")
        ax.set_ylabel("Breakout voltage (kV)")
        ax.grid()
       

class ResN2:

    """
    Class to plot N2 measurement results
    """

    def __init__(self, file):
        """
        Function: __init__:

        Initializing parameters



        Parameters:
            - file (str): file path to excel file
        """

        self.file = file

        xls = pd.ExcelFile(file)

        self.xls_sheetnames = xls.sheet_names

        self.xls_sheetnames.remove(self.xls_sheetnames[0])

        self.df_map = {}

        for sheet_name in self.xls_sheetnames:
            self.df_map[sheet_name] = xls.parse(sheet_name)
        

    def plot(self):

        """
        Function: Plot

        Plotting results of the N2 measurement. Plots created:
            - Results spectrometry per measured pressure
            - Highest counts against pressure (logarthmic y-axis)
        """
# TODO: create plot against breakdown voltage when available



        self.L = []
        self.counts = []
        self.P = [1.6e-1, 3.0e-1, 7.2e-1, 1.1e0, 1.8e0, 2.8e0, 3.7e0, 4.2e0, 4.8e0, 5.8e0, 7.1e0]
        self.voltage = [0.7, 0.6, 0.7, 0.9, 1.4, 2.1, 2.8, 2.4, 2.2, 2.3, 2.4]

        for i in range(len(self.xls_sheetnames)):
            self.L.append(self.df_map[self.xls_sheetnames[i]].L)
            self.counts.append(self.df_map[self.xls_sheetnames[i]].counts)

        self.L_peak = {}
        self.counts_peak = {}

        for i in range(len(self.L)):
            L_new = []
            counts_new = []
            for k, l in enumerate(self.L[0]):
                if l>=330 and l<360:
                    L_new.append(self.L[i][k])
                    counts_new.append(self.counts[i][k])
            self.L_peak[i] = L_new
            self.counts_peak[i] = counts_new


        fig = plt.figure("N2 spectra", figsize=(15,10))

        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.grid(True)
            plt.vlines(self.L_peak[i][np.argmax(self.counts_peak[i])], 0, np.max(self.counts_peak[i]), linestyle = '--', color='k', linewidth=1)
            plt.plot(self.L[i], self.counts[i])
            plt.title(f"P = {self.P[i]} mbar")
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel('counts')
            plt.text(1.1*self.L_peak[i][np.argmax(self.counts_peak[i])], 0.85*np.max(self.counts_peak[i]), f'$\lambda$ = {(self.L_peak[i][np.argmax(self.counts_peak[i])]):.2f}\ncounts = {np.max(self.counts_peak[i]):.1f}')



        self.counts_p = [np.max(self.counts_peak[i]) for i in range(len(self.counts_peak))]
        




# NOTE: The counts vs pressure doesn't seem to be linear. Needs to be fixed --> could be due to the fact of chosing the second hightest peak instead of highest.
# Could also be due gas configuration (not properly filling the vacuum tube)


        fig = plt.figure("N2 druk")
        ax = fig.add_subplot()
        plt.yscale("log")
        ax.scatter(self.P, self.counts_p)
        ax.set_xlabel('P (mbar)')
        ax.set_ylabel(r"$\log_{10} \left(counts\right)$")
        ax.grid(True)

        fig = plt.figure("N2 voltage")
        ax = fig.add_subplot()
        ax.scatter(self.P, self.voltage)
        ax.set_xlabel("P (mbar)")
        ax.set_ylabel("Breakout voltage (kV)")
        ax.grid()


class ResLuftN2:

    """
    Class to plot N2 measurement results
    """

    def __init__(self, file):
        """
        Function: __init__:

        Initializing parameters



        Parameters:
            - file (str): file path to excel file
        """

        self.file = file

        xls = pd.ExcelFile(file)

        self.xls_sheetnames = xls.sheet_names

        self.xls_sheetnames.remove(self.xls_sheetnames[0])

        self.df_map = {}

        for sheet_name in self.xls_sheetnames:
            self.df_map[sheet_name] = xls.parse(sheet_name)
        

    def plot(self):

        """
        Function: Plot

        Plotting results of the N2 measurement. Plots created:
            - Results spectrometry per measured pressure
            - Highest counts against pressure (logarthmic y-axis)
        """
# TODO: create plot against breakdown voltage when available



        self.L = []
        self.counts = []
        self.P = [2.2e-1, 3.8e-1, 8.1e-1, 1.1e0, 2.1e0, 2.7e0, 3.4e0, 4.6e0]
        self.voltage = [0.6, 0.8, 0.8, 0.9, 1.4, 1.5, 1.7, 2.0]

        for i in range(len(self.xls_sheetnames)):
            self.L.append(self.df_map[self.xls_sheetnames[i]].L)
            self.counts.append(self.df_map[self.xls_sheetnames[i]].counts)

        self.L_peak = {}
        self.counts_peak = {}

        for i in range(len(self.L)):
            L_new = []
            counts_new = []
            for k, l in enumerate(self.L[0]):
                if l>=330 and l<360:
                    L_new.append(self.L[i][k])
                    counts_new.append(self.counts[i][k])
            self.L_peak[i] = L_new
            self.counts_peak[i] = counts_new


        fig = plt.figure('Luft spectra', figsize=(15,10))

        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.grid(True)
            plt.vlines(self.L_peak[i][np.argmax(self.counts_peak[i])], 0, np.max(self.counts_peak[i]), linestyle = '--', color='k', linewidth=1)
            plt.plot(self.L[i], self.counts[i])
            plt.title(f"P = {self.P[i]} mbar")
            plt.xlabel("$\lambda$ (nm)")
            plt.ylabel('counts')
            plt.text(1.1*self.L_peak[i][np.argmax(self.counts_peak[i])], 0.85*np.max(self.counts_peak[i]), f'$\lambda$ = {(self.L_peak[i][np.argmax(self.counts_peak[i])]):.2f}\ncounts = {np.max(self.counts_peak[i]):.1f}')



        self.counts_p = [np.max(self.counts_peak[i]) for i in range(len(self.counts_peak))]
        




# NOTE: The counts vs pressure doesn't seem to be linear. Needs to be fixed --> could be due to the fact of chosing the second hightest peak instead of highest.
# Could also be due gas configuration (not properly filling the vacuum tube)


        fig = plt.figure("Luft druk")
        ax = fig.add_subplot()
        plt.yscale("log")
        ax.scatter(self.P, self.counts_p)
        ax.set_xlabel('P (mbar)')
        ax.set_ylabel(r"$\log_{10} \left(counts\right)$")
        ax.grid(True)

        fig = plt.figure("Luft voltage")
        ax = fig.add_subplot()
        ax.scatter(self.P, self.voltage)
        ax.set_xlabel("P (mbar)")
        ax.set_ylabel("Breakout voltage (kV)")
        ax.grid()



class ResLuftCO2:
    """
     Class to plot the results of C02 measurements
    """
    def __init__(self, file):

        """
        Function: _init__

        Initializing the parameters
        


        Parameters:
            
            -file (str): file path

        """
        
        self.file = file

        xls = pd.ExcelFile(file)

        self.xls_sheetnames = xls.sheet_names
        self.xls_sheetnames.remove(self.xls_sheetnames[0])

        self.df_map = {}

        for sheet_name in self.xls_sheetnames:
            self.df_map[sheet_name] = xls.parse(sheet_name)

    def plot(self):
        """
        Plot: Ploting the results for CO2 measurements. Results plotted:
            - Results spectrometry per measured pressure
            - Highest counts against pressure (logarthmic y-axis)
            - Pressure against breakout voltage
        """

        self.L = []
        self.counts = []
        self.P = [2.2e-1, 3.8e-1, 8.1e-1, 1.1e0, 2.1e0, 2.7e0, 3.4e0, 4.6e0]
        self.voltage = [0.6, 0.8, 0.8, 0.9, 1.4, 1.5, 1.7, 2.0]

        for i in range(len(self.xls_sheetnames)):
            self.L.append(self.df_map[self.xls_sheetnames[i]].L)
            self.counts.append(self.df_map[self.xls_sheetnames[i]].counts)

        self.L_peak = {}
        self.counts_peak = {}

        for i in range(len(self.L)):
            L_new = []
            counts_new = []
            for k, l in enumerate(self.L[0]):
                if l>=480 and l<485:
                    L_new.append(self.L[i][k])
                    counts_new.append(self.counts[i][k])
            self.L_peak[i] = L_new
            self.counts_peak[i] = counts_new

        self.counts_p = [np.max(self.counts_peak[i]) for i in range(len(self.counts_peak))]
        




# NOTE: The counts vs pressure doesn't seem to be linear. Needs to be fixed --> could be due to the fact of chosing the second hightest peak instead of highest.
# Could also be due gas configuration (not properly filling the vacuum tube)


        fig = plt.figure("Luft-CO2 druk")
        ax = fig.add_subplot()
        plt.yscale("log")
        ax.scatter(self.P, self.counts_p)
        ax.set_xlabel('P (mbar)')
        ax.set_ylabel(r"$\log_{10} \left(counts\right)$")
        ax.grid(True)

        fig = plt.figure("Luft-CO2 voltage")
        ax = fig.add_subplot()
        ax.scatter(self.P, self.voltage)
        ax.set_xlabel("P (mbar)")
        ax.set_ylabel("Breakout voltage (kV)")
        ax.grid()
