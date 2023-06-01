"""
    main.py: main script for results MSS lab

    packages needed:
        - numpy
        - matplotlib
        - pandas
        - openpyxl

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8-colorblind"])
import resultaten


def main():
    
    res = resultaten.ResCO2("Metingen/2_CO2_P.xlsx")
    res.plot()

    resN2 = resultaten.ResN2("Metingen/N2_nieuw.xlsx")
    resN2.plot()

    resLuft = resultaten.ResLuftN2("Metingen/Lucht_P.xlsx")
    resLuft.plot()
    
    resLuftco2 = resultaten.ResLuftCO2("Metingen/Lucht_P.xlsx")
    resLuftco2.plot()

    P = [resN2.P, resLuft.P]
    counts_p = [resN2.counts_p, resLuft.counts_p]

    fig = plt.figure("N2 + Air")
    # ax = plt.subplots(1,2)
    for i in range(len(P)):
        plt.subplot(1,2,i+1)
        plt.yscale('log')
        plt.scatter(P[i], counts_p[i])

    plt.show()
    
   
if __name__ == "__main__":
    main()
