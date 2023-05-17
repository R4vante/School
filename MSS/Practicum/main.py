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

    
   
if __name__ == "__main__":
    main()
