import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(["seaborn-v0_8-colorblind"])


def main():

    xls = pd.ExcelFile('Metingen/2_CO2_P.xlsx') 

    xls_sheetnames = ['mt1', 'mt2', 'mt3', 'mt4', 'mt5', 'mt6', 'mt7', 'mt8', 'mt9', 'mt10']

    df_map = {}

    for sheet_name in xls_sheetnames:
        df_map[sheet_name] = xls.parse(sheet_name)

    L = []
    counts = []

    for i in range(len(xls_sheetnames)):
        L.append(df_map[xls_sheetnames[i]]["L (nm)"])
        counts.append(df_map[xls_sheetnames[i]].counts)
    
    fig = plt.subplots(2,2,figsize=(10,12))

    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(L[i], counts[i])
        plt.text(x=L[i][counts[i].argmax()]*1.1, y=0.7*np.max(counts[i]), s=f"$\lambda$ ={L[i][counts[i].argmax()]:.2f}\ncounts={np.max(counts[i])}")
        plt.vlines(x=L[i][counts[i].argmax()], ymin=0, ymax=np.max(counts[i]) + 100, color='k', linewidth=1, linestyle='--')
        plt.title(f"Meting {i+1}")

    plt.show()
    
   
if __name__ == "__main__":
    main()
