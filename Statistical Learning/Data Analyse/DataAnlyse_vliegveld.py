import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import vliegveld

file_leroy = "Data.xlsx"

df = 


names = {"doorlooptijd [s]":"Handbagage", 
                    "doorlooptijd [s].1":"Bodyscan",
                    "doorlooptijd [s].2":"Douane"}

# meetnmr = index + 1
df = df.rename(columns=names)
df = df.loc[:, ~df.columns.str.contains("^metingnr")]

# haalt uit de namen van de kolommen uit de dataframe en zet ze in een lijst.
column_names = df.columns.values.tolist()

# Boxplot
fig, ax = plt.subplots(1,1)

# patch_artist zorgt ervoor dat de vierkantjes gevuld worden met een kleur (standaard = blauw)
ax.boxplot(df.values, patch_artist=True, labels=column_names)
plt.show()



