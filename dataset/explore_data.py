import pandas as pd

file_path = r"E:\My Things\task5\dataset\rows.xlsx"
df = pd.read_excel(file_path)

print(df.head())
print(df.info())
