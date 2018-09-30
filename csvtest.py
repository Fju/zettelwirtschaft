import pandas as pd

df = pd.read_csv('train_data/labels.csv', sep=';')
print(df.shape)