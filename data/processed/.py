import pandas as pd

df = pd.read_csv('Iris.csv')

df = df.drop(columns=['Id'])
df['label'] = df['Species'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)
df = df.drop(columns=['Species', 'SepalLengthCm', 'SepalWidthCm'])
df.to_csv('iris.csv', index=False)