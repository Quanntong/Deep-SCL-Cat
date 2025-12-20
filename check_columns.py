import pandas as pd

df = pd.read_csv('data/raw/scl90_data.csv', encoding='utf-8-sig')
print('原始列名:')
for i, col in enumerate(df.columns):
    print(f'{i}: {repr(col)}')
    
print('\n清洗后列名（使用data_loader的清洗逻辑）:')
df.columns = df.columns.astype(str).str.strip()
df.columns = df.columns.str.replace(r'[\n\r\t\s]+', '', regex=True)
df.columns = df.columns.str.replace(r'[\"\'`]', '', regex=True)

for i, col in enumerate(df.columns):
    print(f'{i}: {repr(col)}')
