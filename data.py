import pandas as pd
import numpy as np

dates = pd.date_range('2021-01-01', '2023-01-01')
prices = 100 + np.cumsum(np.random.normal(0.1, 1, len(dates)))

df = pd.DataFrame({'Date': dates, 'Close': prices})
df.to_csv("sample_stock.csv", index=False)
