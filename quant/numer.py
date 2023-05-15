from numerapi import NumerAPI
import pandas as pd
import lightgbm as lgbm
import pyarrow as pa
import pyarrow.parquet as pq

napi = NumerAPI()

napi.download_dataset("v4.1/train.parquet", "train.parquet")
napi.download_dataset("v4.1/live.parquet", "live.parquet")

df = pd.read_parquet("train.parquet")
print(df)
