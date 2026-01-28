# import pandas as pd
# from pandas import read_parquet
#
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
#
#
# path = r"./gsm8k/train.parquet"
# data = read_parquet(path)
# print(data.count())
# print(data.head(n=1))


import pyarrow.parquet as pq

def read_parquet(filename: str) -> None:
    table = pq.read_table(filename)
    # column_names = table.columns()
    # print(table.columns)
    df = table.to_pandas()
    print(df)

if __name__ == "__main__":
    read_parquet(r"./gsm8k/train.parquet")


# import pyarrow.parquet as pq
# import pandas as pd
#
# file = pq.ParquetFile(r"./gsm8k/train.parquet")
# data = file.read().to_pandas()
#
# df = pd.DataFrame(data)
# csv_path = r"./data.csv"
# df.to_csv(csv_path)