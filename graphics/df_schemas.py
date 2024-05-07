import pandas as pd
import numpy as np


def var_row_to_df_schema(var_row, x_min, x_max, h, N) -> pd.DataFrame:
    intervals_series: pd.Series = pd.cut(var_row, include_lowest=True, right=False,
                                         bins=np.arange(start=x_min - h/2, stop=x_max + h, step=h))

    # Получаем dataframe интервалов и частот
    df = intervals_series.value_counts(sort=False).reset_index()
    df.columns = ['Interval', 'Count']
    df.index.name = '№'
    df.index += 1
    df['Relative_count'] = df['Count'] / N
    df['Cumulative_Frequency'] = df['Count'].cumsum()
    df['Cumulative_Count'] = df['Relative_count'].cumsum()
    df['Middle'] = df['Interval'].apply(lambda x: (x.left + x.right) / 2)
    return df


def table_to_var_row(x: pd.Series, n: pd.Series) -> pd.Series:
    res = []
    for i in range(len(x)):
        res += [x[i]] * n[i]
    return pd.Series(res).sort_values()

def row_to_table(x: pd.Series) -> pd.Series: # Статистический ряд
    res = x.value_counts()
    return res
