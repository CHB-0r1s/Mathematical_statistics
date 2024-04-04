import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graphics import graphics
from graphics import df_schemas
from graphics import funcs


def round_up(number, x):
    return np.ceil(number * 10 ** x) / 10 ** x


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.Series(
    [-1.76, -0.291, -0.11, -0.45, 0.512, -0.158, 1.701, 0.634, 0.72, 0.49, 1.531, -0.433, 1.409, 1.74,
     -0.266, -0.058, 0.248, -0.095, -1.488, -0.361, 0.415, -1.382, 0.129, -0.361, -0.087, -0.329,
     0.086, 0.13, -0.244, -0.882, 0.318, -1.087, 0.899, 1.028, -1.304, 0.349, -0.293, 0.105, -0.056,
     0.757, -0.059, -0.539, -0.078, 0.229, 0.194, 0.123, 0.318, 0.367, -0.992, 0.529])

# Приводим к виду вариационного ряда
var_row = data.sort_values()
# var_row = df_schemas.table_to_var_row(pd.Series([0, 1, 2, 3, 4]), pd.Series([6, 7, 3, 6, 3]))
# print(f"{var_row.values=}")

# Объем выборки
N = var_row.count()
print(f"{N=}")

x_min = var_row.min()
x_max = var_row.max()
print(f"{x_min=}")
print(f"{x_max=}")

print(f"M = {float(var_row.mean())}")
print(f"D = {float(var_row.std())}")
print(f"Mode = {float(var_row.mode()[0])}")
print(f"Me = {float(pd.Series(var_row.unique().tolist()).median())}")
D_var = (1 / N) * sum([(elem - var_row.mean()) ** 2 for elem in var_row])
D_cor = D_var * (N / (N-1))
print(f"Выборочная дисперсия вариационного ряда = {D_var}")
print(f"Выборочное среднее квадратичное отклонение = {np.sqrt(D_var)}")
print(f"Исправленная дисперсия = {D_cor}")
print(f"Коэффицент асимметрии = {(1 / (N * (np.sqrt(D_var) ** 3))) * sum([(elem - var_row.mean()) ** 3 for elem in var_row])}")
print(f"Эксцесс = {(1 / (N * (np.sqrt(D_var) ** 4))) * sum([(elem - var_row.mean()) ** 4 for elem in var_row]) - 3}")

# Задаем шаг h
h = round_up(funcs.stergies_formula(x_min, x_max, N), 1)

intervals_series: pd.Series = pd.cut(var_row, bins=np.arange(start=x_min - h / 2, stop=x_max + h, step=h),
                                     include_lowest=True, right=False)
# Разбиваем данные на интервалы с шагом h
df = df_schemas.var_row_to_df_schema(var_row, x_min, x_max, h, N)

print(df)

# Создаем гистограмму
plt.style.use('_mpl-gallery')
fig, axs = plt.subplots(1, 4, figsize=(20, 5),
                        gridspec_kw={'left': 0.1, 'right': 0.9, 'bottom': 0.1, 'top': 0.9})

freq = [df[df['Interval'] == cur_interval]['Count'].values[0] for cur_interval in intervals_series.values]

graphics.gist_count(axs[0], df, var_row, freq, x_min, x_max, h)
graphics.gist_relative_count(axs[1], df, var_row, freq, x_min, x_max, h)

cum_freq = [df[df['Interval'] == cur_interval]['Cumulative_Frequency'].values[0]
            for cur_interval in intervals_series.values]

graphics.cumulyta(axs[2], df, var_row, cum_freq, x_min, x_max, h)
graphics.ogiva(axs[3], df, var_row, cum_freq, x_min, x_max, h)

funcs.empiric_func(df, intervals_series)

# plt.show()
