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
    [42.2, 42.0, 43.3, 43.7, 42.5, 42.9, 43.0, 42.7, 42.4, 42.6, 41.7, 41.9, 41.8, 43.2, 42.4, 42.5, 42.5, 42.5, 43.0,
     42.5, 41.2, 41.6, 42.6, 40.3, 41.6, 43.4, 43.5, 42.7, 43.1, 42.6, 42.5, 40.6, 44.1, 42.0, 40.4, 42.0, 43.1, 42.5,
     43.4, 42.5, 43.2, 42.8, 40.7, 43.6, 42.5, 41.1, 41.3, 43.6, 43.8, 42.8, 42.5, 43.3, 42.9, 40.8, 41.1, 42.3, 40.2,
     43.1, 43.7, 42.6, 42.8, 43.6, 42.5, 41.5, 40.0, 41.4, 42.1, 44.0, 42.4, 43.9, 42.4, 42.5, 42.4, 42.9, 42.2, 43.5,
     43.0, 42.8, 43.2, 43.4, 42.5, 42.6, 42.6, 42.4, 43.5, 43.8, 43.4, 42.5, 42.8, 41.3, 42.1, 42.5, 42.4, 43.4, 44.0,
     42.6, 42.5, 44.0, 41.7, 43.2]
)

# Приводим к виду вариационного ряда
var_row = data.sort_values()
print("Вариационный ряд:")
print(var_row.values)
# var_row = df_schemas.table_to_var_row(pd.Series([0, 1, 2, 3, 4]), pd.Series([6, 7, 3, 6, 3]))
# print(f"{var_row.values=}")
stat_row = df_schemas.row_to_table(data)
print("Статистический ряд:")
print(stat_row)


df_schemas.row_to_table(data)

# Объем выборки
N = var_row.count()
print(f"{N=}")

x_min = var_row.min()
x_max = var_row.max()
print(f"{x_min=}")
print(f"{x_max=}")
print(f"Размах = x_max - x_min = {x_max} - {x_min} = {x_max - x_min}")

print(f"M = {float(var_row.mean())}")
print(f"D = {float(var_row.std())}")
print(f"Mode = {float(var_row.mode()[0])}")
print(f"Me = {float(pd.Series(var_row.unique().tolist()).median())}")
D_var = (1 / N) * sum([(elem - var_row.mean()) ** 2 for elem in var_row])
D_cor = D_var * (N / (N - 1))
print(f"Выборочная дисперсия вариационного ряда = {D_var}")
print(f"Выборочное среднее квадратичное отклонение = {np.sqrt(D_var)}")
print(f"Исправленная дисперсия = {D_cor}")
print(
    f"Коэффицент асимметрии = {(1 / (N * (np.sqrt(D_var) ** 3))) * sum([(elem - var_row.mean()) ** 3 for elem in var_row])}")
print(f"Эксцесс = {(1 / (N * (np.sqrt(D_var) ** 4))) * sum([(elem - var_row.mean()) ** 4 for elem in var_row]) - 3}")

# Задаем шаг h
h = round_up(funcs.stergies_formula(x_min, x_max, N), 1)
print("Округляем полученный результат вверх")
print(f"Итоговая h: {h}")

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

#graphics.gist_count(axs[0], df, var_row, freq, x_min, x_max, h)
#graphics.gist_relative_count(axs[1], df, var_row, freq, x_min, x_max, h)

cum_freq = [df[df['Interval'] == cur_interval]['Cumulative_Frequency'].values[0]
            for cur_interval in intervals_series.values]

#graphics.cumulyta(axs[2], df, var_row, cum_freq, x_min, x_max, h)
#graphics.ogiva(axs[3], df, var_row, cum_freq, x_min, x_max, h)

x_for_F, y_for_F = funcs.empiric_func(df, intervals_series)
print(x_for_F)
print(y_for_F)

# plt.show()
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.subplot.left': 0.1,
    'figure.subplot.right': 0.9,
    'figure.subplot.bottom': 0.1,
    'figure.subplot.top': 0.9
})
plt.figure(figsize=(8, 6))  # Задаем размер графика
plt.plot(x_for_F, y_for_F, marker='o', linestyle='-')  # Строим график через точки с заданными координатами
plt.xlabel('x')  # Подписываем ось x
plt.ylabel('y')  # Подписываем ось y
plt.grid(True)  # Добавляем сетку на график
plt.show()  # Отображаем график