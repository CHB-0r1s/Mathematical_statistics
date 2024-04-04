import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gist_relative_count(ax, df, var_row, freq, x_min, x_max, h):
    ax.step(
        [elem.right for elem in df['Interval']],
        [elem for elem in df['Relative_count']], linewidth=2.5
    )
    ax.set(xlim=(x_min, x_max), xticks=np.arange(start=x_min, stop=x_max, step=h))
    ax.set_title('Гистограмма распределения относительных частот')
    ax_1 = ax.twinx()
    ax_1.plot(var_row.tolist(), freq, linewidth=1.5)


def gist_count(ax, df, var_row, freq, x_min, x_max, h):
    ax.step(
        [x_min] + [elem.right for elem in df['Interval']] + [max([elem.right for elem in df['Interval']])],
        [0] + [elem for elem in df['Count']] + [0], linewidth=2.5
    )
    ax.set(xlim=(x_min, x_max), xticks=np.arange(start=x_min, stop=x_max + h, step=h))
    ax.set_title('Гистограмма распределения частот')


def cumulyta(ax, df, var_row, cum_freq, x_min, x_max, h):
    ax.plot(var_row.tolist(), cum_freq, 'om-', linewidth=2.5)
    ax.set(xlim=(x_min, x_max), xticks=np.arange(start=x_min, stop=x_max, step=h))
    ax.set_title('Кумулята')


def ogiva(ax, df, var_row, cum_freq, x_min, x_max, h):
    ax.plot(cum_freq, var_row.tolist(), 'om-', linewidth=2.5)
    ax.set(xlim=(0, df['Cumulative_Frequency'].max()))
    ax.set_title('Огива')