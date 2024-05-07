import numpy as np


def empiric_func(df, var_rows) -> (list[float], list[float]):
    print("----------------------------")
    print(f"0, x <= {var_rows.min().left}")
    x = [var_rows.min().left]
    y = [0]
    for index, row in df.iterrows():
        if index != len(df):
            print(f"{row['Cumulative_Count']}, {row['Interval'].left} < x <= {row['Interval'].right}")
            x.append(float(row['Interval'].right))
            y.append(float(row['Cumulative_Count']))
        else:
            print(f"{row['Cumulative_Count']}, x > {row['Interval'].left}")
            x.append(row['Interval'].left)
            y.append(1)
    print("----------------------------")
    return x, y


def stergies_formula(x_min, x_max, n) -> int:
    print("Расчет по формуле Стержеса:")
    print(f"h = (x_max - x_min) / (1 + 3.321 * log10(n)) = ({x_max} - {x_min}) / (1 + 3.321 * {np.log10(n)})")
    print(f"h = ({x_max - x_min}) / {1 + 3.321 * np.log10(n)} = {(x_max - x_min) / (1 + 3.321 * np.log10(n))}")
    return (x_max - x_min) / (1 + 3.321 * np.log10(n))