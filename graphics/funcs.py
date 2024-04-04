import numpy as np


def empiric_func(df, var_rows):
    print("----------------------------")
    print(f"0, x <= {var_rows.min().left}")
    for index, row in df.iterrows():
        if index != len(df):
            print(f"{row['Cumulative_Count']}, {row['Interval'].left} < x <= {row['Interval'].right}")
        else:
            print(f"{row['Cumulative_Count']}, x > {row['Interval'].left}")
    print("----------------------------")


def stergies_formula(x_min, x_max, n) -> int:
    return (x_max - x_min) / (1 + 3.321 * np.log10(n))