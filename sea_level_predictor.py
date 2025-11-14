import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    df = pd.read_csv("epa-sea-level.csv")

    plt.figure(figsize=(10, 6))
    plt.scatter(df["Year"], df["CSIRO Adjusted Sea Level"])

    res_all = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])

    years_extended = pd.Series(range(df["Year"].min(), 2051))
    plt.plot(years_extended,
             res_all.slope * years_extended + res_all.intercept,
            label="Fit: All Data")

    df_2000 = df[df["Year"] >= 2000]
    res_2000 = linregress(df_2000["Year"], df_2000["CSIRO Adjusted Sea Level"])

    years_extended_2000 = pd.Series(range(2000, 2051))
    plt.plot(years_extended_2000,
            res_2000.slope * years_extended_2000 + res_2000.intercept,
            label="Fit: From 2000")

    plt.xlabel("Year")
    plt.ylabel("Sea Level (inches)")
    plt.title("Rise in Sea Level")

    plt.savefig("sea_level_plot.png")

    return plt.gca()
