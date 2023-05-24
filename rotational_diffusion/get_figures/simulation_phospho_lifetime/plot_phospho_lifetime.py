import os

import pandas as pd
import matplotlib.pyplot as plt
import datetime


top_path = os.path.join('rotational_diffusion', 'get_figures', 'simulation_phospho_lifetime')
csv_path = os.path.join(top_path, 'data', '20230508_151410_phospho_lifetime.csv')
# Load the CSV data into a pandas dataframe
df = pd.read_csv(csv_path)

# Group the data by "rotational_diffusion_time_us_unpied"
grouped = df.groupby( "rotational_diffusion_time_ns_unpied")

# create a plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6*1.618, 6))

# Set the color cycle for different "rotational_diffusion_time_us" values
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# find number of unique rotational diffusion times
num_unique_rdt = len(df["rotational_diffusion_time_ns_unpied"].unique())

# Plot each group on the same subplot as a scatter plot with y error bars and different colors for each group
for i, (name, group) in enumerate(grouped):
    rotational_diffusion_time_us = name
    ax.errorbar(group["phosphorescence_lifetime"], group["ratio_xy_mean"], yerr=group["ratio_xy_std"],
                label=f"{int(rotational_diffusion_time_us)} ns", fmt='o', color=color_cycle[i % num_unique_rdt],
                alpha=0.2)
    ax.set_ylabel(f"XY Ratio")
    ax.legend()

    # Plot a rolling average line between the data points
    rolling_mean = group["ratio_xy_mean"].rolling(window=1, min_periods=1, center=True).mean()
    ax.plot(group["phosphorescence_lifetime"], rolling_mean, color=color_cycle[i % num_unique_rdt])

ax.set_xlabel("Phosphorescence lifetime (ns)")


# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# put legend on bottom right of plot
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.1), ncol=1, fancybox=True, shadow=False)

# get datetime
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# save the figure
fig.savefig(os.path.join(top_path, 'plots', f'{now}_phosphorescence_lifetime.png'), dpi=300, bbox_inches='tight')
