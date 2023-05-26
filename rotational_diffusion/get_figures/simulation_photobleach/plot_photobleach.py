import os

import pandas as pd
import matplotlib.pyplot as plt
import datetime

check_photobleach_rate = 0.05

top_path = os.path.join('rotational_diffusion', 'get_figures', 'simulation_photobleach')
csv_path = os.path.join(top_path, 'data', '20230524_133450_photobleach.csv')
# Load the CSV data into a pandas dataframe
df_all = pd.read_csv(csv_path)
df_all = df_all[df_all["bleach_rate"] == check_photobleach_rate]

# Group the data by "rotational_diffusion_time_us_unpied"
atpase = {'ab': 250, 'ab-m': 260, 'ab-agg1': 417, 'ab-sol': 666, 'ab-proto': 2000}  # these get multiplied by pi during the simulation
group_names = list(atpase.keys())

# get unique singlet intensities in the df
# singlet_intensities = df_all["singlet_intensity"].unique()

# for singlet_intensity in singlet_intensities:
#     df = df_all[df_all["singlet_intensity"] == singlet_intensity]
df = df_all
grouped = df.groupby("rotational_diffusion_time_ns_unpied")

# create a plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(7, 6))

# Set the color cycle for different "rotational_diffusion_time_us" values
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# find number of unique rotational diffusion times
num_unique_rdt = len(df["rotational_diffusion_time_ns_unpied"].unique())

# Plot each group on the same subplot as a scatter plot with y error bars and different colors for each group
for i, (name, group) in enumerate(grouped):
    rotational_diffusion_time_us = name
    ax.errorbar(group["singlet_intensity"], group["ratio_xy_mean"], yerr=group["ratio_xy_std"],
                label=f"{group_names[i]}", fmt='o', color=color_cycle[i % num_unique_rdt],
                alpha=0.2)
    ax.set_ylabel(f"XY Ratio")
    ax.legend()

    # Plot a rolling average line between the data points
    rolling_mean = group["ratio_xy_mean"].rolling(window=1, min_periods=1, center=True).mean()
    ax.plot(group["singlet_intensity"], rolling_mean, color=color_cycle[i % num_unique_rdt])

ax.set_xlabel("singlet_intensities")

#set ylim between 1 and 4
ax.set_ylim(0.5, 1)
# set x as logarithmic
# ax.set_xscale('log')

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# put legend on bottom right of plot
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.1), ncol=1, fancybox=True, shadow=False)

# get datetime
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# str_intensity = str(singlet_intensity).replace('.', 'p')
str_bleach = str(check_photobleach_rate).replace('.', 'p')

# save the figure
csv_dir = os.path.join(top_path, 'plots')
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
fig.savefig(os.path.join(csv_dir, f'{now}-bleach_{str_bleach}.png'),
            dpi=300, bbox_inches='tight')

# close plot
plt.close(fig)
