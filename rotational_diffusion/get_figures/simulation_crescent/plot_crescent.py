import datetime
import os

import pandas as pd
import matplotlib.pyplot as plt

top_path = os.path.join('rotational_diffusion', 'get_figures', 'simulation_crescent')
csv_path = os.path.join(top_path, 'data', '20230508_152725_crescent_beads.csv')
# Load the CSV data into a pandas dataframe
df = pd.read_csv(csv_path)

# Group the data by "crescent_intensity" and "rotational_diffusion_time_us"
grouped = df.groupby(["rotational_diffusion_time_ns_unpied", "crescent_intensity"])

# Create a grid of subplots based on the number of "crescent_intensity" groups
fig, axs = plt.subplots(nrows=len(df["rotational_diffusion_time_ns_unpied"].unique()), ncols=1, sharex=True, figsize=(6, 14))

# Set the color cycle for different "rotational_diffusion_time_us" values
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# find number of unique rotational diffusion times
num_unique_rdt = len(df["rotational_diffusion_time_ns_unpied"].unique())
num_unique_crescent = len(df["crescent_intensity"].unique())

group_names = {7249.0: '40nm', 24465.0: '60nm', 113263.0: '100nm', 906106.0: '200nm'}
# Plot each group on a separate subplot as a scatter plot with y error bars

for i, (name, group) in enumerate(grouped):
    group = group[:-60]
    rotational_diffusion_time_us, crescent_intensity = name
    ax = axs[df["rotational_diffusion_time_ns_unpied"].unique().tolist().index(rotational_diffusion_time_us)]
    ax.errorbar(group["collection_time_point_us"], group["ratio_xy_mean"], yerr=group["ratio_xy_std"],
                label=f"{crescent_intensity} crescent intensity", fmt='o', color=color_cycle[i % num_unique_crescent],
                alpha=0.2)
    ax.set_ylabel(f"XY Ratio, bead size: {group_names[rotational_diffusion_time_us]}")
    print(rotational_diffusion_time_us)
    ax.legend(loc='lower right')

    # Plot a rolling average line between the data points
    rolling_mean = group["ratio_xy_mean"].rolling(window=2, min_periods=1, center=True).mean()
    ax.plot(group["collection_time_point_us"], rolling_mean, color=color_cycle[i % num_unique_crescent])


# Set the x-axis label
axs[-1].set_xlabel("Collection time point (us)")

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# get datetime
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# save the figure
fig.savefig(os.path.join(top_path, 'plots', f'{now}_crescent.png'), dpi=300, bbox_inches='tight')

