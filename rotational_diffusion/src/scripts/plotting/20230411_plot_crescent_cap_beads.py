import pandas as pd
import matplotlib.pyplot as plt

csv_path = r'C:\Users\austin\GitHub\Rotational_diffusion-AELxJLD\rotational_diffusion\data\20230411_beads.csv'
# Load the CSV data into a pandas dataframe
df = pd.read_csv(csv_path)

# Group the data by "crescent_intensity" and "rotational_diffusion_time_us"
grouped = df.groupby(["crescent_intensity", "rotational_diffusion_time_ns_unpied"])

# Create a grid of subplots based on the number of "crescent_intensity" groups
fig, axs = plt.subplots(nrows=len(df["crescent_intensity"].unique()), ncols=1, sharex=True, figsize=(6, 10))

# Set the color cycle for different "rotational_diffusion_time_us" values
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# find number of unique rotational diffusion times
num_unique_rdt = len(df["rotational_diffusion_time_ns_unpied"].unique())

group_names = ['40nm', '60nm', '100nm', '200nm']

# Plot each group on a separate subplot as a scatter plot with y error bars

for i, (name, group) in enumerate(grouped):
    crescent_intensity, rotational_diffusion_time_us = name
    ax = axs[df["crescent_intensity"].unique().tolist().index(crescent_intensity)]
    ax.errorbar(group["collection_time_point_us"], group["ratio_xy_mean"], yerr=group["ratio_xy_std"],
                label=f"Bead: {group_names[i % num_unique_rdt]}", fmt='o', color=color_cycle[i % num_unique_rdt],
                alpha=0.2)
    ax.set_ylabel(f"XY Ratio, crescent intensity: {crescent_intensity}")
    ax.legend()

    # Plot a rolling average line between the data points
    rolling_mean = group["ratio_xy_mean"].rolling(window=10, min_periods=1, center=True).mean()
    ax.plot(group["collection_time_point_us"], rolling_mean, color=color_cycle[i % num_unique_rdt])


# Set the x-axis label
axs[-1].set_xlabel("Collection time point (us)")

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

