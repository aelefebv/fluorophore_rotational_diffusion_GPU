import os

import pandas as pd
import matplotlib.pyplot as plt
import datetime


top_path = os.path.join('rotational_diffusion', 'get_figures', 'simulation_OADF_cw')
csv_path = os.path.join(top_path, 'data', '20230508_145902_cw_lifetime.csv')
# Load the CSV data into a pandas dataframe
df = pd.read_csv(csv_path)

# create a plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6*1.618, 6))


# Plot each group on the same subplot as a scatter plot with y error bars and different colors for each group
# ax.errorbar(df["trigger_intensity"], df["calculated_lifetime_mean"], yerr=df["calculated_lifetime_std"],
#             fmt='o', color='black',
#             alpha=0.2)
ax.errorbar(df["trigger_intensity"], df["ratio_xy_mean"], yerr=df["ratio_xy_std"],
            fmt='o', color='black',
            alpha=0.2)
ax.set_ylabel(f"Ratio XY")
# ax.set_ylabel(f"Calculated lifetime (ns)")

# Plot a rolling average line between the data points
# rolling_mean = df["calculated_lifetime_mean"].rolling(window=1, min_periods=1, center=True).mean()
rolling_mean = df["ratio_xy_mean"].rolling(window=2, min_periods=1, center=True).mean()
ax.plot(df["trigger_intensity"], rolling_mean, color='black',)

ax.set_xlabel("Trigger intensity")


# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# get datetime
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# save the figure
fig.savefig(os.path.join(top_path, 'plots', f'{now}_OADF_cw_ratio_xy.png'), dpi=300, bbox_inches='tight')
