import os
import datetime

import pandas as pd                 # for opening csv files into a nice format for plotting
import matplotlib.pyplot as plt     # for the actual plotting

## User variables:
CSV_NAME = None                 # default None,     specify a csv file name here to use instead of the latest
ROLLING_AVERAGE_WINDOW = 1      # default 1,        increase for smoother rolling average, 1 is just connecting points
PLOT_DPI = 300                  # default 300,      increase for higher resolution
SHOW_PLOT = True                # default True,     set to False if you want to save the plot without showing it

## Load data:
path_to_this_script = os.path.abspath(__file__)
figure_dir = os.path.dirname(path_to_this_script)
csv_dir = os.path.join(figure_dir, 'data')
# find the latest csv file in the data directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
csv_files.sort()
CSV_NAME = CSV_NAME or csv_files[-1]
csv_path = os.path.join(figure_dir, 'data', CSV_NAME)
df = pd.read_csv(csv_path)

## Group data by rdt and crescent intensity:
grouped = df.groupby(["sample_rdt", "crescent_intensity"])

# find number of unique rotational diffusion times and crescent intensities
num_unique_rdt = len(df["sample_rdt"].unique())
num_unique_crescent = len(df["crescent_intensity"].unique())

group_names = {7249.0: '40nm', 24465.0: '60nm', 113263.0: '100nm', 906106.0: '200nm'}

## Plot the data:
# create a plot, using subplot here to keep it similar to other code
fig, axs = plt.subplots(nrows=num_unique_rdt, ncols=1, sharex=True, figsize=(6, 14))

# set the color cycle for unique sample values
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, (name, group) in enumerate(grouped):
    rotational_diffusion_time_us, crescent_intensity = name
    ax = axs[df["sample_rdt"].unique().tolist().index(rotational_diffusion_time_us)]

    # plot each group on the same subplot as a scatter plot with y error bars
    ax.errorbar(group["collection_time_point"], group["ratio_xy_mean"], yerr=group["ratio_xy_std"],
                label=f"{crescent_intensity} crescent intensity", fmt='o', color=color_cycle[i % num_unique_crescent],
                alpha=0.2)
    ax.set_ylabel(f"XY Ratio, bead size: {group_names[rotational_diffusion_time_us]}")
    ax.legend(loc='lower right')

    # plot a rolling average line between the data points
    rolling_mean = group["ratio_xy_mean"].rolling(window=ROLLING_AVERAGE_WINDOW, min_periods=1, center=True).mean()
    ax.plot(group["collection_time_point"], rolling_mean, color=color_cycle[i % num_unique_rdt])

axs[-1].set_xlabel("Collection time point")

# get datetime
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# save the figure
csv_dir = os.path.join(figure_dir, 'plots')
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
fig.savefig(os.path.join(csv_dir, f'{now}-crescent.png'), dpi=PLOT_DPI, bbox_inches='tight')

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
