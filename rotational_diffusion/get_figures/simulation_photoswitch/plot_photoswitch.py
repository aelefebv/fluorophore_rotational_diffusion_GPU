import os
import datetime

import pandas as pd  # for opening csv files into a nice format for plotting
import matplotlib.pyplot as plt  # for the actual plotting

## User variables:
# CSV_NAME = '20230524_162759_photoswitch.csv'
ROLLING_AVERAGE_WINDOW = 1
PLOT_DPI = 300
SHOW_PLOT = True

## Load data:
path_to_this_script = os.path.abspath(__file__)
figure_dir = os.path.dirname(path_to_this_script)
csv_dir = os.path.join(figure_dir, 'data')
# find the latest csv file in the data directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
csv_files.sort()
CSV_NAME = csv_files[-1]
csv_path = os.path.join(figure_dir, 'data', CSV_NAME)
df = pd.read_csv(csv_path)

## Group data by sample:
atpase = {'ab': 250, 'ab-m': 260, 'ab-agg1': 417, 'ab-sol': 666, 'ab-proto': 2000}
group_names = list(atpase.keys())
grouped = df.groupby("sample_rdt_unpied")

## Plot the data:
# create a plot, using subplot here to keep it similar to other code
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(7, 6))
# set the color cycle for unique sample values
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# find number of unique rotational diffusion times for color cycling
num_unique_rdt = len(df["sample_rdt_unpied"].unique())

for i, (name, group) in enumerate(grouped):
    # plot each group on the same subplot as a scatter plot with y error bars
    ax.errorbar(x=group["off_intensity"], y=group["ratio_xy_mean"], yerr=group["ratio_xy_std"],
                label=f"{group_names[i]}", fmt='o', color=color_cycle[i % num_unique_rdt],
                alpha=0.2)

    # plot a rolling average line between the data points
    rolling_mean = group["ratio_xy_mean"].rolling(window=ROLLING_AVERAGE_WINDOW, min_periods=1, center=True).mean()
    ax.plot(group["off_intensity"], rolling_mean, color=color_cycle[i % num_unique_rdt])

ax.set_xlabel("off_intensity")
ax.set_ylabel(f"XY Ratio")
ax.legend()

# put legend on bottom right of plot
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.1), ncol=1, fancybox=True, shadow=False)

# get datetime
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# save the figure
csv_dir = os.path.join(figure_dir, 'plots')
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)
fig.savefig(os.path.join(csv_dir, f'{now}-photoswitch.png'), dpi=PLOT_DPI, bbox_inches='tight')

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
