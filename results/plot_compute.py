import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.cm as cm

# Set plot styling
sns.set(font_scale=2)
sns.set_style(
	"whitegrid",
	rc={"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]},
)

Temps = ["Vanilla Model", r"$N_t=10$", r"$N_t=20$", r"$N_t=30$"]
FLOPS = [159780470784.0, 266255302656.0, 266255302656.0, 266255302656.0]
TIMES = [3205.51, 28261.13, 53971.03, 79509.17]

# Viridis[2], Coolwarm[0], Magma[1], Magma[3]
custom_colors = [
	cm.summer(2 / 256),
	cm.coolwarm(0 / 256),
	cm.magma(150 / 256),
	cm.magma(80 / 256),
]

# create palette

# Plot bar chart with sns
plt.figure(figsize=(15, 11))
ax = sns.barplot(x=Temps, y=FLOPS, palette=custom_colors)
ax.set_ylabel(r"FLOPS (flop/s)")
# ax.set_yscale("log")

# Add text labels to the bars to indicate the value
for i in range(len(FLOPS)):
	ax.text(i, FLOPS[i], "{:.2e}".format(FLOPS[i]), ha="center", va="bottom")

plt.xticks(rotation=0)
plt.title(r"FLOPS to Evaluate $-\log(p_\theta(\mathbf{x}))$")
plt.tight_layout()
plt.savefig("results/flops.png")


plt.figure(figsize=(15, 10))
ax = sns.barplot(x=Temps, y=TIMES, palette=custom_colors)
ax.set_ylabel(r"Time (s)")
# ax.set_yscale("log")

# Add text labels to the bars to indicate the value,
for i in range(len(TIMES)):
	bar_value = TIMES[i]
	ax.text(
		i, TIMES[i], "{:.2f} hrs".format(bar_value / 3600), ha="center", va="bottom"
	)

plt.xticks(rotation=0)
plt.title(r"Time Taken to Train for 50 Epochs")
plt.ylim(0, 90000)
plt.tight_layout()
plt.savefig("results/times.png")
