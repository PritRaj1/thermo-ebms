import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import optax

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

# Set plot styling
sns.set(font_scale=1.75)
sns.set_style(
	"whitegrid",
	rc={"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]},
)

begin = (
	int(parser["LR_SCHEDULE"]["BEGIN_EPOCH"])
	* int(parser["PIPELINE"]["NUM_TRAIN_DATA"])
	/ int(parser["PIPELINE"]["BATCH_SIZE"])
)
LR_0 = float(parser["OPTIMIZER"]["G_INITIAL_LR"])
LR_1 = float(parser["OPTIMIZER"]["G_FINAL_LR"])
GAMMA = float(parser["LR_SCHEDULE"]["DECAY_RATE"])
STEP = (
	int(parser["LR_SCHEDULE"]["STEP_INTERVAL"])
	* int(parser["PIPELINE"]["NUM_TRAIN_DATA"])
	/ int(parser["PIPELINE"]["BATCH_SIZE"])
)

LR_schedule = optax.exponential_decay(
	init_value=LR_0,
	transition_steps=STEP,
	decay_rate=GAMMA,
	transition_begin=begin,
	end_value=LR_1,
)

epochs = np.linspace(0, int(parser["PIPELINE"]["NUM_EPOCHS"]))
steps = (
	epochs
	* int(parser["PIPELINE"]["NUM_TRAIN_DATA"])
	/ int(parser["PIPELINE"]["BATCH_SIZE"])
)

plt.figure()
plt.plot(epochs, [LR_schedule(step) for step in steps])
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Parameter update")
plt.savefig("results/lr_schedule.png")

p_list = [0, 0.3, 0.1, 1, 2, 3, 6, 10]
temp_cmap = plt.get_cmap("coolwarm")
temp_colors = [temp_cmap(i) for i in np.linspace(0, 1, len(p_list))]
num_temps = 100

plt.figure(figsize=(10, 6))
for p in p_list:
	temp = np.linspace(0, 1, num_temps) ** p
	label = "p = {}".format(p)
	plt.plot(temp, label=label, color=temp_colors[p_list.index(p)])
plt.xlabel("Schedule/Summation Index, $i$")
plt.ylabel("Temperature, $t_i$")
plt.title("Temperature Schedule")
plt.legend(loc="upper left")
plt.savefig("results/temperature_schedule.png")

reduced_p = [0.3, 1, 3]
reduced_temps = [temp_colors[1], temp_colors[3], temp_colors[5]]
num_temps = 30


def y_function(x):
	return (x - x**2) * np.exp(x)


t = np.linspace(0, 1, 100)

# Plot discretsised integral
for p, color in zip(reduced_p, reduced_temps):
	plt.figure(figsize=(10, 6))
	plt.plot(t, y_function(t), label=r"Arbitrary Integrand, $E_i$", color="maroon")
	plt.fill_between(t, y_function(t), alpha=0.2, color=color, label=r"Area")
	temps = np.linspace(0, 1, num_temps) ** p
	for i in range(num_temps):
		if i == 0:
			label = "Discretisation: p = {}".format(p)
		else:
			label = None
		plt.vlines(temps[i], 0, y_function(temps[i]), color="black", label=label)
	plt.xlabel(r"Temperature $t_i$")
	plt.ylabel(r"Arbitrary Integrand, $E_i$")
	plt.title("Discretised Integral")
	plt.legend(loc="upper left")
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.tight_layout()
	plt.savefig("results/temperature_schedule_{}.png".format(p))
