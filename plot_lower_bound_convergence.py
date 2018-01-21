import sys

import matplotlib.pyplot as plt

logFile = sys.argv[1]

llhs = list()
with open(logFile, 'r') as f:
	for line in f:
		if "Complete Likelihood" in line and "New Alpha" in line:
			llh = float(line.split("Likelihood:")[1][:-3])
			llhs.append(llh)

plt.plot(llhs)
plt.xlabel("iteration", fontsize = 24)
plt.ylabel("lower bound", fontsize = 24)
plt.tight_layout()
plt.show()

