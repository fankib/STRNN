import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def plot(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

plot(0.10158, 0.014129779427389)
plot(0.08084, 0.011239236826602)
plot(0.09407, 0.01167409763346)
plot(0.10127, 0.013852560934515)

plt.legend(['Default', 'Tanh', 'No RNN', 'Single Sample'])
plt.show()
