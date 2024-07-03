import numpy as np

x = np.load('observations.npy', allow_pickle=True)

for i, obs in enumerate(x[:5]):
    print(f"Observation {i}: {obs}")
