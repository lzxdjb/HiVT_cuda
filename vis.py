import torch
import numpy as np
import matplotlib.pyplot as plt

for i in range(20):
    f = torch.load(f"cnts/count{i}.pt")
    f = f.cpu().numpy()
    # f = np.sort(f)
    plt.plot(f)
    plt.show()
