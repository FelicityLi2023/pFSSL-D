import matplotlib.pyplot as plt
import torch

sim_values = torch.linspace(0, 1, 100)
lambda_values = 0.5 + 0.5 * (1 - sim_values)

plt.plot(sim_values.numpy(), lambda_values.numpy())
plt.xlabel('Cosine Similarity Normalized')
plt.ylabel('Lambda Value')
plt.title('Lambda Value vs. Cosine Similarity Normalized')
plt.show()
