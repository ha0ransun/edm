import torch
import torchvision
import torchvision.transforms as T
import numpy as np

device = torch.device("cpu")
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
x = torch.tensor(dataset.data).to(device)

x = (x.reshape(x.shape[0], 3072) / 127.5) - 1
print(x.shape)
batch = x[torch.randint(x.shape[0], (64,))]
print(batch.shape)

s_min = 0.002
s_max = 80
n = 18
rho = 1 / 7
sigma = (s_max ** rho + torch.arange(n) / (n - 1) * (s_min ** rho - s_max ** rho)) ** (1 / rho)
batch_size = 64
print(sigma)

for i in range(n - 1):
    all_errors = []
    for j in range(10):
        y = x[torch.randint(x.shape[0], (batch_size,))]
        yn = y + torch.randn_like(y) * sigma[i]
        c_dist_hat = torch.cdist(yn, y)
        logp_hat = - 2 * c_dist_hat / (sigma[i] + sigma[i+1]) ** 2
        grad_hat = (torch.softmax(logp_hat, dim=1).unsqueeze(1) @ (yn.unsqueeze(1) - y.unsqueeze(0))).squeeze(1) / sigma[i]
        c_dist = torch.cdist(yn, x)
        logp = - 2 * c_dist / (sigma[i] + sigma[i+1]) ** 2
        grad = torch.zeros_like(grad_hat)
        for k in range(batch_size):
            grad[k] += torch.softmax(logp[k], dim=0) @ (yn[k] - x) / sigma[i]
        error = (((grad - grad_hat) ** 2).sum(-1) / (grad ** 2).sum(-1)).mean()
        all_errors.append(error.item())
    all_errors = np.array(all_errors)
    print(f"sigma {i}: relative error is {all_errors.mean():.4f} +/- {all_errors.std():.4f}")