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

scale = np.log((batch_size - 1) / (x.shape[0] - 1))
for i in range(n - 1):
    all_errors_1 = []
    all_errors_2 = []
    all_probs = []
    all_probs_hat = []
    for j in range(2):
        idx = torch.randint(x.shape[0], (batch_size,))
        y = x[idx]
        z = torch.randn_like(y)
        yn = y + z * sigma[i]
        c_dist_hat = torch.cdist(yn, y) ** 2
        logp_hat = - 2 * c_dist_hat / (sigma[i] + sigma[i+1]) ** 2
        logp_hat += torch.diag(torch.ones_like(torch.diag(logp_hat)) * scale)
        probs_hat = torch.diag(torch.softmax(logp_hat, dim=1))
        grad_hat = (torch.softmax(logp_hat, dim=1).unsqueeze(1) @ (yn.unsqueeze(1) - y.unsqueeze(0))).squeeze(1) / sigma[i]
        c_dist = torch.cdist(yn, x) ** 2
        logp = - 2 * c_dist / (sigma[i] + sigma[i+1]) ** 2
        prob = torch.softmax(logp, dim=-1)
        grad = torch.zeros_like(grad_hat)
        probs = []
        for k in range(batch_size):
            probs.append(prob[k, idx[k]].item())
            grad[k] += prob[k] @ (yn[k] - x) / sigma[i]
        error_1 = (((grad - grad_hat) ** 2).sum(-1) / (grad ** 2).sum(-1)).mean()
        all_errors_1.append(error_1.item())
        error_2 = (((grad - z) ** 2).sum(-1) / (grad ** 2).sum(-1)).mean()
        all_errors_2.append(error_2.item())
        all_probs.append(np.array(probs))
        all_probs_hat.append(probs_hat.numpy())
    all_errors_1 = np.array(all_errors_1)
    all_errors_2 = np.array(all_errors_2)
    all_probs = np.concatenate(all_probs, axis=0)
    all_probs_hat = np.concatenate(all_probs_hat, axis=0)
    print(f"### sigma {i + 1} = {sigma[i]:.8f} ###")
    print(f"error1 is {all_errors_1.mean():.8f} +/- {all_errors_1.std():.8f}")
    print(f"error2 is {all_errors_2.mean():.8f} +/- {all_errors_2.std():.8f}")
    print(f"all_probs is {all_probs.mean():.8f} +/- {all_probs.std():.8f}")
    print(f"all_probs_hat is {all_probs_hat.mean():.8f} +/- {all_probs_hat.std():.8f}")

    