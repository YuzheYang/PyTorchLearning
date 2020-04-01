import torch
# Fully connected model via auto-grad

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# Create random Tensors for weights; setting requires_grad=True means that we
# want to compute gradients for these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
loss_his = []
for t in range(500):

  y_pred = x.mm(w1).clamp(min=0).mm(w2)

  loss = (y_pred - y).pow(2).sum()
  loss_his.append(loss.item())
  print(t, loss.item())

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Tensors with requires_grad=True.
  # After this call w1.grad and w2.grad will be Tensors holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()
  # Update weights using gradient descent.
  with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad
    # clear the weight parameters manually.
    w1.grad.zero_()
    w2.grad.zero_()
from matplotlib import pyplot as plt
plt.figure()
import numpy as np
loss_his = np.array(loss_his)
plt.plot(loss_his)
plt.waitforbuttonpress()