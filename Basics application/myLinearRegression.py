import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
# First we need to obtain the training data
# one dim data processing np.shape == (15, 1)
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
# one dim label processing np.shape == (15, 1)
y_train = np.array([[1.17], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Then we need to set up the linear regression model
# linear regression only contains two parameters : input size and output size
model = nn.Linear(input_size, output_size)
# criterion is the loss function
criterion = nn.MSELoss()
# optimizer is the way we compute gradient
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
# for in epochs
#  if the input data is ndarray formate we need to transfer it into tensor through
# torch.from_numpy
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # Forward pass
    # pass the prepared data into our model and save the ourput result
    outputs = model(inputs)
    # define our loss function
    loss = criterion(outputs, targets)
    # Backward and optimizer definition
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.savefig('result.png')
plt.show()


# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')