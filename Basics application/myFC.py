import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# set device for sending data to this GPU for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNetwork(nn.Module):
    """
    Define a model class for training
    Neural Network contains three fc or linear layer and use relu as activation function
    """

    def __init__(self, input_size, hidden_size, num_classes):
        """super neutal network from self
        Neural Network contains two parts: init and forward
        Init for initialize each layer used in forward
        Forward part define the neural network structure
        """
        super(NeuralNetwork, self).__init__()
        # Fully connected layer with para: input size and output size
        # Since it is dense layer, the input size must be confirmed

        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Fully connected layer with para: input size and output size
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        # Activation function
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out

# Instantiation model
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
# define loss function
criterion = nn.CrossEntropyLoss()
# define optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# Start training our model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Fully connected layer based network, we need to reshape image into 2 dims
        # Send images to GPU
        images = images.reshape(-1, 28*28).to(device)
        # Send labels to GPU
        labels = labels.to(device)
        # Feed input into model
        outputs = model(images)
        # Compute epoch loss
        loss = criterion(outputs, labels)
        # Define optimization solution
        optimizer.zero_grad()
        # update model parameter
        # we use loss to backward rather than others
        loss.backward()
        # Then we use optimizer to optimize model parameters
        # Note that this is a single time optimization
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# without updating the model parameters
# close the grad computing process
with torch.no_grad():
    correct = 0
    total = 0
    # Evaluating on the test data
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint ckpt
torch.save(model.state_dict(), 'model.ckpt')