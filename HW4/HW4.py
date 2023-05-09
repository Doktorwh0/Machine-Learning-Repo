"""
Kyle Herbruger (Visit my GitHub! https://github.com/Doktorwh0 )
EE 399 HW04
5/8/2023
This program trains a neural network on a simple class data set,
and then another on the MNIST digit data set. This second neural
network is then compared against other machine learning models. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt


plt.close('all')
""" ------------------------------ HW4 (1.i) ------------------------------- """
print("------------------------------ Part 1.i ")
"""Place Holders."""

# Define the neural network architecture


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 150)  # Tune these sizes. 1->10->5->1
        self.fc2 = nn.Linear(150, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Create an instance of the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Generate the data
X_data = np.arange(0, 31)
Y_data = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
                   40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
# X = torch.from_numpy(X_data.reshape(-1, 1)).float()
# Y = torch.from_numpy(Y_data.reshape(-1, 1)).float()

# # Train the neural network
# losses = []
# for epoch in range(1000):
#     optimizer.zero_grad()
#     output = net(X)
#     loss = criterion(output, Y)
#     loss.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: loss={loss.item()}")
#     losses.append(loss.item())

""" ------------------------------ HW4 (1.ii) ----------------------------- """
print("------------------------------ Part 1.ii ")

# # Part 1.ii: Evaluate on training and test data
# train_X = torch.from_numpy(X_data[:20].reshape(-1, 1)).float()
# train_Y = torch.from_numpy(Y_data[:20].reshape(-1, 1)).float()
# test_X = torch.from_numpy(X_data[20:].reshape(-1, 1)).float()
# test_Y = torch.from_numpy(Y_data[20:].reshape(-1, 1)).float()

# # Train the neural network
# losses = []
# for epoch in range(1000):
#     optimizer.zero_grad()
#     output = net(train_X)
#     loss = criterion(output, train_Y)
#     loss.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: loss={loss.item()}")
#     losses.append(loss.item())


# train_output = net(train_X)
# train_loss_ii = ((train_output - train_Y)**2).mean(dim=None).item()
# print(f"Training error: {train_loss_ii}")

# test_output = net(test_X)
# test_loss_ii = ((test_output - test_Y)**2).mean(dim=None).item()
# print(f"Test error: {test_loss_ii}")

# # Plot the loss vs epoch
# plt.figure(1)
# plt.plot(range(1000), loss_norm)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()


""" ----------------------------- HW4 (1.iii) ----------------------------- """
print("------------------------------ Part 1.iii ")
# Part 1.iii - using the first 10 and last 10 data points as training data

# split data into training and test sets
X_train = torch.cat((torch.from_numpy(X_data[:10]), torch.from_numpy(X_data[20:])), dim=0).reshape(-1, 1)
Y_train = torch.cat((torch.from_numpy(Y_data[:10]), torch.from_numpy(Y_data[20:])), dim=0).reshape(-1, 1)
X_test = torch.from_numpy(X_data[10:20].reshape(-1, 1))
Y_test = torch.from_numpy(Y_data[10:20].reshape(-1, 1))

"""Model 2"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 2500)  # Tune these sizes. 1->10->5->1
        self.fc2 = nn.Linear(2500, 200)
        self.fc3 = nn.Linear(200, 2500)
        self.fc4 = nn.Linear(2500, 5000)
        self.fc5 = nn.Linear(5000, 2500)
        self.fc6 = nn.Linear(2500, 250)
        self.fc7 = nn.Linear(250, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# Create an instance of the neural network
net = Net()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# train the model
num_epochs = 2000
best_loss = float('inf')
for epoch in range(num_epochs):

    # # randomly reverse the training data set
    # if random.random() > 0.5:
    #     X_train = torch.flip(X_train, dims=[0])
    #     Y_train = torch.flip(Y_train, dims=[0])

    # # shuffle the training data set
    # permutation = torch.randperm(X_train.size()[0])
    # X_train = X_train[permutation]
    # Y_train = Y_train[permutation]

    # forward pass
    outputs = net(X_train.float())
    loss = criterion(outputs, Y_train.float())

    # backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save the best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model = net.state_dict()
    # print loss at every 50 epochs
    if (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# evaluate the best model on test data
net.load_state_dict(best_model)
with torch.no_grad():
    outputs = net(X_test.float())
    test_loss = criterion(outputs, Y_test.float())
    print('Test Loss: {:.4f}'.format(test_loss.item()))

# compare with Part 1.ii
train_loss_iii = loss.item()
test_loss_iii = test_loss.item()
#print('Part 1.ii - Train Loss: {:.4f}, Test Loss: {:.4f}'.format(train_loss_ii, test_loss_ii))
#print('Part 1.iii - Train Loss: {:.4f}, Test Loss: {:.4f}'.format(train_loss_iii, test_loss_iii))


# evaluate the best model on test data
net.load_state_dict(best_model)


# compute the model's predicted values for the training and test data
# %%
X_train = torch.cat((torch.from_numpy(X_data[:10]), torch.from_numpy(X_data[20:])), dim=0).reshape(-1, 1)
Y_train = torch.cat((torch.from_numpy(Y_data[:10]), torch.from_numpy(Y_data[20:])), dim=0).reshape(-1, 1)
X_test = torch.from_numpy(X_data[10:20].reshape(-1, 1))
Y_test = torch.from_numpy(Y_data[10:20].reshape(-1, 1))
plt.close('all')

# plot the data points and the fit function
plt.figure(2)
with torch.no_grad():
    y_train_pred = net(X_train.float())
    y_test_pred = net(X_test.float())

# plot the predicted vs actual values for the training and test data
plt.subplot(3, 1, 1)
plt.title('Test Loss: {:.4f}'.format(test_loss.item()))
plt.plot(X_train.numpy(), Y_train.numpy(), 'ro', label='Training data')
plt.plot(X_train.numpy(), y_train_pred.numpy(), 'g-', label='Predicted values')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(X_test.numpy(), Y_test.numpy(), 'bo', label='Test data')
plt.plot(X_test.numpy(), y_test_pred.numpy(), 'g-', label='Predicted values')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(X_train.numpy(), Y_train.numpy(), 'ro', label='Training data')
plt.plot(X_test.numpy(), Y_test.numpy(), 'bo', label='Test data')
plt.plot(X_test.numpy(), y_test_pred.numpy(), 'go', label='Fit function')
plt.plot(X_train.numpy(), y_train_pred.numpy(), 'go')
plt.legend()

plt.show()
