import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

# device configurs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Hyper-parameters
input_size =784 # 28 x 28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,),(0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root = './data',train= True, transform=transform, download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset= train_dataset,batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=False)

example = iter(test_loader)
example_data, example_targets=example.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0],cmap='gray')
plt.show()

# fully connected neural nerword with ane hidden layer
class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1=nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        # no activation and no softmax at the end
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100,1,28,28]
        # resized: [100, 784]
        images = images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        # Forwars pass
        outputs = model(images)
        loss=criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print(f'epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
# test the model
# in test phse, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        output=model(images)
        # max returns (value, index)
        _, predicted = torch.max(output.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item()
    acc=100.0 * n_correct / n_samples
    print(f'accuracy of the netword on the 1000 test images: {acc} %')

torch.save(model.state_dict(),'mnist_ffn.pth')