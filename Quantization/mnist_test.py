import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

try:
    from .models import MLP
    from .modules.qlayers import *
except :
    from models import MLP
    from modules.qlayers import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
Quant = True 
Retrain = False
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

if Quant : 
    nn.Linear = QLinear 
    nn.Conv2d = QConv2d
    nn.ReLU = QReLu



# Train the model
def train(model, train_loader, num_epochs, device, optimizer, criterion):
    model.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
def test(model, test_loader, device):

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))



def main():


    # MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='/Datadisk/jmlu/Pytorch_Data/', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=False)

    test_dataset = torchvision.datasets.MNIST(root='/Datadisk/jmlu/Pytorch_Data/', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)


    model = MLP(input_size, hidden_size, num_classes).to(device)
    print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    if Quant :
        model.load_state_dict(torch.load("model.ckpt"))
        test(model, test_loader, device)
        if Retrain:
            train()
            test()
            # Save the model checkpoint
            torch.save(model.state_dict(), 'model.ckpt')  
        exit()

    train()
    test()
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
if __name__ == "__main__":
    main()