import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

try:
    from .models import MLP
    from .modules.topk_pruning import *
except :
    from models import MLP
    from modules.topk_pruning import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
Baseline = False
Prune = False 
Retrain = True
prune_ratio = 0.9
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


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

    if Baseline: 
        print("Baselie Training ----------------")
        train(model, train_loader, num_epochs, device, optimizer, criterion)
        test(model, test_loader, device)
        # Save the model checkpoint
        torch.save(model.state_dict(), 'model_base.ckpt')
        exit()
    if Prune: 
        print("Model Pruning -------------------")
        model.load_state_dict(torch.load('model_base.ckpt'))
        model_prune(model)
        print('\nTesting After Prune: ')
        test(model, test_loader, device)
        torch.save(model.state_dict(), 'model_pruned.ckpt')
        exit()

    if Retrain:
        print("Model Retrain -------------------")
        model.load_state_dict(torch.load('model_pruned.ckpt'))
        model_finetune(model, train_loader, num_epochs, device, optimizer, criterion)
        test(model, test_loader, device)
        torch.save(model.state_dict(), 'model_finetuned.ckpt')
        exit()


def model_prune(model):
    # -------------------------------------------------------------
    #pruning 
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.data.numel()
    weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            size = m.weight.data.numel()
            weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
            index += size

    y, i = torch.sort(weights)
    thre_index = int(total * prune_ratio)
    thre = y[thre_index]
    pruned = 0
    print('Pruning threshold: {}'.format(thre))
    zero_flag = False
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.Linear):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.numel() - torch.sum(mask)
            m.weight.data.mul_(mask)
            if int(torch.sum(mask)) == 0:
                zero_flag = True
            print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                format(k, mask.numel(), int(torch.sum(mask))))
    print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))


    
    return model


# Train the model
def model_finetune(model, train_loader, num_epochs, device, optimizer, criterion):
    
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

            for k, m in enumerate(model.modules()):
                # print(k, m)
                if isinstance(m, nn.Linear):
                    if m.weight.grad is not None:
                        # print(k, m)
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(0).float().cuda()
                        m.weight.grad.data.mul_(mask)
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



if __name__ == "__main__":
    main()