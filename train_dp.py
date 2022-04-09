"""
Mostly based on the official pytorch tutorial
Link: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
Modified for educational purposes.
Nikolas, AI Summer
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision

def create_data_loader_cifar10():
    transform = transforms.Compose(
        [
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256*4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=20, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=20)
    return trainloader, testloader



def train(net, trainloader):
    print("Start training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 1
    num_of_batches = len(trainloader)
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data

            labels = labels.cuda() 

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        print(f'[Epoch {epoch + 1}/{epochs}] loss: {running_loss / num_of_batches:.3f}')
    
    print('Finished Training')


def test(net, PATH, testloader):
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            labels = labels.cuda() 
            
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct // total
    print(f'Accuracy of the network on the 10000 test images: {acc} %')


if __name__ == '__main__':
    start = time.time()
    

    
    PATH = './cifar_net.pth'
    trainloader, testloader = create_data_loader_cifar10()

    net = torchvision.models.resnet50(False)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # Batch size should be divisible by number of GPUs
        net = nn.DataParallel(net)

    net.cuda()
    
    start_train = time.time()
    train(net, trainloader)
    end_train = time.time()
    # save
    torch.save(net.state_dict(), PATH)
    # test
    test(net, PATH, testloader)

    end = time.time()
    seconds = (end - start)
    seconds_train = (end_train - start_train)
    print(f"Total elapsed time: {seconds:.2f} seconds, \
     Train 1 epoch {seconds_train:.2f} seconds")



