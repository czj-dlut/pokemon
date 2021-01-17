from load import *
from model import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

classes = ('bulbasaur','charmander','mewtwo','pikachu','squirtle')

def trainandsave():
    trainloader = train_dataset_loader
    net = inception_v4(5)
    for epoch in range(5):
        if epoch<=1:   
            optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
        elif epoch<=3:
            optimizer = optim.SGD(net.parameters(), lr=0.0002, momentum=0.9)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)    
        criterion = nn.CrossEntropyLoss()                    
        running_loss = 0.0  
        for i, data in enumerate(trainloader, 0):  
            
            inputs, labels = data  

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels) 
            inputs = inputs.cuda()
            labels = labels.cuda()
            net = net.cuda()            
            optimizer.zero_grad()        

            outputs = net(inputs)        
            loss = criterion(outputs, labels)  
            loss.backward()                    
            optimizer.step()                 
            running_loss += loss.data      
            if i % 10 == 9:                 
                print('[%d, %5d] loss: %.3f' %
                       (epoch + 1, i + 1, running_loss / 10)) 
                running_loss = 0.0  
    print('Finished Training')
    t.save(net,'net.pkl')
    
def reload_net():
    trainednet = t.load('net.pkl')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    testloader = test_dataset_loader
    net = reload_net()
    dataiter = iter(testloader)  
    images, labels = dataiter.next()              
    imshow(torchvision.utils.make_grid(images,nrow=5))    
    print('GroundTruth:', " ".join('%10s' % classes[labels[j]]    for j in range(25)))  
    images = images.cuda()
    labels = labels.cuda()
    outputs = net(Variable(images))  
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted  :', " ".join('%10s' % classes[predicted[j]] for j in range(25)))  


if __name__ == "__main__":
    test()