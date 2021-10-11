import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import time
import datetime
import os
import argparse
import pylab as pl
import matplotlib.pyplot as plt
import argparse
import logging


global trainloader
global testloader
plt_loss=[]
now = datetime.datetime.now()
timestart1 = now.strftime("%Y-%m-%d %H:%M")

parser = argparse.ArgumentParser(description='None')
parser.add_argument('--epoch', dest='epoch', type=int, default=40, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# images in batch')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("../log/"+str(timestart1)+"_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def Load_dataset(trans: bool = False):
    '''
    return train dataset loader and test dataset loader.
    if trans==True,train datasets will use data augmentation.
    '''
    if(trans==True):
        print('Data augmentation.')
        transform = transforms.Compose(
            [
             transforms.RandomHorizontalFlip(),
             transforms.RandomGrayscale(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform1 = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #load train set
    trainset = data.CIFAR10(root='../Data', train=True,
                                            download=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=16)
    #load test set
    testset = data.CIFAR10(root='../Data', train=False,
                                           download=False, transform=transform1)
    test_loader = DataLoader(testset, args.batch_size,
                                             shuffle=False, num_workers=16)
                                             
    return train_loader, test_loader
    
class mycovNet(nn.Module):
    
    def __init__(self):
        super(mycovNet,self).__init__()
        self.conv1 = nn.Conv2d(3,32,6,padding=1)
        self.conv2 = nn.Conv2d(32,64,6,padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64,128,6,padding=1)
        self.conv4 = nn.Conv2d(128,256,6,padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(256*3*3,1024)
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(1024,512)
        self.drop2 = nn.Dropout2d()
        self.fc3 = nn.Linear(512,10)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.relu1(x)


        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.relu2(x)
        #print("=====================================================================================")
        #print(" x shape ",x.size())
        x = x.view(-1,256*3*3)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

    def train(self,device):
        #Save loss(log)
        optimizer = optim.Adam(self.parameters(), lr=args.lr)
        print('lr:',args.lr)
        print('epoch:',args.epoch)
        logger.info("lr={}".format(args.lr))
        logger.info("epoch={}".format(args.epoch))
        path = '../saved_model/model_'+str(timestart1)+'.pkl'
        #path='../saved_model/model_2021-10-07 21:58.pkl'
        initepoch = 0

        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()

        else:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss = checkpoint['loss']

        for epoch in range(initepoch,args.epoch):   
            t_loss=0.0
            running_loss = 0.0
            total = 0
            correct = 0
            timestart=time.time()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device),labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()
                
                running_loss += l.item()
                
                if i % 500 == 499:
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i+1, running_loss / 500))
                    torch.save({'epoch':epoch,
                                'model_state_dict':net.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':loss,
                                'run_loss':running_loss
                                },path)
                    t_loss=running_loss/500
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d images: %.3f %%' % (total,
                            100.0 * correct / total))
                    logger.info('Accuracy of the network on the %d images: %.3f %%' % (total,
                            100.0 * correct / total))
                    total = 0
                    correct = 0
            print('===================================epoch %d cost %3f sec=================================' %(epoch,time.time()-timestart))
            logger.info('===================================epoch %d cost %3f sec=================================' %(epoch,time.time()-timestart))
            plt_loss.append(t_loss)
        print('Finished Training')
        
    def evaluation(self,device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                #Index the maximum value per row
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))
        logger.info('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))
    def Plt(self):
        x_axis_data = [i for i in range(len(plt_loss))]
        y_axis_data = plt_loss
        
        fig = plt.figure(figsize = (10,10)) 
        ax1 = fig.add_subplot(1, 1, 1) 
        plt.plot(x_axis_data, y_axis_data,'g-',label=u'CIFAR10_Test_Loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('../Figure/loss/loss_'+str(timestart1)+'.jpg')
        
if __name__=='__main__':
    
    t=True
    trainloader,testloader=Load_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = mycovNet()
    net = net.to(device)
    net.train(device)
    net.evaluation(device)
    net.Plt()
    
   
    print(args.epoch)
    print('=======================================Done!=====================================')