# CIFAR10_Test
Some attempts based on CIFAR10 dataset.This project is a simple exercise using CIFAR10 data sets and Pytorch.
And the directory structure of the project is shown below.

```

|-- CIFAR10_Test
    |-- README.md
    |-- Data
    |   |-- cifar-10-batches-py
    |       |-- batches.meta
    |       |-- data_batch_1
    |       |-- data_batch_2
    |       |-- data_batch_3
    |       |-- data_batch_4
    |       |-- data_batch_5
    |       |-- readme.html
    |       |-- test_batch
    |-- Figure
    |   |-- loss
    |       |-- loss_2021-10-07 23_22.jpg
    |       |-- loss_2021-10-07 23_29.jpg
    |       |-- loss_2021-10-07 23_39.jpg
    |       |-- loss_2021-10-08 08_38.jpg
    |       |-- loss_2021-10-08 10_28.jpg
    |       |-- loss_2021-10-08 15_45.jpg
    |       |-- loss_2021-10-08 15_59.jpg
    |       |-- loss_2021-10-08 16_34.jpg
    |       |-- loss_2021-10-08 17_34.jpg
    |       |-- loss_2021-10-08 17_44.jpg
    |       |-- loss_2021-10-08 18_08.jpg
    |       |-- loss_2021-10-08 19_02.jpg
    |       |-- loss_2021-10-08 19_33.jpg
    |       |-- loss_2021-10-08 19_53.jpg
    |       |-- loss_2021-10-08 20_09.jpg
    |       |-- loss_2021-10-08 20_19.jpg
    |       |-- loss_2021-10-08 20_42.jpg
    |       |-- loss_2021-10-08 21_08.jpg
    |       |-- loss_2021-10-08 21_15.jpg
    |       |-- loss_2021-10-08 21_35.jpg
    |       |-- loss_2021-10-08 22_04.jpg
    |       |-- loss_2021-10-08 22_36.jpg
    |       |-- loss_2021-10-08 22_56.jpg
    |       |-- loss_2021-10-08 23_04.jpg
    |       |-- loss_2021-10-08 23_12.jpg
    |       |-- loss_2021-10-08 23_29.jpg
    |-- log
    |   |-- 2021-10-07 23_39_log.txt
    |   |-- 2021-10-07 23_46_log.txt
    |   |-- 2021-10-08 08_38_log.txt
    |   |-- 2021-10-08 10_25_log.txt
    |   |-- 2021-10-08 10_28_log.txt
    |   |-- 2021-10-08 15_45_log.txt
    |   |-- 2021-10-08 15_59_log.txt
    |   |-- 2021-10-08 16_34_log.txt
    |   |-- 2021-10-08 17_34_log.txt
    |   |-- 2021-10-08 17_44_log.txt
    |   |-- 2021-10-08 18_08_log.txt
    |   |-- 2021-10-08 19_02_log.txt
    |   |-- 2021-10-08 19_33_log.txt
    |   |-- 2021-10-08 19_53_log.txt
    |   |-- 2021-10-08 20_09_log.txt
    |   |-- 2021-10-08 20_19_log.txt
    |   |-- 2021-10-08 20_42_log.txt
    |   |-- 2021-10-08 21_04_log.txt
    |   |-- 2021-10-08 21_08_log.txt
    |   |-- 2021-10-08 21_15_log.txt
    |   |-- 2021-10-08 21_35_log.txt
    |   |-- 2021-10-08 22_04_log.txt
    |   |-- 2021-10-08 22_36_log.txt
    |   |-- 2021-10-08 22_56_log.txt
    |   |-- 2021-10-08 23_04_log.txt
    |   |-- 2021-10-08 23_12_log.txt
    |   |-- 2021-10-08 23_29_log.txt
    |-- main
    |   |-- load_datasets.py
    |   |-- main.py
    |-- saved_model
        |-- model_2021-10-07 21_58.pkl
        |-- model_2021-10-07 22_00.pkl

```

- `saved_model`:This folder is used to save the training model.
- `log`:This folder is used to record the log of the training process.
- `Figure`:This folder is used to save some figures about this project, like loss figures, etc.
- `Data`:This folder holds the data set.
- `main`:The main program is saved in this folder.
## Quick Links
 - [Data](#data)
 - [Train](#train)
   -  [Requirements](#requirements)
   -  [Load Data Sets](#load-dataset)
   -  [Network Structure](#net)
   -  [Training](#training)
 -  [Results and Findings](#results)
 -  [Summary](#summary)


## data
- The data set is CIFAR-10(Python). And You can download the dataset and get a more detailed introduction at [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html).
- To save running time, in this project, I downloaded the dataset to load locally.

## Train
### `Requirements`
I ran this program on a Nvidia 1080 graphics card, which required a GPU version of Pytorch and a number of other packages. If your CUDA version is 11.0, you can run 
 ```bash
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
 ``` to install the appropriate Pytorch package.
### `Load Data Sets`
Pytorch offers a very convenient package torchvision. Torchvison also provides dataloader to load common MNIST, CIFAR-10, ImageNet, etc data sets. The project uses torchvision to load local CIFAR-10 datasets. 
For example code, you can see file **load__datasets.py**.For training sets, data enhancements can be made using  `transforms`. You can see the predicted results after and after data augmentation in the following sections.
The code is in the Load_data() function in **main.py**.
### `Network Structure`
I used convolutional neural networks and used torch.nn to write the code.
<code>
    
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
</code>
The above is the part of my definition of convolutional network. Considering that the current network structure is relatively simple, I will gradually optimize it in the future study.

### `Training`
The cross-entropy-Loss function is used as the loss function and Adam is used as the Optimizer.Because the current network structure is relatively simple, only the learning rate of the optimizer is adjusted during training.
You can run <code>python main.py --lr 0.0001 --epoch 100</code> to start training and evaluating.
lr and epoch are optional parameters and their default values are 0.0001 and 40, respectively.
Finally, the evaluation function is used to predict and evaluate, and the Plt function is used to make the loss changes in the running process into line charts and save them in the Figure/ Loss folder.
The evaluation result is calculated by dividing the number of predicted successful images by the total number of images.
<code>

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
</code>
<code>

    def Plt(self):
        x_axis_data = [i for i in range(len(plt_loss))]
        y_axis_data = plt_loss
        
        plt.plot(x_axis_data, y_axis_data, 'ro-', color='red', alpha=0.8, linewidth=2)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        
        #plt.show()
        plt.savefig('../Figure/loss/loss_'+str(timestart1)+'.jpg')
</code>

##Results and Findings
In the process of debugging, the main learning rate, the epoch for fine tuning. First, keep the learning rate at 0.0001 and gradually increase the epoch from 30 to 200.
After that, the epoch was also increased from 30 to 100 each time the learning rate was changed.
The picture below is an example of loss changing with epoch. And each training generates a graph.The specific code is shown in the Plt function.

The table below shows the parameter values and the corresponding predicted results.

- batch_size:32

| learning rate  | epoch | Accuracy  |
| :--------:     | :----:|  :----:   |
|   0.0005       | 30    |  68.930 % |
|   0.0005       | 60    |  70.460 % |
|   0.0005       | 100   |  69.620 % |
|   0.0001       | 30    |  77.250 % |
|   0.0001       | 60    |  77.310 % |
|   0.0001       | 100   |  77.870 % |
|   0.00005      | 30    |  75.790 % |
|   0.00005      | 60    |  75.880 % |
|   0.00005      | 100   |  75.870 % |

- batch_size:64

| learning rate  | epoch | Accuracy  |
| :--------:     | :----:|  :----:   |
|   0.0005       | 30    |  73.650 % |
|   0.0005       | 60    |  73.610 % |
|   0.0005       | 100   |  73.230 % |
|   0.0001       | 30    |  76.330 % |
|   0.0001       | 60    |  76.620 % |
|   0.0001       | 100   |  76.600 % |
|   0.00005      | 30    |  75.350 % |
|   0.00005      | 60    |  75.550 % |
|   0.00005      | 100   |  75.510 % |

- From the point of batch_size, the larger the batch__size, the less time each epoch takes. When batch_size is 32, it takes about 18-19 seconds to process a epoch; when batch_size is 64, it takes about 12-13 seconds to process a epoch. 
But it is not that the larger the batch__size, the higher the accuracy.I have tried batch__size equal to 128, but the result is not very good.
- As can be seen from the above two tables, when the learning rate is *0.0001*, the prediction accuracy is the highest.
If the learning rate is too large or too small, it will affect the final accuracy.So it is very important to choose a suitable learning rate.
- For the epoch, according to the experimental results, obviously it cannot be considered that the larger the epoch, the higher the accuracy.
In my experiment, I found that when the epoch exceeded a value, it did not have a great influence on the accuracy. 
- According to the currently known results of CIFAR10, the accuracy can reach up to 96%. Since my network structure is relatively simple, after using the Adam optimizer and tuning, the best experimental result is only 77.870%.
In subsequent experiments, I will spend more time tuning parameters and adjusting the network structure.
## Summary
This project is an exercise based on the Pytorch framework CIFAR10 dataset. 
In the process of implementation, I simply defined the depth of the network and the size of the convolution kernel, but found that the effect was not very good in the process of parameter tuning. After that, I will re-debug the code reference to [VGG16's](https://arxiv.org/abs/1409.1556) network design.

