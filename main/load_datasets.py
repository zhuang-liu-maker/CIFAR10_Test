from torchvision import transforms, datasets as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

#load train set
train_set = data.CIFAR10(root='../Data', train=False, transform=transform, target_transform=None, download=False)
data_loader = DataLoader(dataset=train_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=2)

to_pil_image = transforms.ToPILImage()
cnt = 0
for image,label in data_loader:
    if cnt>=1:
        break
    print(label)
    img = to_pil_image(image[0])
    img.show()
    plt.imshow(img)
    plt.show()

    cnt += 1