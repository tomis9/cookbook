import torchvision
from PIL import Image

path ='/home/tomek/parkmediaoutlet/data/images/8618276024.jpg' 
im = Image.open(path)
im.show()

from IPython.display import Image 
pil_img = Image(filename=path)
display(pil_img)


# https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import helper

path = '/home/tomek/nauka/torchvision'
transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(path, transform=transform)

len(dataset[0])
dataset[0][0].shape
dataset[0][1]

im1 = transforms.ToPILImage(dataset[0])
transform = transforms.ToPILImage(mode='RGB')

transform(dataset[0][0])
