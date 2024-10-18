import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from model import *

img_path = "../img/dog_03.png"
image = Image.open(img_path)

print(image)

image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])


image = transform(image)
print(image.shape)




model = torch.load("tudui_25.pth")
print(model)
image = torch.reshape(image,(-1,3,32,32))
#model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))
#模型从GPU到CPU的映射
image = image.cuda()

model.eval()
with torch.no_grad():
    output = model(image)

print(output)
print(output.argmax(1))