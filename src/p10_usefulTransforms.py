from os import write

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
"""
注意如何学习使用：
    1.关注输入和输出类型
    2.多看官方文档
    3.关注方法需要什么参数
    4.多用print()等方法调试来看返回值的type, shape...
"""
writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants_img/5650366_e22b7e1065.jpg")
print(img)

#ToTensor方法

trans_tt = transforms.ToTensor()
img_tensor = trans_tt(img)

writer.add_image("ToTensor_Img", img_tensor)
#writer.close()

#Normalize方法
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)




#Resize方法
print(img.size)
trans_resize = transforms.Resize((128,128))
img_resize = trans_resize(img)
print(img_resize)
img_resize = trans_tt(img_resize)
print(img_resize)
writer.add_image("Resize_ToTensor",img_resize)


#Compose - Resize - 2

trans_resize_2 = transforms.Resize(200)
trans_compose = transforms.Compose([trans_resize_2, trans_tt])

img_resize_2 = trans_compose(img)
writer.add_image("Resize_ToTensor",img_resize_2,1)


#RandomCrop
trans_random = transforms.RandomCrop((300,200))
trans_compose_2 = transforms.Compose([trans_random, trans_tt])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)


writer.close()