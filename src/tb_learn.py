from torch.fx.immutable_collections import immutable_dict
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "dataset/train/bees_img/16838648_415acd9e3f.jpg"
img_PIL = Image.open(img_path)
img_arr = np.array(img_PIL)

print(img_arr.shape)

writer.add_image("test",img_arr,0, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x",i,2*i)

# writer.add_image()


writer.close()

