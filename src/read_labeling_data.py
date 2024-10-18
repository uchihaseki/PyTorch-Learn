from cProfile import label

from torch.utils.data import Dataset
from PIL import Image
import os

# class MyData(Dataset):
#
#     def __init__(self, root_dir, label_dir):
#         self.root_dir = root_dir
#         self.label_dir = label_dir
#         self.path = os.path.join(self.root_dir, self.label_dir)
#         self.img_path = os.listdir(self.path)
#
#     def __getitem__(self, idx):
#         img_name = self.img_path[idx]
#         img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
#         img = Image.open(img_item_path)
#         label =
#         return img, label
#
#     def __len__(self):
#         return len(self.img_path)




# 生成单独的txt文件
def create_label_files(img_path, root_dir, target_dir, label_dir, label_name):
    # 创建存放标签文件的目录（如果不存在的话）
    #label_dir = os.path.join()
    #os.makedirs(label_dir, exist_ok=True)

    # 遍历每一张图片
    for img in img_path:
        if img.endswith(".jpg") or img.endswith(".png"):
            # 获取图片的完整路径
            img_full_path = os.path.join(root_dir, target_dir, img)

            # 生成相应的txt文件名，和图片同名但后缀为txt
            label_file_name = os.path.splitext(img)[0] + ".txt"
            label_file_path = os.path.join(root_dir, label_dir, label_file_name)

            # 将label写入txt文件
            with open(label_file_path, "w") as f:
                f.write(label_name)

            print(f"Label file created: {label_file_path}")

root_dir = "dataset\\train"
target_dir = "bees_img"
label_dir = "bees_label"
img_path = os.listdir(os.path.join(root_dir,target_dir))
label_name = target_dir.split('_')[0]

# 调用函数为图片生成label文件
create_label_files(img_path, root_dir, target_dir, label_dir, label_name)