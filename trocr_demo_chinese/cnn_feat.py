import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import scipy.misc
from torchvision import transforms

plt.rcParams['font.sans-serif'] = ['STSong']
import torchvision.models as models

model = models.alexnet(pretrained=True)


# 1.模型查看
# print(model)#可以看出网络一共有3层，两个Sequential()+avgpool
# model_features = list(model.children())
# print(model_features[0][3])#取第0层Sequential()中的第四层
# for index,layer in enumerate(model_features[0]):
#     print(layer)


# 2. 导入数据
# 以RGB格式打开图像
# Pytorch DataLoader就是使用PIL所读取的图像格式
# 建议就用这种方法读取图像，当读入灰度图像时convert('')
def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')  # 是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)  # torch.Size([3, 224, 224])
    image_info = image_info.unsqueeze(0)  # torch.Size([1, 3, 224, 224])因为model的输入要求是4维，所以变成4维
    return image_info  # 变成tensor数据


# 2. 获取第k层的特征图
'''
args:
k:定义提取第几层的feature map
x:图片的tensor
model_layer：是一个Sequential()特征层
'''


def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):  # model的第一个Sequential()是有多层，所以遍历
            x = layer(x)  # torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(
        feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds

    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
   # feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
   # upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
   # feature_map = upsample(feature_map)
   # feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    '''
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(int(row_num), int(row_num), int(index))
        plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        #scipy.misc.imsave('feature_map_save//' + str(index) + ".png", feature_map[index - 1])'''
    plt.imshow(gray_scale, cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
    plt.savefig('./cust-data/resnet18_feature_maps.jpg', bbox_inches='tight')
    plt.show()
def get_weights(model,image):
    from torchvision import transforms
    transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image=Image.open(image)
    image = transforms(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")

    image = image.to(device)
    model_weights = []  # append模型的权重
    conv_layers = []  # append模型的卷积层本身

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0  # 统计模型里共有多少个卷积层
    model_children = list(model.children())
    print(model_children)
    from torch import nn

    # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):  # 遍历最表层(Sequential就是最表层)
        if type(model_children[i]) == nn.Conv2d:  # 最表层只有一个卷积层
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    outputs = []
    names = []

    for layer in conv_layers[0:]:  # conv_layers即是存储了所有卷积层的列表
        image = layer(image)  # 每个卷积层对image做计算，得到以矩阵形式存储的图片，需要通过matplotlib画出
        outputs.append(image)
        names.append(str(layer))

    processed = []

    for feature_map in outputs:
        feature_map = feature_map.squeeze(
            0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[
            0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
        processed.append(
            gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy
    return processed


if __name__ == '__main__':

    image_dir =
    # 定义提取第几层的feature map
    k = 0
    image_info = get_image_info(image_dir)
    model = models.alexnet(pretrained=True)
    model_layer = list(model.children())
    #print(model_layer)
    model_layer = model_layer[1]
    #processed=get_weights(model,image_dir)
    feature_map = get_k_layer_feature_map(model_layer, k, image_info)
    #print(feature_map)





    show_feature_map(feature_map)
