import torchvision

model = torchvision.models.resnet18(pretrained=True)
import torch
import torchvision
import cv2
#from torchsummary import summary
def res18(image_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torchvision.models.resnet18(pretrained=True)
    model.to(device)
    from PIL import Image
    from torchvision import transforms
    import numpy as np
    # 从硬盘里加载图片
    image = Image.open(image_dir).convert('RGB')
    #image=image_dir
    transforms = transforms.Compose([#transforms.Resize(256),
                                     #transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    # 图片需要经过一系列数据增强手段以及统计信息(符合ImageNet数据集统计信息)的调整，才能输入模型
    image = transforms(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")

    image = image.to(device)
    model_weights = []   # append模型的权重
    conv_layers = []   # append模型的卷积层本身

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0  # 统计模型里共有多少个卷积层
    model_children = list(model.children())
    #print(model_children)
    from torch import nn

    # append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):  # 遍历最表层(Sequential就是最表层)
        if type(model_children[i]) == nn.Conv2d:   # 最表层只有一个卷积层
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    #print(f"Total convolution layers: {counter}")
    outputs = []
    names = []

    for layer in conv_layers[0:]:    # conv_layers即是存储了所有卷积层的列表
        image = layer(image)  # 每个卷积层对image做计算，得到以矩阵形式存储的图片，需要通过matplotlib画出
        outputs.append(image)
        names.append(str(layer))


    processed = []

    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
        processed.append(gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy
    #return processed[0]
#print(processed[0])
    import matplotlib.pyplot as plt

    fig = plt.figure()


    img_plot = plt.imshow(processed[0])
    plt.axis("off")
    #a.set_title(names[i].split('(')[0], fontsize=30)   # names[i].split('(')[0] 结果为Conv2d
    print("the pic{} is finished".format(image_dir[58:]))
    plt.savefig("".format(image_dir[58:]), bbox_inches='tight',pad_inches=0)

