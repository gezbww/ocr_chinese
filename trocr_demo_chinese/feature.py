import torchvision

model = torchvision.models.resnet18(pretrained=True)
print(model)
import torch
import torchvision
#from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.resnet18(pretrained=True)
model.to(device)

#summary(model,(3,224,224))


from PIL import Image
from torchvision import transforms

# 从硬盘里加载图片
image = Image.open('100186.jpg')

transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

# 图片需要经过一系列数据增强手段以及统计信息(符合ImageNet数据集统计信息)的调整，才能输入模型
image = transforms(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")

image = image.to(device)