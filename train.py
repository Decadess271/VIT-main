import torch
from torchvision import transforms
from PIL import Image

from vit_pytorch import ViT

# 加载图像
image_path = 'tests/images/JPEGImages/000026.jpg'  # 自定义图像路径
img = Image.open(image_path).convert('RGB')

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 预处理图像
img_tensor = transform(img).unsqueeze(0)  # 增加一个维度，变成 (1, 3, 224, 224)

# 将预处理后的图像输入到模型中
v = ViT(  # 可自定义ViT模型参数
    image_size=224,  # 图像大小
    patch_size=16,  # patch分块的大小
    num_classes=1000,  # imagenet数据集1000分类
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

v.eval()  # 将模型设为评估模式

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    preds = v(img_tensor)  # (1, 1000)

print(preds)  # 输出预测结果