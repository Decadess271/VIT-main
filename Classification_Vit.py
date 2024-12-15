import os
import requests
import json
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("GPU is not available, running on CPU.")
# 读取类别标签
with open("D:/githubData/vit-pytorch-main/imagenet-simple-labels.json", "r") as f:
    labels = json.load(f)

# 加载模型和处理器
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.to(device)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 设置图片目录路径
image_dir = "D:/githubData/vit-pytorch-main/tests/images/JPEGImages"  # 想要进行分类的图片目录路径，自定义

# 遍历目录下的所有图片文件
for filename in os.listdir(image_dir):
    # 只处理常见的图片格式文件
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 加载和处理输入图像
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        # 处理图片，转换为模型输入格式
        inputs = processor(images=image, return_tensors="pt")

        # 进行预测
        # 进行预测
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入数据移动到GPU
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(-1).item()  # 获取预测的类别索引

        # 输出预测结果
        predicted_label = labels[predicted_class]
        print(f"File: {filename}, Predicted class: {predicted_class}, Label: {predicted_label}")