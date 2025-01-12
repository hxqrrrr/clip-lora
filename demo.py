import os
import torch
import clip
from PIL import Image
import torchvision.transforms as T

# 增加PIL的图片大小限制
Image.MAX_IMAGE_PIXELS = None  # 添加这行来禁用大小限制

def create_demo_dataset():
    # 创建数据集目录
    os.makedirs("demo_data/images/test", exist_ok=True)
    
    # 复制图片到数据集目录
    source_image = "peft2.jpg"
    target_path = "demo_data/images/test/peft2.jpg"
    if os.path.exists(source_image):
        from shutil import copy2
        copy2(source_image, target_path)
    
    return "demo_data"

class DemoDataset:
    def __init__(self, image_path):
        self.image_path = image_path
        self.classnames = ['test_image']
        self.template = ['a photo of a {}.']
        
        # 添加初始的Resize来处理大图片
        self.transform = T.Compose([
            T.Resize((512, 512)),  # 先将图片调整到合理大小
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                       (0.26862954, 0.26130258, 0.27577711))
        ])
        
    def load_image(self):
        try:
            image = Image.open(self.image_path).convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"加载图片时出错: {str(e)}")
            return None

def demo_clip():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("已加载CLIP模型")
    
    # 加载图片
    dataset = DemoDataset("peft2.jpg")
    image = dataset.load_image().to(device)
    
    # 准备文本
    text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)
    
    # 获取图像和文本特征
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # 归一化特征
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    # 打印结果
    print("\n预测结果:")
    for i, label in enumerate(["猫", "狗"]):
        print(f"{label}: {similarity[0][i].item():.2%}")

if __name__ == "__main__":
    demo_clip()
