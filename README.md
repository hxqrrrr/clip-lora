# CLIP-LoRA

这是一个基于CLIP和LoRA的图像分类项目，目前支持DTD（Describable Textures Dataset）数据集的训练和测试。

## 环境要求

- Python 3.8+
- PyTorch 1.7.1+
- CUDA（推荐）

## 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/CLIP-LoRA.git
cd CLIP-LoRA
```

2. 创建并激活conda环境
```bash
conda create -n clip-lora python=3.8
conda activate clip-lora
```

3. 安装依赖
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
```

4. 下载DTD数据集
```bash
# 下载DTD数据集并解压到data/dtd目录
mkdir -p data/dtd
cd data/dtd
# 从官方网站下载数据集：https://www.robots.ox.ac.uk/~vgg/data/dtd/
```

## 使用方法

### 训练模型

```bash
python main.py --root_path C:\Users\hxq11\Desktop\CLIP-LoRA\data\dtd --dataset dtd --seed 1 --save_path C:\Users\hxq11\Desktop\CLIP-LoRA\saved_models
```

主要参数说明：
- `--dataset`: 数据集名称，目前支持 'dtd'
- `--root_path`: 数据集根目录
- `--backbone`: CLIP模型类型，可选 'ViT-B/32', 'ViT-B/16' 等
- `--shots`: few-shot学习中每个类别的样本数
- `--batch_size`: 训练时的批量大小
- `--save_path`: 模型保存的根目录

### 模型保存位置

训练好的LoRA模型会按照以下结构保存：
```
{save_path}/{backbone}/{dataset}/{shots}shots/seed{seed}/{filename}.pt

示例：
saved_models/vitb32/dtd/1shots/seed1/model.pt
```

### 评估模型（暂未支持）

```bash
python main.py \
    --dataset dtd \
    --root_path data/dtd \
    --backbone ViT-B/32 \
    --eval_only \
    --resume path/to/your/checkpoint.pth
```

## 数据集结构

DTD数据集应按以下结构组织：
```
data/dtd/
    ├── images/
    │   ├── banded/
    │   ├── blotchy/
    │   └── ...
    └── labels/
        ├── train1.txt
        ├── val1.txt
        └── test1.txt
```

## 注意事项

1. 首次运行时会自动下载CLIP预训练模型
2. 确保有足够的GPU内存用于训练
3. 训练日志和模型检查点会保存在项目目录下

