import os
import random
import os.path as osp

from .utils import Datum, DatasetBase, listdir_nohidden
from .oxford_pets import OxfordPets

"""
template = ['{} texture.']
"""
template = ['a photo of a {}.']

class DescribableTextures(DatasetBase):

    dataset_dir = ''

    def __init__(self, root, num_shots):
        self.dataset_dir = root
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.train_split = os.path.join(self.dataset_dir, 'labels', 'train1.txt')
        self.val_split = os.path.join(self.dataset_dir, 'labels', 'val1.txt')
        self.test_split = os.path.join(self.dataset_dir, 'labels', 'test1.txt')

        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Image directory: {self.image_dir}")
        print(f"Train split: {self.train_split}")
        print(f"Val split: {self.val_split}")
        print(f"Test split: {self.test_split}")

        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f'Dataset directory not found: {self.dataset_dir}')
        if not osp.exists(self.image_dir):
            raise RuntimeError(f'Image directory not found: {self.image_dir}')
        for split_file in [self.train_split, self.val_split, self.test_split]:
            if not osp.exists(split_file):
                raise RuntimeError(f'Split file not found: {split_file}')

        self.template = ['a photo of a {} texture.']

        # 先读取所有类别
        categories = set()
        for split_file in [self.train_split, self.val_split, self.test_split]:
            with open(split_file, 'r') as f:
                for line in f:
                    classname = line.strip().split('/')[0]
                    categories.add(classname)
        self._classnames = sorted(list(categories))  # 存储为实例变量
        
        # 创建类别到标签的映射
        self.classname_to_label = {name: i for i, name in enumerate(self._classnames)}

        # 然后读取数据
        train = self._read_split_file(self.train_split)
        val = self._read_split_file(self.val_split)
        test = self._read_split_file(self.test_split)

        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    def _read_split_file(self, split_file):
        items = []
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            impath = line.strip()  # 只读取图片路径
            classname = impath.split('/')[0]  # 从路径中提取类别名
            impath = os.path.join(self.image_dir, impath)
            
            # 在创建 Datum 时就设置正确的标签
            item = Datum(
                impath=impath,
                label=self.classname_to_label[classname],  # 直接使用正确的标签
                classname=classname
            )
            items.append(item)
        return items

    @property
    def classnames(self):
        return self._classnames
