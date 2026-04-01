import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple


class MultimodalDataset(Dataset):
    """
    多模态数据集类，用于处理非对齐的图像和长文本数据
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        vocab_size: int = 10000,
        max_text_len: int = 512,
        min_text_len: int = 10
    ):
        """
        初始化多模态数据集

        参数:
            num_samples: 数据集样本数量
            image_size: 图像尺寸 (channels, height, width)
            vocab_size: 词汇表大小
            max_text_len: 文本最大长度
            min_text_len: 文本最小长度
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.min_text_len = min_text_len

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        返回:
            dict: 包含以下键值对的字典
                - 'image': 图像张量，形状为 (C, H, W)
                - 'text': 文本 token 序列，形状为 (seq_len,)
                - 'text_len': 文本实际长度，标量
                - 'label': 标签，标量
        """
        image = torch.randn(*self.image_size)

        text_len = torch.randint(
            low=self.min_text_len,
            high=self.max_text_len + 1,
            size=()
        ).item()

        text = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(text_len,)
        )

        label = torch.randint(low=0, high=10, size=())

        return {
            'image': image,
            'text': text,
            'text_len': torch.tensor(text_len, dtype=torch.long),
            'label': label
        }


def multimodal_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    padding_value: int = 0,
    padding_side: str = 'right'
) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，用于处理多模态数据的 batch 组装

    参数:
        batch: 样本列表，每个样本是一个字典
        padding_value: 用于填充的值
        padding_side: 填充方向，'right' 或 'left'

    返回:
        dict: 组装后的 batch 数据
            - 'image': 图像 batch，形状为 (B, C, H, W)
            - 'text': 填充后的文本 batch，形状为 (B, max_seq_len)
            - 'text_len': 每个文本的实际长度，形状为 (B,)
            - 'label': 标签 batch，形状为 (B,)
    """
    images = torch.stack([item['image'] for item in batch])
    text_lengths = torch.tensor([item['text_len'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    max_len = text_lengths.max().item()
    batch_size = len(batch)

    padded_texts = torch.full(
        (batch_size, max_len),
        fill_value=padding_value,
        dtype=torch.long
    )

    for i, item in enumerate(batch):
        text = item['text']
        seq_len = text.shape[0]

        if padding_side == 'right':
            padded_texts[i, :seq_len] = text
        else:
            padded_texts[i, -seq_len:] = text

    return {
        'image': images,
        'text': padded_texts,
        'text_len': text_lengths,
        'label': labels
    }


def create_multimodal_dataloader(
    dataset: MultimodalDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    padding_value: int = 0,
    padding_side: str = 'right'
) -> DataLoader:
    """
    创建多模态 DataLoader 的便捷函数

    参数:
        dataset: 多模态数据集
        batch_size: batch 大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
        padding_value: 文本填充值
        padding_side: 填充方向

    返回:
        DataLoader: 配置好的 DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: multimodal_collate_fn(
            batch, padding_value, padding_side
        )
    )
