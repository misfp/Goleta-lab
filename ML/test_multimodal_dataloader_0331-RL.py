import torch
import unittest
from multimodal_dataloader import (
    MultimodalDataset,
    multimodal_collate_fn,
    create_multimodal_dataloader
)


class TestMultimodalDataset(unittest.TestCase):
    """测试 MultimodalDataset 类 - 校验 __getitem__ 返回的 Tensor 维度"""

    def setUp(self):
        """测试前的准备工作"""
        self.image_size = (3, 224, 224)
        self.vocab_size = 10000
        self.max_text_len = 512
        self.min_text_len = 10
        self.num_samples = 50

        self.dataset = MultimodalDataset(
            num_samples=self.num_samples,
            image_size=self.image_size,
            vocab_size=self.vocab_size,
            max_text_len=self.max_text_len,
            min_text_len=self.min_text_len
        )

    def test_dataset_length(self):
        """测试数据集长度是否正确"""
        self.assertEqual(len(self.dataset), self.num_samples)

    def test_getitem_returns_dict(self):
        """测试 __getitem__ 返回的是否是字典"""
        sample = self.dataset[0]
        self.assertIsInstance(sample, dict)

    def test_getitem_contains_required_keys(self):
        """测试返回的字典包含所有必需的键"""
        sample = self.dataset[0]
        required_keys = ['image', 'text', 'text_len', 'label']
        for key in required_keys:
            self.assertIn(key, sample)

    def test_image_tensor_dimensions(self):
        """测试图像张量的维度是否正确 (C, H, W)"""
        sample = self.dataset[0]
        image = sample['image']
        self.assertEqual(image.shape, self.image_size)
        self.assertEqual(image.dim(), 3)

    def test_text_tensor_dimensions(self):
        """测试文本张量的维度是否正确 (seq_len,)"""
        sample = self.dataset[0]
        text = sample['text']
        self.assertEqual(text.dim(), 1)

    def test_text_tensor_type(self):
        """测试文本张量的类型是否正确"""
        sample = self.dataset[0]
        text = sample['text']
        self.assertEqual(text.dtype, torch.long)

    def test_text_length_matches(self):
        """测试 text_len 标量与文本序列实际长度一致"""
        for i in range(min(10, len(self.dataset))):
            sample = self.dataset[i]
            text_len = sample['text_len'].item()
            text_seq_len = sample['text'].shape[0]
            self.assertEqual(text_len, text_seq_len)

    def test_text_length_range(self):
        """测试文本长度是否在指定范围内"""
        for i in range(min(10, len(self.dataset))):
            sample = self.dataset[i]
            text_len = sample['text_len'].item()
            self.assertGreaterEqual(text_len, self.min_text_len)
            self.assertLessEqual(text_len, self.max_text_len)

    def test_text_token_range(self):
        """测试文本 token 是否在词汇表范围内"""
        sample = self.dataset[0]
        text = sample['text']
        self.assertTrue((text >= 0).all())
        self.assertTrue((text < self.vocab_size).all())

    def test_text_len_tensor(self):
        """测试 text_len 是标量张量"""
        sample = self.dataset[0]
        text_len = sample['text_len']
        self.assertEqual(text_len.dim(), 0)
        self.assertEqual(text_len.dtype, torch.long)

    def test_label_tensor(self):
        """测试 label 是标量张量"""
        sample = self.dataset[0]
        label = sample['label']
        self.assertEqual(label.dim(), 0)


class TestMultimodalCollateFn(unittest.TestCase):
    """测试 multimodal_collate_fn 函数 - 确保 Batch 中文本正确 Padding"""

    def setUp(self):
        """测试前的准备工作"""
        self.image_size = (3, 224, 224)
        self.dataset = MultimodalDataset(
            num_samples=20,
            image_size=self.image_size,
            max_text_len=100,
            min_text_len=10
        )

    def test_collate_fn_returns_dict(self):
        """测试 collate_fn 返回字典"""
        batch = [self.dataset[i] for i in range(4)]
        collated = multimodal_collate_fn(batch)
        self.assertIsInstance(collated, dict)

    def test_collated_image_batch_shape(self):
        """测试组装后的图像 batch 形状 (B, C, H, W)"""
        batch_size = 8
        batch = [self.dataset[i] for i in range(batch_size)]
        collated = multimodal_collate_fn(batch)
        expected_shape = (batch_size,) + self.image_size
        self.assertEqual(collated['image'].shape, expected_shape)
        self.assertEqual(collated['image'].dim(), 4)

    def test_collated_text_batch_shape(self):
        """测试组装后的文本 batch 形状 (B, max_seq_len)"""
        batch_size = 8
        batch = [self.dataset[i] for i in range(batch_size)]
        collated = multimodal_collate_fn(batch)
        max_len = max([item['text_len'].item() for item in batch])
        expected_shape = (batch_size, max_len)
        self.assertEqual(collated['text'].shape, expected_shape)
        self.assertEqual(collated['text'].dim(), 2)

    def test_padding_value_correct(self):
        """测试填充值是否正确"""
        padding_value = 999
        batch = [self.dataset[i] for i in range(4)]
        collated = multimodal_collate_fn(batch, padding_value=padding_value)
        text_lengths = collated['text_len']
        texts = collated['text']
        for i in range(len(batch)):
            seq_len = text_lengths[i].item()
            padded_region = texts[i, seq_len:]
            self.assertTrue((padded_region == padding_value).all())

    def test_no_padding_in_original_text(self):
        """测试原始文本区域没有填充"""
        padding_value = 999
        batch = [self.dataset[i] for i in range(4)]
        collated = multimodal_collate_fn(batch, padding_value=padding_value)
        text_lengths = collated['text_len']
        texts = collated['text']
        for i in range(len(batch)):
            seq_len = text_lengths[i].item()
            original_region = texts[i, :seq_len]
            self.assertFalse((original_region == padding_value).any())

    def test_padding_side_right(self):
        """测试右侧填充正确性"""
        padding_value = 999
        batch = [self.dataset[i] for i in range(2)]
        collated = multimodal_collate_fn(batch, padding_value=padding_value, padding_side='right')
        max_len = collated['text'].shape[1]
        for i in range(2):
            seq_len = collated['text_len'][i].item()
            if seq_len < max_len:
                self.assertEqual(collated['text'][i, -1].item(), padding_value)

    def test_padding_side_left(self):
        """测试左侧填充正确性"""
        padding_value = 999
        batch = [self.dataset[i] for i in range(2)]
        collated = multimodal_collate_fn(batch, padding_value=padding_value, padding_side='left')
        max_len = collated['text'].shape[1]
        for i in range(2):
            seq_len = collated['text_len'][i].item()
            if seq_len < max_len:
                self.assertEqual(collated['text'][i, 0].item(), padding_value)

    def test_collated_text_len_shape(self):
        """测试组装后的 text_len 形状 (B,)"""
        batch_size = 8
        batch = [self.dataset[i] for i in range(batch_size)]
        collated = multimodal_collate_fn(batch)
        self.assertEqual(collated['text_len'].shape, (batch_size,))
        self.assertEqual(collated['text_len'].dim(), 1)

    def test_collated_label_shape(self):
        """测试组装后的 label 形状 (B,)"""
        batch_size = 8
        batch = [self.dataset[i] for i in range(batch_size)]
        collated = multimodal_collate_fn(batch)
        self.assertEqual(collated['label'].shape, (batch_size,))
        self.assertEqual(collated['label'].dim(), 1)

    def test_text_len_preserved(self):
        """测试组装后的 text_len 与原始一致"""
        batch = [self.dataset[i] for i in range(4)]
        original_lengths = [item['text_len'].item() for item in batch]
        collated = multimodal_collate_fn(batch)
        collated_lengths = collated['text_len'].tolist()
        self.assertEqual(original_lengths, collated_lengths)

    def test_original_text_content_preserved(self):
        """测试原始文本内容在 padding 后保持不变"""
        batch = [self.dataset[i] for i in range(4)]
        collated = multimodal_collate_fn(batch)
        for i, item in enumerate(batch):
            seq_len = item['text_len'].item()
            original_text = item['text']
            padded_text = collated['text'][i, :seq_len]
            self.assertTrue(torch.all(original_text == padded_text))


class TestCreateMultimodalDataloader(unittest.TestCase):
    """测试 create_multimodal_dataloader 便捷函数"""

    def test_dataloader_creation(self):
        """测试 DataLoader 创建成功"""
        dataset = MultimodalDataset(num_samples=20)
        dataloader = create_multimodal_dataloader(dataset, batch_size=4)
        self.assertIsNotNone(dataloader)

    def test_dataloader_iteration(self):
        """测试 DataLoader 可以正常迭代"""
        dataset = MultimodalDataset(num_samples=20)
        dataloader = create_multimodal_dataloader(dataset, batch_size=4)
        for batch in dataloader:
            self.assertIn('image', batch)
            self.assertIn('text', batch)
            self.assertIn('text_len', batch)
            self.assertIn('label', batch)
            break

    def test_dataloader_batch_size(self):
        """测试 DataLoader 的 batch 大小正确"""
        batch_size = 5
        dataset = MultimodalDataset(num_samples=20)
        dataloader = create_multimodal_dataloader(dataset, batch_size=batch_size)
        for batch in dataloader:
            self.assertEqual(batch['image'].shape[0], batch_size)
            self.assertEqual(batch['text'].shape[0], batch_size)
            self.assertEqual(batch['text_len'].shape[0], batch_size)
            self.assertEqual(batch['label'].shape[0], batch_size)
            break


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("运行多模态 DataLoader 测试套件")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalCollateFn))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateMultimodalDataloader))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✓ 所有测试通过！")
    else:
        print(f"✗ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")
    print("=" * 70)

    return result


if __name__ == "__main__":
    run_tests()
