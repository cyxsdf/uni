import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
import pickle  # 提前导入pickle模块


class HumorDataset(Dataset):
    """
    幽默多模态数据集加载器，支持文本、MFCC音频和视觉特征的加载与预处理

    支持两种模式：
    - 在线模式：从原始特征文件动态加载和处理
    - 离线模式：加载预处理后的numpy数组
    """

    def __init__(self, mode, path, max_context_len=5, max_sen_len=20, online=False):
        """
        初始化数据集

        参数:
            mode (str): 数据集模式，如'train'、'val'、'test'
            path (str): 数据存放路径
            max_context_len (int): 最大上下文长度
            max_sen_len (int): 句子/序列的最大长度
            online (bool): 是否使用在线模式
        """
        self.mode = mode
        self.path = path
        self.max_context_len = max_context_len
        self.max_sen_len = max_sen_len
        self.online = online

        # 特征维度定义
        self.word_d = 300  # 词向量维度
        self.mfcc_d = 40  # MFCC音频特征维度（替代原COVAREP）
        self.of_d = 371  # 视觉特征维度

        # 加载词表（仅在线模式需要）
        self.vocab = None
        if self.online:
            vocab_path = os.path.join(path, "vocab.json")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    self.vocab = json.load(f)
                self.vocab_size = len(self.vocab)
            else:
                raise FileNotFoundError(f"词表文件 {vocab_path} 不存在")

        # 加载数据
        self._load_data()

    def _load_data(self):
        """内部方法：根据模式加载数据"""
        if self.online:
            # 在线模式 - 从原始特征文件加载
            required_files = [
                f"{self.mode}_ids.txt",
                "text_features.json",
                "mfcc_features.json",
                "openface_features.json",
                "labels.json",
                "word_vectors.npy"
            ]

            # 检查必要文件是否存在
            for file in required_files:
                file_path = os.path.join(self.path, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"在线模式需要文件: {file_path}")

            # 加载ID列表
            with open(os.path.join(self.path, f"{self.mode}_ids.txt"), 'r', encoding='utf-8') as f:
                self.id_list = [line.strip() for line in f.readlines()]

            # 加载文本特征
            with open(os.path.join(self.path, "text_features.json"), 'r', encoding='utf-8') as f:
                self.word_aligned_text_sdk = json.load(f)

            # 加载MFCC音频特征（替代COVAREP）
            with open(os.path.join(self.path, "mfcc_features.json"), 'r', encoding='utf-8') as f:
                self.word_aligned_mfcc_sdk = json.load(f)

            # 加载视觉特征
            with open(os.path.join(self.path, "openface_features.json"), 'r', encoding='utf-8') as f:
                self.word_aligned_openface_sdk = json.load(f)

            # 加载标签
            with open(os.path.join(self.path, "labels.json"), 'r', encoding='utf-8') as f:
                self.label_sdk = json.load(f)

            # 加载词向量
            with open(os.path.join(self.path, "word_vectors.npy"), 'rb') as f:
                self.word_vectors = np.load(f)
        else:
            # 离线模式 - 从预处理的npy文件加载
            required_files = [
                f"{self.mode}_text.npy",
                f"{self.mode}_mfcc.npy",
                f"{self.mode}_video.npy",
                f"{self.mode}_labels.npy"
            ]

            # 检查必要文件是否存在
            for file in required_files:
                file_path = os.path.join(self.path, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"离线模式需要文件: {file_path}")

            self.text = np.load(os.path.join(self.path, f"{self.mode}_text.npy"))
            self.audio = np.load(os.path.join(self.path, f"{self.mode}_mfcc.npy"))  # MFCC音频特征
            self.video = np.load(os.path.join(self.path, f"{self.mode}_video.npy"))
            self.y = np.load(os.path.join(self.path, f"{self.mode}_labels.npy"))

    def paded_word_idx(self, seq, max_sen_len=None):
        """
        将文本序列转换为词索引并填充/截断

        参数:
            seq (str): 文本序列
            max_sen_len (int, optional): 最大句子长度，默认使用类的max_sen_len

        返回:
            np.array: 处理后的词索引序列
        """
        max_len = max_sen_len or self.max_sen_len
        if not self.vocab:
            raise ValueError("词表未加载，请确保在在线模式下使用此方法")

        # 将词转换为索引
        seq_idx = [self.vocab.get(word, 0) for word in seq.split()]  # 未知词用0表示

        # 截断或填充
        if len(seq_idx) > max_len:
            return np.array(seq_idx[:max_len], dtype=np.int32)
        else:
            return np.array(seq_idx + [0] * (max_len - len(seq_idx)), dtype=np.int32)

    def padded_openface_features(self, seq, max_sen_len=None):
        """处理视觉特征，进行填充/截断"""
        max_len = max_sen_len or self.max_sen_len
        seq = np.array(seq) if not isinstance(seq, np.ndarray) else seq

        # 截断过长序列
        if len(seq) > max_len:
            seq = seq[:max_len]

        # 填充零向量至max_sen_len长度
        pad_length = max_len - len(seq)
        if pad_length > 0:
            return np.concatenate([seq, np.zeros((pad_length, self.of_d), dtype=np.float32)], axis=0)
        return seq.astype(np.float32)

    def padded_mfcc_features(self, seq, max_sen_len=None):
        """处理MFCC音频特征，进行填充/截断"""
        max_len = max_sen_len or self.max_sen_len
        seq = np.array(seq) if not isinstance(seq, np.ndarray) else seq

        # 截断过长序列
        if len(seq) > max_len:
            seq = seq[:max_len]

        # 填充零向量至max_sen_len长度
        pad_length = max_len - len(seq)
        if pad_length > 0:
            return np.concatenate([seq, np.zeros((pad_length, self.mfcc_d), dtype=np.float32)], axis=0)
        return seq.astype(np.float32)

    def padded_context_features(self, context_w, context_of, context_mfcc):
        """处理上下文的多模态特征组合"""
        # 只保留最近的max_context_len个上下文
        context_w = context_w[-self.max_context_len:]
        context_of = context_of[-self.max_context_len:]
        context_mfcc = context_mfcc[-self.max_context_len:]

        padded_context = []
        for text, of, mfcc in zip(context_w, context_of, context_mfcc):
            p_seq_w = self.paded_word_idx(text).reshape(self.max_sen_len, -1)
            p_seq_of = self.padded_openface_features(of)
            p_seq_mfcc = self.padded_mfcc_features(mfcc)

            # 拼接特征：文本 + MFCC + 视觉
            combined = np.concatenate([p_seq_w, p_seq_mfcc, p_seq_of], axis=1)
            padded_context.append(combined)

        # 填充上下文至max_context_len长度
        pad_c_len = self.max_context_len - len(padded_context)
        if pad_c_len > 0:
            total_dim = padded_context[0].shape[1] if padded_context else (self.word_d + self.mfcc_d + self.of_d)
            padding = np.zeros((pad_c_len, self.max_sen_len, total_dim), dtype=np.float32)
            padded_context = np.concatenate([padding, padded_context], axis=0)
        else:
            padded_context = np.array(padded_context, dtype=np.float32)

        return padded_context

    def padded_punchline_features(self, punchline_w, punchline_of, punchline_mfcc):
        """处理笑点的多模态特征组合"""
        p_seq_w = self.paded_word_idx(punchline_w).reshape(self.max_sen_len, -1)
        p_seq_of = self.padded_openface_features(punchline_of)
        p_seq_mfcc = self.padded_mfcc_features(punchline_mfcc)

        # 拼接笑点特征
        return np.concatenate([p_seq_w, p_seq_mfcc, p_seq_of], axis=1).astype(np.float32)

    def __len__(self):
        """返回数据集大小"""
        if self.online:
            return len(self.id_list)
        else:
            return len(self.y)

    def __getitem__(self, index):
        """获取索引对应的样本"""
        if self.online:
            hid = self.id_list[index]

            # 验证ID是否存在
            if hid not in self.word_aligned_text_sdk or hid not in self.label_sdk:
                raise KeyError(f"ID {hid} 在数据集中不存在")

            # 获取上下文特征
            context_w = self.word_aligned_text_sdk[hid]['context']
            context_of = [np.array(of) for of in self.word_aligned_openface_sdk[hid]['context_features']]
            context_mfcc = [np.array(mfcc) for mfcc in self.word_aligned_mfcc_sdk[hid]['context_features']]

            # 处理上下文特征
            context_features = self.padded_context_features(context_w, context_of, context_mfcc)

            # 获取笑点特征
            punchline_w = self.word_aligned_text_sdk[hid]['punchline']
            punchline_of = np.array(self.word_aligned_openface_sdk[hid]['punchline_features'])
            punchline_mfcc = np.array(self.word_aligned_mfcc_sdk[hid]['punchline_features'])

            # 处理笑点特征
            punchline_features = self.padded_punchline_features(punchline_w, punchline_of, punchline_mfcc)

            # 组合特征
            x = np.concatenate([context_features, punchline_features[np.newaxis, ...]], axis=0)

            # 获取标签
            y = np.array([self.label_sdk[hid]], dtype=np.float32)

            return torch.FloatTensor(x), torch.FloatTensor(y)
        else:
            # 离线模式直接返回预处理的特征
            x = np.concatenate([self.text[index], self.audio[index], self.video[index]], axis=2)
            y = np.array([self.y[index]], dtype=np.float32)
            return torch.FloatTensor(x), torch.FloatTensor(y)


def load_pickle(file_path):
    """
    加载pickle格式的文件

    参数:
        file_path (str): pickle文件路径

    返回:
        反序列化后的对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        raise ValueError(f"无法解析pickle文件 {file_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"加载pickle文件时出错: {str(e)}")
