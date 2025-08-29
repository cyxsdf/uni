import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from transformers import BertTokenizer


class HumorBertDataset(Dataset):
    def __init__(self, mode, path, max_context_len=5, max_sen_len=20, online=False):
        self.mode = mode
        self.path = path
        self.max_context_len = max_context_len
        self.max_sen_len = max_sen_len
        self.online = online

        # 特征维度定义 - 替换COVAREP为MFCC
        self.word_d = 300  # 词向量维度
        self.mfcc_d = 40  # MFCC特征维度（替代原COVAREP的81维）
        self.of_d = 371  # 视觉特征维度

        # 加载BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 加载数据
        if self.online:
            # 在线模式 - 从原始特征加载
            self.id_list = []
            self.word_aligned_text_sdk = {}
            self.word_aligned_mfcc_sdk = {}  # 存储MFCC特征
            self.word_aligned_openface_sdk = {}
            self.label_sdk = {}

            # 加载数据文件
            with open(os.path.join(path, f"{mode}_ids.txt"), 'r') as f:
                self.id_list = [line.strip() for line in f.readlines()]

            # 加载文本特征
            with open(os.path.join(path, "text_features.json"), 'r') as f:
                self.word_aligned_text_sdk = json.load(f)

            # 加载MFCC音频特征（替代COVAREP）
            with open(os.path.join(path, "mfcc_features.json"), 'r') as f:
                self.word_aligned_mfcc_sdk = json.load(f)

            # 加载视觉特征
            with open(os.path.join(path, "openface_features.json"), 'r') as f:
                self.word_aligned_openface_sdk = json.load(f)

            # 加载标签
            with open(os.path.join(path, "labels.json"), 'r') as f:
                self.label_sdk = json.load(f)
        else:
            # 离线模式 - 从预处理的npy文件加载
            self.text = np.load(os.path.join(path, f"{mode}_text.npy"))
            self.audio = np.load(os.path.join(path, f"{mode}_mfcc.npy"))  # 音频为MFCC特征
            self.video = np.load(os.path.join(path, f"{mode}_video.npy"))
            self.y = np.load(os.path.join(path, f"{mode}_labels.npy"))

    def paded_word_idx(self, seq, max_sen_len=20):
        """处理文本特征，转换为BERT索引并填充"""
        tokens = self.tokenizer.tokenize(seq)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(token_ids) > max_sen_len:
            token_ids = token_ids[:max_sen_len]
        else:
            token_ids += [0] * (max_sen_len - len(token_ids))  # 用0填充

        return np.array(token_ids)

    def padded_openface_features(self, seq, max_sen_len=20):
        """处理视觉特征，右填充至max_sen_len长度"""
        seq = seq[:max_sen_len]  # 截断过长序列
        # 填充零向量至max_sen_len长度
        return np.concatenate((seq, np.zeros((max_sen_len - len(seq), self.of_d))), axis=0)

    def padded_mfcc_features(self, seq, max_sen_len=20):
        """处理MFCC特征，右填充至max_sen_len长度"""
        seq = seq[:max_sen_len]  # 截断过长序列
        # 填充零向量至max_sen_len长度，MFCC维度为self.mfcc_d
        return np.concatenate((seq, np.zeros((max_sen_len - len(seq), self.mfcc_d))), axis=0)

    def padded_context_features(self, context_w, context_of, context_mfcc, max_context_len=5, max_sen_len=20):
        """拼接上下文的文本、视觉和MFCC音频特征"""
        # 只保留最近的max_context_len个上下文
        context_w = context_w[-max_context_len:]
        context_of = context_of[-max_context_len:]
        context_mfcc = context_mfcc[-max_context_len:]

        padded_context = []
        for i in range(len(context_w)):
            p_seq_w = self.paded_word_idx(context_w[i], max_sen_len)
            p_seq_mfcc = self.padded_mfcc_features(context_mfcc[i], max_sen_len)
            p_seq_of = self.padded_openface_features(context_of[i], max_sen_len)

            # 拼接特征：文本 + MFCC + 视觉
            combined = np.concatenate((p_seq_w.reshape(max_sen_len, -1),
                                       p_seq_mfcc,
                                       p_seq_of), axis=1)
            padded_context.append(combined)

        # 填充上下文至max_context_len长度
        pad_c_len = max_context_len - len(padded_context)
        if pad_c_len > 0:
            # 创建零填充
            total_dim = padded_context[0].shape[1] if padded_context else (self.word_d + self.mfcc_d + self.of_d)
            padding = np.zeros((pad_c_len, max_sen_len, total_dim))
            padded_context = np.concatenate((padding, padded_context), axis=0)
        else:
            padded_context = np.array(padded_context)

        return padded_context

    def padded_punchline_features(self, punchline_w, punchline_of, punchline_mfcc, max_sen_len=20):
        """处理笑点的文本、视觉和MFCC音频特征"""
        p_seq_w = torch.FloatTensor(self.paded_word_idx(punchline_w, max_sen_len))
        p_seq_mfcc = torch.FloatTensor(self.padded_mfcc_features(punchline_mfcc, max_sen_len))
        p_seq_of = torch.FloatTensor(self.padded_openface_features(punchline_of, max_sen_len))
        return p_seq_w, p_seq_mfcc, p_seq_of

    def __len__(self):
        if self.online:
            return len(self.id_list)
        else:
            return len(self.y)

    def __getitem__(self, index):
        if self.online:
            hid = self.id_list[index]

            # 获取上下文特征
            context_w = self.word_aligned_text_sdk[hid]['context']
            context_of = [np.array(of) for of in self.word_aligned_openface_sdk[hid]['context_features']]
            context_mfcc = [np.array(mfcc) for mfcc in self.word_aligned_mfcc_sdk[hid]['context_features']]  # MFCC特征

            # 处理上下文特征
            context_features = self.padded_context_features(
                context_w, context_of, context_mfcc,
                self.max_context_len, self.max_sen_len
            )

            # 获取笑点特征
            punchline_w = self.word_aligned_text_sdk[hid]['punchline']
            punchline_of = np.array(self.word_aligned_openface_sdk[hid]['punchline_features'])
            punchline_mfcc = np.array(self.word_aligned_mfcc_sdk[hid]['punchline_features'])  # MFCC特征

            # 处理笑点特征
            x_p = self.padded_punchline_features(
                punchline_w, punchline_of, punchline_mfcc,
                self.max_sen_len
            )

            # 获取标签
            y = torch.FloatTensor([self.label_sdk[hid]])

            # 转换上下文特征为张量
            context_features = torch.FloatTensor(context_features)

            return (context_features, x_p), y
        else:
            # 离线模式直接返回预处理的特征
            x_p = (
                torch.FloatTensor(self.text[index]),
                torch.FloatTensor(self.audio[index]),  # MFCC特征
                torch.FloatTensor(self.video[index])
            )
            y = torch.FloatTensor([self.y[index]])
            return x_p, y
