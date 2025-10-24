# src/utils/collate.py

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def collate_epc(batch):
    """
    自定义 collate_fn，用于将不同长度的 EPC (Emotion Prediction in Conversation)
    样本填充对齐到批次中的最大序列长度。

    输入: batch (一个包含字典的列表, 字典由 Dataset.__getitem__ 返回)
          每个字典包含: 'F_t', 'F_s', 'target_label'
    输出: 一个包含填充后张量、标签、Mask 和序列长度的字典
    """
    
    F_t_list, F_s_list, labels = [], [], []
    lengths = [] # 用于存储每个样本的实际序列长度

    for b in batch:
        # 收集变长序列张量
        F_t_list.append(b["F_t"])
        F_s_list.append(b["F_s"])
        
        # 收集固定大小的标签
        labels.append(b["target_label"])
        
        # 记录每个样本的实际序列长度 (即第一个维度的大小)
        lengths.append(b["F_t"].shape[0])

    # 1. 对变长序列进行填充 (Padding)
    # pad_sequence 自动找到批次中的最长序列 L_max，并用 0 填充较短序列
    # batch_first=True: 输出形状为 [B, L_max, Feature_Dim]
    F_t_padded = pad_sequence(F_t_list, batch_first=True)  # [B, L_max, 768]
    F_s_padded = pad_sequence(F_s_list, batch_first=True)  # [B, L_max, 768]
    
    # 2. 堆叠固定大小的标签
    labels = torch.stack(labels) # [B]
    
    # 3. 转换序列长度列表为张量
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    # 4. 生成 Mask (指示哪些位置是有效数据，哪些是填充)
    max_len = F_t_padded.shape[1] # 获取最大序列长度 L_max
    
    # 初始化一个全为 False 的 mask: [B, L_max]
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    # 根据实际长度 L 填充 mask
    for i, L in enumerate(lengths):
        mask[i, :L] = True

    return {
        "F_t": F_t_padded,
        "F_s": F_s_padded,
        "target_label": labels,
        "mask": mask,
        "lengths": lengths_tensor, # 关键：返回长度张量，用于 GRU 的 pack_padded_sequence
    }