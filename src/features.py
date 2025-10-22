# src/features.py

import torch
import torchaudio
import librosa
from transformers import AutoModel, AutoTokenizer
import os
import numpy as np

# --- 特征维度常量 ---
TEXT_DIM = 768    # 假设使用 BERT Base，其输出维度为 768 (D_t)
SPEECH_DIM = 1024 # 假设 emotion2vec 使用 1024 维度 (D_s) 

# --- 全局模型实例 ---
# 这些模型将在 Kaggle 上加载到 GPU
global_models = {
    'text_model': None,
    'speech_model': None,
    'tokenizer': None,
    'device': torch.device("cpu") # 默认为 CPU，在 run_experiment 中会被更新
}


def load_feature_extractors(device):
    """
    加载所有预训练的特征提取器 (BERT 和 emotion2vec)。
    """
    print(f"Loading feature extractors to device: {device}...")
    
    # 1. 文本特征提取器 (BERT Base Cased)
    # 使用 AutoModel 自动加载模型，确保使用适当的模型名
    MODEL_NAME = "bert-base-uncased" 
    
    global_models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_NAME)
    global_models['text_model'] = AutoModel.from_pretrained(MODEL_NAME).to(device)
    
    # 2. 语音特征提取器 (emotion2vec)
    # 根据 emotion2vec 论文，它基于 WavLM，通常在 HuggingFace 上有特定的命名空间
    # 假设使用 emotion2vec/emotion2vec_base 或一个兼容 WavLM 的检查点
    SPEECH_MODEL_NAME = "audeering/wavlm-base" # 占位符，如果 emotion2vec 库不同需修改
    
    # 警告：emotion2vec 的官方实现可能需要特定的加载代码。这里使用 WavLM 兼容的占位符。
    try:
        global_models['speech_model'] = AutoModel.from_pretrained("emotion2vec/emotion2vec_base").to(device)
    except Exception:
        print("Warning: emotion2vec model not found directly. Using WavLM base as placeholder.")
        global_models['speech_model'] = AutoModel.from_pretrained(SPEECH_MODEL_NAME).to(device)


    global_models['text_model'].eval()
    global_models['speech_model'].eval()
    global_models['device'] = device
    print("Feature extractors loaded successfully.")


def extract_single_feature(text_list, audio_path_list):
    """
    提取单个对话样本（L个回合）的特征。这是在 Dataloader 的 worker 中调用的函数。
    
    :param text_list: [u_t-L+1_text, ..., u_t_text]
    :param audio_path_list: [u_t-L+1_path, ..., u_t_path]
    :return: F_t (L, D_t), F_s (L, D_s)
    """
    device = global_models['device']
    text_model = global_models['text_model']
    speech_model = global_models['speech_model']
    tokenizer = global_models['tokenizer']
    
    # 初始化特征列表
    F_t_list = []
    F_s_list = []

    # 遍历 L 个回合
    for text, audio_path in zip(text_list, audio_path_list):
        
        # --- 1. 文本特征提取 (F_t) ---
        
        # 将文本编码为 tokens
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 提取特征
        with torch.no_grad():
            outputs = text_model(**inputs)
            # 提取 [CLS] token 的向量作为 utterance 级别的文本特征
            text_feature = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu() # (D_t)
        
        F_t_list.append(text_feature)

        # --- 2. 语音特征提取 (F_s) ---
        
        try:
            # 加载音频文件 (使用 librosa 或 torchaudio)
            speech_array, sampling_rate = torchaudio.load(audio_path)
            
            # 确保采样率为 16kHz (WavLM/emotion2vec 要求)
            if sampling_rate != 16000:
                speech_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(speech_array)
            
            # 语音模型通常只需要单声道
            speech_array = speech_array.squeeze(0).to(device).unsqueeze(0) # [1, Samples]
            
            # 提取特征
            with torch.no_grad():
                # 注意：emotion2vec/WavLM 的输入处理和输出层可能需要调整，这里是基础 WavLM/HuBERT 接口
                speech_outputs = speech_model(speech_array, output_hidden_states=True)
                # 提取 utterance 级别的语音特征（通常是最后一层所有帧的平均值）
                
                # 假设提取最后一层隐藏状态的平均值
                last_hidden_state = speech_outputs.last_hidden_state.squeeze(0) # [Frames, D_s]
                speech_feature = torch.mean(last_hidden_state, dim=0).cpu() # (D_s)
            
            F_s_list.append(speech_feature)

        except Exception as e:
            print(f"Error loading or processing audio {audio_path}: {e}. Returning zero vector.")
            # 如果音频处理失败，返回零向量作为占位符
            F_s_list.append(torch.zeros(SPEECH_DIM))
            
    
    # 将 L 个回合的特征堆叠
    F_t_sequence = torch.stack(F_t_list, dim=0) # [L, D_t]
    F_s_sequence = torch.stack(F_s_list, dim=0) # [L, D_s]
    
    return F_t_sequence, F_s_sequence


# ----------------------------------------------------------------------
# ！！！保留本地调试用的虚拟数据函数！！！
# ----------------------------------------------------------------------

# 虚拟数据生成函数 (用于本地 model.py 和 trainer.py 的调试)
def get_dummy_features(batch_size, sequence_length):
    """
    返回随机生成的特征张量，模拟真正的特征提取器输出。
    """
    F_t = torch.randn(batch_size, sequence_length, TEXT_DIM) 
    F_s = torch.randn(batch_size, sequence_length, SPEECH_DIM) 
    return F_t, F_s

def get_dummy_labels(batch_size, num_classes):
    """
    返回随机生成的整数标签。
    """
    labels = torch.randint(0, num_classes, (batch_size,)) 
    return labels

if __name__ == '__main__':
    # 运行此块来测试模型加载和维度
    print("Testing feature extractor loading...")
    try:
        load_feature_extractors(torch.device("cpu"))
        print(f"Text Model output dim (assumed): {global_models['text_model'].config.hidden_size}")
        print(f"Speech Model output dim (assumed): {global_models['speech_model'].config.hidden_size}")
    except Exception as e:
        print(f"Model loading FAILED. This is expected if 'transformers' cannot find the checkpoint: {e}")
