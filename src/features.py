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
    
    # 1. 文本特征提取器 (BERT Base Uncased)
    MODEL_NAME = "bert-base-uncased" # 使用最原始名称

    # 关键修正：移除 use_fast=False (可能在低版本中不兼容或导致问题)
    # 显式传递 token=None (或 use_auth_token=False/None) 来阻止传递不兼容参数
    
    # 尝试使用 token=None，因为这是新旧版本兼容的参数
    global_models['tokenizer'] = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        token=None # 尝试用 None 覆盖可能传递的默认值
    )
    global_models['text_model'] = AutoModel.from_pretrained(
        MODEL_NAME, 
        token=None # 尝试用 None 覆盖可能传递的默认值
    ).to(device)
    
    
    # 2. 语音特征提取器 (emotion2vec)
    EMOTION2VEC_MODEL_ID = "emotion2vec/emotion2vec_plus_base" 
    
    try:
        global_models['speech_model'] = AutoModel.from_pretrained(
            EMOTION2VEC_MODEL_ID,
            token=None # 语音模型也应用相同的逻辑
        ).to(device)
        print(f"✅ emotion2vec model loaded: {EMOTION2VEC_MODEL_ID}")
        
    except Exception as e:
        # 如果加载失败（例如网络问题或模型结构特殊），则给出明确提示并退出
        raise RuntimeError(f"Failed to load emotion2vec model {EMOTION2VEC_MODEL_ID}. The specific error is: {e}") 


    global_models['text_model'].eval()
    global_models['speech_model'].eval()
    global_models['device'] = device

    # === 新增代码块：验证模型维度 ===
    # 验证文本维度
    actual_text_dim = global_models['text_model'].config.hidden_size
    print(f"✅ Text Model loaded. Configured dim: {TEXT_DIM}, Actual dim: {actual_text_dim}")
    if actual_text_dim != TEXT_DIM:
        print(f"⚠️ 警告：TEXT_DIM 常量 ({TEXT_DIM}) 与实际模型维度 ({actual_text_dim}) 不匹配。请修正 TEXT_DIM。")

    # 验证语音维度 (您关注的重点)
    actual_speech_dim = global_models['speech_model'].config.hidden_size
    print(f"✅ Speech Model loaded. Configured dim: {SPEECH_DIM}, Actual dim: {actual_speech_dim}")
    if actual_speech_dim != SPEECH_DIM:
        print(f"⚠️ 警告：SPEECH_DIM 常量 ({SPEECH_DIM}) 与实际模型维度 ({actual_speech_dim}) 不匹配。请修正 SPEECH_DIM。")
    # ================================
    
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
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

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
