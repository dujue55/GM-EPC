# src/features.py
import torch
import torchaudio
import librosa
from transformers import AutoModel, AutoTokenizer
import os
import numpy as np
from transformers.utils import hub # 如果不再使用 hub.disable_chat_templates() 也可以删除这个导入
from funasr import AutoModel # <-- 确保 funasr 已安装


# --- 特征维度常量 ---
TEXT_DIM = 768 # 假设使用 BERT Base，其输出维度为 768 (D_t)
SPEECH_DIM = 1024 # 假设 emotion2vec 使用 1024 维度 (D_s) 

# --- 全局模型实例 ---
# ... (保持不变)

global_models ={
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
    MODEL_NAME = "bert-base-uncased"  # 使用最原始名称

    global_models['tokenizer'] = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        revision="main",
        token=None
    )
    global_models['text_model'] = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        revision="main",
        token=None
    ).to(device)
    
    # 2. 语音特征提取器 (emotion2vec)
    # 【已修正缩进：属于 load_feature_extractors 函数体】
    EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_base" 

    try:
        # 修正点 1：使用 FunASR 的 AutoModel，并将其存储到全局字典
        global_models['speech_model'] = AutoModel(model=EMOTION2VEC_MODEL_ID)
        print(f"✅ emotion2vec model loaded: {EMOTION2VEC_MODEL_ID}")

    except Exception as e:
        raise RuntimeError(f"Failed to load emotion2vec model {EMOTION2VEC_MODEL_ID}. The specific error is: {e}")

    # === 验证模型维度 ===
    # 验证文本维度 (保持不变，BERT模型支持)
    actual_text_dim = global_models['text_model'].config.hidden_size
    print(f"✅ Text Model loaded. Configured dim: {TEXT_DIM}, Actual dim: {actual_text_dim}")
    if actual_text_dim != TEXT_DIM:
        print(f"⚠️ 警告：TEXT_DIM 常量 ({TEXT_DIM}) 与实际模型维度 ({actual_text_dim}) 不匹配。请修正 TEXT_DIM。")

    # 验证语音维度 (您关注的重点)
    # 修正点 2：FunASR 模型实例不提供标准的 .config.hidden_size 属性，此处仅打印常量，并依赖特征提取时进行运行时验证。
    print(f"✅ Speech Model loaded. Configured dim: {SPEECH_DIM}") 

    print("Feature extractors loaded successfully.")
    # 【load_feature_extractors 函数体结束】


def extract_single_feature(text_list, audio_path_list):
    """
    提取单个对话样本（L个回合）的特征。
    """
    device = global_models['device']
    text_model = global_models['text_model']
    speech_model = global_models['speech_model']
    tokenizer = global_models['tokenizer']
    
    # 初始化特征列表
    F_t_list = []
    F_s_list = []

    # 遍历 L 个回合
    # 【已修正缩进：属于 extract_single_feature 函数体】
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
            # 提取特征：使用 FunASR 模型的 generate 接口
            with torch.no_grad():
                # FunASR 模型的 generate 方法直接接受文件路径作为输入
                res = speech_model.generate(
                    input=audio_path, 
                    granularity="utterance", # 提取话语级别的特征
                    extract_embedding=True # 确保返回特征向量
                )

                # FunASR 的 generate 结果通常是一个包含字典的列表，其中 'feats' 键对应特征
                if isinstance(res, list) and res and 'feats' in res[0]:
                    speech_feature_np = res[0]['feats']
                    # 将 NumPy 数组转换为 PyTorch 张量
                    speech_feature = torch.from_numpy(speech_feature_np).float().cpu().squeeze() 
                    
                    # 运行时验证：确保实际维度与常量一致
                    if speech_feature.shape[-1] != SPEECH_DIM:
                        print(f"⚠️ 运行时警告：音频文件 {audio_path} 实际语音维度 ({speech_feature.shape[-1]}) 与 SPEECH_DIM ({SPEECH_DIM}) 不匹配！")

                    F_s_list.append(speech_feature)
                else:
                    raise RuntimeError("FunASR generate did not return expected feature format or 'feats' key.")

        except Exception as e:
            print(f"Error loading or processing audio {audio_path} using FunASR: {e}. Returning zero vector.")
            # 如果音频处理失败，返回零向量作为占位符
            F_s_list.append(torch.zeros(SPEECH_DIM))
                
        
    # 将 L 个回合的特征堆叠
    F_t_sequence = torch.stack(F_t_list, dim=0) # [L, D_t]
    F_s_sequence = torch.stack(F_s_list, dim=0) # [L, D_s]
    
    return F_t_sequence, F_s_sequence # 【已修正缩进：属于 extract_single_feature 函数体】


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
        # FunASR 模型没有 .config 属性，这里需要手动打印或在 extract 函数中验证
        print(f"Speech Model output dim (expected): {SPEECH_DIM}") 
    except Exception as e:
        print(f"Model loading FAILED: {e}")