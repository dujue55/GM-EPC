import torch
# 1. 导入 transformers 的 AutoModel 并命名为 TransformersAutoModel 或 TextAutoModel
from transformers import AutoModel as TransformersAutoModel, AutoTokenizer

# 2. 导入 funasr 的 AutoModel，我们继续使用 AutoModel，或者命名为 FunASRAutoModel 或 SpeechAutoModel
from funasr import AutoModel
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 特征维度常量 ---
TEXT_DIM = 768 
SPEECH_DIM = 768 # 已根据之前的调试结果修正为 768

# --- 全局模型实例 ---
global_models ={
    'text_model': None,
    'speech_model': None,
    'tokenizer': None,
    'device': torch.device("cpu") # 默认为 CPU，在 load_feature_extractors 中会被更新
}


def load_feature_extractors(device):
    """
    加载所有预训练的特征提取器 (BERT 和 emotion2vec)。
    """
    print(f"Loading feature extractors to device: {device}...")

    # 1. 文本特征提取器 (BERT Base Uncased)
    MODEL_NAME = "bert-base-uncased"

    global_models['tokenizer'] = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        revision="main",
        token=None
    )
    global_models['text_model'] = TransformersAutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=False,
        revision="main",
        token=None
    ).to(device)
    
    # 2. 语音特征提取器 (emotion2vec)
    EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_base"

    try:
        global_models['speech_model'] = AutoModel(model=EMOTION2VEC_MODEL_ID)
        print(f"✅ emotion2vec model loaded: {EMOTION2VEC_MODEL_ID}")

    except Exception as e:
        raise RuntimeError(f"Failed to load emotion2vec model {EMOTION2VEC_MODEL_ID}. The specific error is: {e}")

    # === 关键修正 1：更新全局设备状态 ===
    global_models['device'] = device 
    print(f"✅ Global device state updated to: {global_models['device']}") # 保留这条打印，确保设备状态更新成功

    # === 验证模型维度 ===
    actual_text_dim = global_models['text_model'].config.hidden_size
    print(f"✅ Text Model loaded. Configured dim: {TEXT_DIM}, Actual dim: {actual_text_dim}")
    if actual_text_dim != TEXT_DIM:
        print(f"⚠️ 警告：TEXT_DIM 常量 ({TEXT_DIM}) 与实际模型维度 ({actual_text_dim}) 不匹配。请修正 TEXT_DIM。")

    print(f"✅ Speech Model loaded. Configured dim: {SPEECH_DIM}") 

    print("Feature extractors loaded successfully.")


def extract_single_feature(text_list, audio_path_list):
    """
    提取单个对话样本（L个回合）的特征。
    """
    device = global_models['device']
    text_model = global_models['text_model']
    speech_model = global_models['speech_model']
    tokenizer = global_models['tokenizer']
    
    # [已移除] 调试信息 1：确认当前使用的设备
    
    # 初始化特征列表
    F_t_list = []
    F_s_list = []

    for text, audio_path in zip(text_list, audio_path_list):
        
        # --- 1. 文本特征提取 (F_t) ---
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # === 关键修正 2：使用字典遍历和解包 ===
        # 确保 inputs 字典中的所有张量都移动到正确的设备 (device)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # [已移除] 调试信息 2：检查输入张量的设备
            
        # 提取特征
        with torch.no_grad():
            # 使用字典解包传入所有在 GPU 上的张量
            outputs = text_model(**inputs) 
            
            # 🚨 修正：新增特征赋值行
            text_feature = outputs.last_hidden_state[:, 0, :].squeeze(0) # (D_t)

        
        F_t_list.append(text_feature)

        # --- 2. 语音特征提取 (F_s) ---
        
        try:
            # 提取特征：使用 FunASR 模型的 generate 接口
            with torch.no_grad():
                res = speech_model.generate(
                    input=audio_path,
                    granularity="utterance",
                    extract_embedding=True,
                    progress_bar=False,   # ✅ 禁止 tqdm 进度条
                    show_progress=False,  # ✅ 一些版本用这个参数
                    verbose=False         # ✅ 有的版本还要加这个防止打印
                )

                if isinstance(res, list) and res and 'feats' in res[0]:
                    speech_feature_np = res[0]['feats']
                    
                    # 确保转换为 Tensor 后，发送到 DEVICE
                    speech_feature = torch.from_numpy(speech_feature_np).float().to(device).squeeze() 
                    
                    # 运行时验证：
                    if speech_feature.shape[-1] != SPEECH_DIM:
                        # 保留这个警告，因为它检查的是数据本身的完整性，非常重要。
                        print(f"⚠️ 运行时警告：音频文件 {audio_path} 实际语音维度 ({speech_feature.shape[-1]}) 与 SPEECH_DIM ({SPEECH_DIM}) 不匹配！")

                    F_s_list.append(speech_feature)
                else:
                    raise RuntimeError("FunASR generate did not return expected feature format or 'feats' key.")

        except Exception as e:
            print(f"Error loading or processing audio {audio_path} using FunASR: {e}. Returning zero vector.")
            # 确保零向量占位符在正确的设备上
            F_s_list.append(torch.zeros(SPEECH_DIM, device=device)) 
            
            
    # 将 L 个回合的特征堆叠
    F_t_sequence = torch.stack(F_t_list, dim=0) # [L, D_t]
    F_s_sequence = torch.stack(F_s_list, dim=0) # [L, D_s]
    
    # 🚨 注意：这里返回的张量现在将留在 GPU 上
    return F_t_sequence, F_s_sequence 


# ----------------------------------------------------------------------
# 虚拟数据生成函数 (用于本地 model.py 和 trainer.py 的调试)
# ... (保持不变) ...

def get_dummy_features(batch_size, sequence_length):
    """
    返回随机生成的特征张量，模拟真正的特征提取器输出。
    """
    # 假设在 CPU 上生成虚拟数据，但在实际训练中需要 .to(device)
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
    print("Testing feature extractor loading...")
    try:
        load_feature_extractors(torch.device("cpu"))
        print("Model loading test completed.") 
    except Exception as e:
        print(f"Model loading FAILED: {e}")