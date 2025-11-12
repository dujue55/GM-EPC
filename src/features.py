# src/features.py (最终版本：支持 mode 参数控制加载)

import torch
from transformers import AutoModel as TransformersAutoModel, AutoTokenizer, AutoFeatureExtractor
import torchaudio
from funasr import AutoModel
import sys, os
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 特征维度常量 & 模型 ID ---
TEXT_DIM = 768 
SPEECH_DIM = 768 
# EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_base"
EMOTION2VEC_MODEL_ID = "iic/emotion2vec_base"
#EMOTION2VEC_MODEL_ID = "iic/emotion2vec_plus_seed"
WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"



# --- 全局模型实例 ---
global_models ={
    'text_model': None,
    'e2v_model': None,                      
    'wavlm_model': None,                   
    'wavlm_feature_extractor': None,       
    'tokenizer': None,
    'device': torch.device("cpu")
}

@contextmanager
def suppress_funasr_output():
    """Temporarily suppress FunASR model's stderr output."""
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

# 【核心修改 1】：load_feature_extractors 接受 mode 参数，并只加载所需的模型
def load_feature_extractors(device, mode="all"):
    """
    根据 mode 参数加载所需的特征提取器。
    mode 可选: 'text', 'e2v', 'wavlm', 'all'
    """
    print(f"Loading feature extractors to device: {device} in mode: {mode}...")

    # --- 0. 卸载语音模型 (只清理需要切换的) ---

    global_models['e2v_model'] = None
    global_models['wavlm_model'] = None
    global_models['wavlm_feature_extractor'] = None
    
    # 1. BERT (Text) - 仅在 'text' 模式下加载/保持加载
    # 在其他模式下，我们将显式卸载/跳过加载，以保持内存最低
    if mode == 'text' or mode == 'all':
        if global_models['text_model'] is None:
            # ... (BERT 加载逻辑) ...
            MODEL_NAME = "bert-base-uncased"
            global_models['tokenizer'] = AutoTokenizer.from_pretrained(MODEL_NAME)
            global_models['text_model'] = TransformersAutoModel.from_pretrained(MODEL_NAME).to(device)
            print("✅ BERT Text Model loaded.")
    else:
        global_models['text_model'] = None
        global_models['tokenizer'] = None
        
    # 2. Emotion2vec (e2v)
    if mode in ["all", "e2v"]:
        try:
            # global_models['e2v_model'] = AutoModel(model=EMOTION2VEC_MODEL_ID)
            global_models['e2v_model'] = AutoModel(model=EMOTION2VEC_MODEL_ID)
            print(f"✅ emotion2vec model loaded: {EMOTION2VEC_MODEL_ID}")
        except Exception as e:
            print(f"⚠️ Warning: E2V failed to load. {e}")
            
    # 3. WavLM
    if mode in ["all", "wavlm"]:
        try:
            global_models['wavlm_feature_extractor'] = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
            global_models['wavlm_model'] = TransformersAutoModel.from_pretrained(WAVLM_MODEL_ID).to(device)
            print(f"WavLM model loaded: {WAVLM_MODEL_ID}")
        except Exception as e:
            print(f"⚠️ Warning: WavLM failed to load. {e}")
    
    global_models['device'] = device 
    print(f"Current active models: T:{bool(global_models['text_model'])}, E2V:{bool(global_models['e2v_model'])}, WLM:{bool(global_models['wavlm_model'])}")


    # === 验证模型维度 ===
    if global_models['text_model'] is not None:
        actual_text_dim = global_models['text_model'].config.hidden_size
        if actual_text_dim != TEXT_DIM:
            print(f"⚠️ 警告：TEXT_DIM 常量 ({TEXT_DIM}) 与实际模型维度 ({actual_text_dim}) 不匹配。请修正 TEXT_DIM。")
    else:
        print("跳过 TEXT 模型维度检查 (当前模式不包含 text 模型)。")

    print("Feature extractors loading process finished. Be aware of potential OOM issues when running all models on GPU.")

# 【核心修改 2】：extract_single_feature 返回三个特征序列
def extract_single_feature(text_list, audio_path_list):
    """
    提取单个回合的特征，不活动的模型返回零向量占位符。
    返回 F_t, F_s_e2v, F_s_wavlm
    """
    device = global_models['device']
    text_model = global_models['text_model']
    e2v_model = global_models['e2v_model']
    wavlm_model = global_models['wavlm_model']
    wavlm_extractor = global_models['wavlm_feature_extractor']
    tokenizer = global_models['tokenizer']
    
    # 初始化三个特征列表
    F_t_list = []
    F_s_e2v_list = []
    F_s_wavlm_list = []

    for text, audio_path in zip(text_list, audio_path_list):
        
        # --- 1. 文本特征提取 (F_t) ---
        if text_model is not None:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # 确保 inputs 字典中的所有张量都移动到正确的设备 (device)
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            with torch.no_grad():
                outputs = text_model(**inputs) 
                # 提取 [CLS] token 的特征
                text_feature = outputs.last_hidden_state[:, 0, :].squeeze(0) # [D_t]
            
            F_t_list.append(text_feature)

        else:
            # 如果 text_model 为 None (即当前 mode 不是 'text' 或 'all')，返回零向量占位符
            # 零向量必须位于正确的设备上，并具有正确的维度
            F_t_list.append(torch.zeros(TEXT_DIM, device=device))

        # --- 2. 语音特征提取 (Emotion2vec) ---
        if e2v_model is not None:
            try:
                with torch.no_grad():
                    with suppress_funasr_output():
                        res = e2v_model.generate(input=audio_path, granularity="utterance", extract_embedding=True)
                    
                    if isinstance(res, list) and res and 'feats' in res[0]:
                        e2v_feature_np = res[0]['feats']
                        e2v_feature = torch.from_numpy(e2v_feature_np).float().to(device).squeeze() 
                        
                        if e2v_feature.shape[-1] != SPEECH_DIM:
                            print(f"⚠️ Warning: E2V dim mismatch for {audio_path}. ({e2v_feature.shape[-1]})")
                        F_s_e2v_list.append(e2v_feature)
                    else:
                        raise RuntimeError("FunASR generate did not return expected feature format or 'feats' key.")
            except Exception as e:
                print(f"Error processing audio {audio_path} using E2V: {e}. Returning zero vector.")
                F_s_e2v_list.append(torch.zeros(SPEECH_DIM, device=device)) 
        else:
             F_s_e2v_list.append(torch.zeros(SPEECH_DIM, device=device)) 


        # --- 3. 语音特征提取 (WavLM) ---
        if wavlm_model is not None:
            try:
                # 必须使用 torchaudio 加载原始音频
                waveform, sr = torchaudio.load(audio_path)
                # WavLM 要求 16k Hz
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)(waveform)
                
                # 预处理
                inputs_wavlm = wavlm_extractor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000)
                inputs_wavlm = {k: v.to(device) for k, v in inputs_wavlm.items()}
                
                with torch.no_grad():
                    outputs_wavlm = wavlm_model(**inputs_wavlm)
                    # WavLM 特征提取：取所有时间步的平均值 (Mean Pooling) 作为 utterance-level 特征
                    wavlm_feature = outputs_wavlm.last_hidden_state.mean(dim=1).squeeze(0) # [D_s]
                
                F_s_wavlm_list.append(wavlm_feature)
            except Exception as e:
                print(f"⚠️ Warning: WavLM feature extraction failed for {audio_path}. {e}. Returning zero vector.")
                F_s_wavlm_list.append(torch.zeros(SPEECH_DIM, device=device))
        else:
             F_s_wavlm_list.append(torch.zeros(SPEECH_DIM, device=device))


    # 将 L 个回合的特征堆叠
    F_t_sequence = torch.stack(F_t_list, dim=0)          # [L, D_t]
    F_s_emotion2vec_sequence = torch.stack(F_s_e2v_list, dim=0) # [L, D_s]
    F_s_wavlm_sequence = torch.stack(F_s_wavlm_list, dim=0)     # [L, D_s]
    
    # 返回三个特征序列

    return F_t_sequence, F_s_emotion2vec_sequence, F_s_wavlm_sequence

# ----------------------------------------------------------------------
# 虚拟数据生成函数 (用于本地 model.py 和 trainer.py 的调试)
# ... (保持不变) ...

def get_dummy_features(batch_size, sequence_length):
    """
    返回随机生成的特征张量，模拟真正的特征提取器输出。
    """
    # 模拟 BERT 特征 (F_t)
    F_t = torch.randn(batch_size, sequence_length, TEXT_DIM) 
    # 模拟 Emotion2vec 特征 (F_s_e2v)
    F_s_e2v = torch.randn(batch_size, sequence_length, SPEECH_DIM) 
    # 模拟 WavLM 特征 (F_s_wavlm)
    F_s_wavlm = torch.randn(batch_size, sequence_length, SPEECH_DIM) 
    
    return F_t, F_s_e2v, F_s_wavlm

def get_dummy_labels(batch_size, num_classes):
    """
    返回随机生成的整数标签。
    """
    labels = torch.randint(0, num_classes, (batch_size,)) 
    return labels

# ... (其余代码保持不变) ...

if __name__ == '__main__':
    print("--- Testing feature extractor loading (Mode Check) ---")
    device = torch.device("cpu")
    
    # Test 1: Only Text mode
    print("\n[TEST 1] Mode: 'text' (Should load only BERT)")
    try:
        load_feature_extractors(device, mode='text')
        assert global_models['text_model'] is not None
        assert global_models['e2v_model'] is None
        assert global_models['wavlm_model'] is None
        print("✅ Test 1 Passed.")
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")

    # Test 2: Only E2V mode
    print("\n[TEST 2] Mode: 'e2v' (Should load only E2V)")
    try:
        # 此时 global_models['text_model'] 会被卸载 (设置为 None)
        load_feature_extractors(device, mode='e2v')
        assert global_models['text_model'] is None
        assert global_models['e2v_model'] is not None
        assert global_models['wavlm_model'] is None
        print("✅ Test 2 Passed.")
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        
    # Test 3: All mode (Warning: Memory heavy)
    print("\n[TEST 3] Mode: 'all' (Should load all three)")
    try:
        load_feature_extractors(device, mode='all')
        assert global_models['text_model'] is not None
        assert global_models['e2v_model'] is not None
        assert global_models['wavlm_model'] is not None
        print("✅ Test 3 Passed.")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")