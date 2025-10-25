# src/dataset.py (已修改：支持加载 F_s_e2v 或 F_s_wavlm)

import torch
from torch.utils.data import Dataset
import os
import re # 用于正则表达式解析文件
import numpy as np 

# 定义情绪标签到 ID 的映射
EMO_MAP = {
    'hap': 0, 'exc': 0, # 统一到 Happy/Excited (ID 0)
    'ang': 1, # Angry (ID 1)
    'sad': 2, # Sad (ID 2)
    'neu': 3, # Neutral (ID 3)
}
TARGET_EMOS = list(EMO_MAP.keys())


class IEMOCAPDataset(Dataset):
    """
    处理 IEMOCAP 数据集，并将其格式化为 EPC 任务的输入。
    """
    
    def __init__(self, data_root, target_session, is_train=True, history_len=3, 
                 feature_cache_path=None, speech_feature_tag=None): # <-- 关键新增 tag 参数
        """
        初始化数据集。
        :param feature_cache_path: 只有在模型训练阶段才传入。
        :param speech_feature_tag: 'e2v' 或 'wavlm'。决定加载哪套语音特征。
        """
        self.data_root = data_root
        self.history_len = history_len
        self.feature_cache_path = feature_cache_path
        self.speech_feature_tag = speech_feature_tag # <-- 保存 tag
        
        # --- 关键修改 1：检查是否为缓存模式 ---
        self.is_cached_mode = (feature_cache_path is not None)
        self.cached_features = {} 
        
        # 预加载所有 Session 的数据 (获取 ID, Label, 顺序)
        self.raw_utterances_by_session = {}
        for i in range(1, 6):
            session_name = f"Session{i}"
            self.raw_utterances_by_session[session_name] = self._collect_session_utterances(session_name)
        
        # --- 关键修改 2：在缓存模式下预加载所有特征文件 ---
        if self.is_cached_mode:
             self._load_all_cached_features()
             print(f"🔍 DEBUG | Cached sessions detected: {list(self.cached_features.keys())}")

        # 基于 LOO-CV 规则切分样本
        self.samples = self._load_and_split_data(target_session, is_train)
        print(f"Loaded {'Train' if is_train else 'Test'} samples: {len(self.samples)} for target session {target_session}")

    # --- 新增辅助方法 1：加载所有缓存特征文件 (支持 tag 区分) ---
    def _load_all_cached_features(self):
        if self.speech_feature_tag not in ['e2v', 'wavlm']:
            raise ValueError("speech_feature_tag must be 'e2v' or 'wavlm' in cached mode.")
            
        tag = self.speech_feature_tag

        for i in range(1, 6):
            session = f"Session{i}"
            ft_path = os.path.join(self.feature_cache_path, f'{session}_F_t.npy')
            fs_path = os.path.join(self.feature_cache_path, f'{session}_F_s_{tag}.npy')
            ids_path = os.path.join(self.feature_cache_path, f'{session}_utt_ids.npy')

            # ✅ 必须确保三者都存在再加载
            if not all(os.path.exists(p) for p in [ft_path, fs_path, ids_path]):
                print(f"⚠️ Skipping {session}: Missing one or more required feature files.")
                continue

            try:
                F_t = np.load(ft_path)
                F_s = np.load(fs_path)
                utt_ids = np.load(ids_path, allow_pickle=True)
            except Exception as e:
                print(f"❌ Error loading cached features for {session}: {e}")
                continue

            self.cached_features[session] = {
                'F_t': F_t,
                'F_s': F_s,
                'id_to_index': {id: i for i, id in enumerate(utt_ids)}
            }

        print(f"✅ Cached sessions loaded: {list(self.cached_features.keys())}")

    # --- _load_and_split_data 保持不变 (它只管数据切分和标签) ---
    def _load_and_split_data(self, target_session, is_train):
        # ... (保持不变) ...
        # 注意: 这里需要添加你之前在 discussion 中确认的 'session', 'history_utt_ids', 'target_utt_id' 字段
        
        all_samples = []
        sessions_all = [f"Session{i}" for i in range(1, 6)]
        
        data_sessions = [s for s in sessions_all if s != target_session] if is_train else [target_session]
        
        print(f"Processing sessions: {data_sessions}")

        for session in data_sessions:
            utterances = self.raw_utterances_by_session[session]
            if not utterances:
                continue

            for i in range(len(utterances)):
                if i + 1 < len(utterances):
                    
                    target_utterance = utterances[i + 1]
                    target_emo_str = target_utterance['label']
                    
                    if target_emo_str in TARGET_EMOS:
                        history_start_index = max(0, i - self.history_len + 1)
                        history = utterances[history_start_index : i + 1]
                        target_label_id = EMO_MAP[target_emo_str]
                        
                        sample = {
                            'session': session, 
                            'history_utt_ids': [u['utt_id'] for u in history], 
                            'target_utt_id': target_utterance['utt_id'], 
                            
                            'history_texts': [u['text'] for u in history],
                            'history_audio_paths': [u['audio_path'] for u in history],
                            'target_label': target_label_id,
                            'target_text': target_utterance['text'], 
                            'target_audio_path': target_utterance['audio_path'] 
                        }
                        all_samples.append(sample)

        return all_samples
        

    def _collect_session_utterances(self, session):
        """
        解析 IEMOCAP 原始文件，收集一个 Session 内所有回合的文本、标签和音频路径。
        """
        
        # 1. 找到该 Session 下的对话目录 (转录)
        session_dir = os.path.join(self.data_root, session, 'dialog', 'transcriptions')
        
        # DEBUG: 检查路径
        # print(f"DEBUG: Checking transcription dir: {session_dir}") 
        
        if not os.path.exists(session_dir):
            print(f"ERROR: Transcription directory not found: {session_dir}.")
            return [] 

        # 2. 遍历转录文件以获取 Utterance ID, 文本和时间顺序
        dialog_trans_files = [f for f in os.listdir(session_dir) if f.endswith('.txt')]
        
        dialog_data = {} 
        
        # 【转录正则】匹配：UtteranceID [TIME_START-TIME_END]: Text
        trans_regex_full = re.compile(r'(\w+)\s*\[([\d\.]+)-([\d\.]+)]:\s*(.*)', re.M)

        for trans_file_name in dialog_trans_files:
            trans_path = os.path.join(session_dir, trans_file_name)
            content = ""
            
            try:
                with open(trans_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(trans_path, 'r', encoding='latin-1') as f:
                    content = f.read()

            matches = trans_regex_full.findall(content)

            for utt_id, start_time, end_time, text_raw in matches:
                text = text_raw.strip().replace('\n', ' ')
                dialog_data[utt_id] = {
                    'text': text,
                    'start': float(start_time), 
                    'end': float(end_time),
                }

        # 3. 遍历情绪标注文件来添加情绪标签和时间顺序
        emotion_dir = os.path.join(self.data_root, session, 'dialog', 'EmoEvaluation')
        
        if not os.path.exists(emotion_dir):
            print(f"ERROR: Emotion directory not found: {emotion_dir}.")
            return [] 
            
        dialog_emo_files = [f for f in os.listdir(emotion_dir) if f.endswith('.txt')]
        
        # 【最终且最宽松的情绪正则】
        emo_regex = re.compile(
            r'\[.+?\]\s+'
            r'([\w\-]+)\s+'
            r'([A-Za-z]+)',
            re.IGNORECASE
        )

        final_utterance_list = [] # 最终按时间顺序排列的回合列表

        for emo_file_name in dialog_emo_files:
            emo_path = os.path.join(emotion_dir, emo_file_name)
            content = ""
            
            try:
                with open(emo_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(emo_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                
            # 找到所有情绪标注匹配项
            matches = emo_regex.findall(content)
            
            for utt_id, label in matches:
                label = label.lower()
                
                # 检查是否在我们已经解析的转录文本中
                if utt_id in dialog_data:
                    data = dialog_data[utt_id]
                    dialog_name = "_".join(utt_id.split('_')[:-1])
                    audio_sub_path = os.path.join(
                        session, 'sentences', 'wav', dialog_name, f'{utt_id}.wav'
                    )
                    full_audio_path = os.path.join(self.data_root, audio_sub_path)
                    
                    if label in TARGET_EMOS or label in ['fru', 'sur', 'dis', 'oth', 'xxx']:
                        final_utterance_list.append({
                            'utt_id': utt_id, 'text': data['text'], 'audio_path': full_audio_path,
                            'label': label, 'start': data['start'], 'end': data['end']
                        })


        # 4. 按 utt_id 排序确保顺序正确
        final_utterance_list.sort(key=lambda x: x['utt_id'])
        
        return final_utterance_list


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.is_cached_mode:
            # --- 缓存模式：返回特征张量和标签 ---
            session = sample['session']
            if session not in self.cached_features:
                raise RuntimeError(f"Missing cached features for {session}. Check if {self.feature_cache_path} is correct.")
            
            cached_data = self.cached_features[session]
            
            # 1. 查找所有回合的特征
            all_utt_ids = sample['history_utt_ids'] + [sample['target_utt_id']]
            F_t_sequence_list = []
            F_s_sequence_list = []
            
            for utt_id in all_utt_ids:
                if utt_id not in cached_data['id_to_index']:
                    raise IndexError("Missing ID in cache map.") 
                    
                index = cached_data['id_to_index'][utt_id]
                F_t_sequence_list.append(cached_data['F_t'][index])
                F_s_sequence_list.append(cached_data['F_s'][index])

            # 2. 堆叠并转换为 Tensor
            F_t_sequence = torch.tensor(np.stack(F_t_sequence_list, axis=0), dtype=torch.float32)
            F_s_sequence = torch.tensor(np.stack(F_s_sequence_list, axis=0), dtype=torch.float32)
            
            # 标签
            target_label = torch.tensor(sample['target_label'], dtype=torch.long)
            
            # 序列长度
            seq_len = F_t_sequence.shape[0]

            return {
                'F_t': F_t_sequence,
                'F_s': F_s_sequence, 
                'target_label': target_label,
                'mask': torch.ones(seq_len, dtype=torch.bool)
            }


        else:
            # --- 原始逻辑 (特征提取阶段): 返回文本/音频路径 ---
            return {
                'history_texts': sample['history_texts'],
                'history_audio_paths': sample['history_audio_paths'],
                'target_text': sample['target_text'],
                'target_audio_path': sample['target_audio_path'],
                'target_label': sample['target_label']
            }

# ====================================================================
# 本地调试代码 (IEMOCAP 数据下载完成后，再运行)
# ====================================================================
if __name__ == '__main__':
    print("--- Testing IEMOCAP Dataset Initialization (Requires actual data structure) ---")
    
    # 假设这里的路径是原始数据路径
    IEMOCAP_ROOT = 'YOUR_IEMOCAP_ROOT_PATH' 
    # 假设缓存特征路径
    CACHE_ROOT = 'YOUR_CACHE_ROOT_PATH' 

    try:
        # 模式 1: 特征提取模式 (无 cache path)
        print("\n--- Testing Feature Extraction Mode (Original Data) ---")
        train_dataset_raw = IEMOCAPDataset(IEMOCAP_ROOT, target_session='Session5', is_train=True, history_len=3)
        print(f"Train Dataset Size (Raw Mode): {len(train_dataset_raw)}")
        # 检查 __getitem__ 返回原始路径
        sample_raw = train_dataset_raw[0]
        print(f"Raw Sample Return: Text={sample_raw['history_texts'][0][:15]}..., Audio Path={sample_raw['history_audio_paths'][0]}")

        # 模式 2: 模型训练模式 (有 cache path)
        print("\n--- Testing Model Training Mode (Cached Features) ---")
        # 注意: 运行此模式需要 CACHE_ROOT 真实存在且包含特征文件
        
        # 🚨 修正：必须传入 speech_feature_tag 参数
        train_dataset_cached = IEMOCAPDataset(
            IEMOCAP_ROOT, 
            target_session='Session5', 
            is_train=True, 
            history_len=3, 
            feature_cache_path=CACHE_ROOT,
            speech_feature_tag='e2v' # <-- 假设我们加载 e2v 特征进行测试
        )
        print(f"Train Dataset Size (Cached Mode): {len(train_dataset_cached)}")
        # 检查 __getitem__ 返回特征张量
        sample_cached = train_dataset_cached[0]
        print(f"Cached Sample Return: F_t Shape={sample_cached['F_t'].shape}, Label={sample_cached['target_label']}")
        
    except Exception as e:
        print(f"\nFATAL ERROR: Dataset testing failed. Ensure paths and data exist.")
        print(f"Details: {e}")