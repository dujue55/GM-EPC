# src/dataset.py

import torch
from torch.utils.data import Dataset
import os
import re # 用于正则表达式解析文件

# 定义情绪标签到 ID 的映射（如Happy/Excited, Angry, Sad, Neutral）
EMO_MAP = {
    'hap': 0, 'exc': 0,  # 统一到 Happy/Excited (ID 0)
    'ang': 1,            # Angry (ID 1)
    'sad': 2,            # Sad (ID 2)
    'neu': 3,            # Neutral (ID 3)
}
# 过滤掉不需要的情绪，如 frustration, surprise, disgust, other, unknown, XXX
TARGET_EMOS = list(EMO_MAP.keys())


class IEMOCAPDataset(Dataset):
    """
    处理 IEMOCAP 数据集，并将其格式化为 EPC 任务的输入。
    """
    
    def __init__(self, data_root, target_session, is_train=True, history_len=3):
        """
        初始化数据集。
        """
        self.data_root = data_root
        self.history_len = history_len
        
        # 预加载所有 Session 的数据
        self.raw_utterances_by_session = {}
        for i in range(1, 6):
            session_name = f"Session{i}"
            self.raw_utterances_by_session[session_name] = self._collect_session_utterances(session_name)

        # 基于 LOO-CV 规则切分样本
        self.samples = self._load_and_split_data(target_session, is_train)
        print(f"Loaded {'Train' if is_train else 'Test'} samples: {len(self.samples)} for target session {target_session}")

    def _load_and_split_data(self, target_session, is_train):
        """
        加载 IEMOCAP 原始数据，根据交叉验证规则切分，并格式化为 [历史回合, 目标情绪] 的样本。
        """
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
        
        # --- 调试点 A：确认转录文件夹路径和存在性 ---
        print(f"DEBUG A [{session}]: Checking Transcription Dir: {session_dir}") 
        
        if not os.path.exists(session_dir):
            print(f"ERROR: Transcription directory not found: {session_dir}. Check data_root path.")
            return [] 

        # 2. 遍历转录文件以获取 Utterance ID, 文本和时间顺序
        dialog_trans_files = [f for f in os.listdir(session_dir) if f.endswith('.txt')]

        # --- 调试点 B：确认找到转录文件数量 ---
        print(f"DEBUG B [{session}]: Found {len(dialog_trans_files)} transcription files.")
        
        dialog_data = {} 
        
        # 【转录正则 - 已证明正确】匹配：UtteranceID [TIME_START-TIME_END]: Text
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

        # --- 调试点 C：确认正则匹配成功解析到数据 ---
        print(f"DEBUG C [{session}]: Successfully parsed {len(dialog_data)} utterances from transcriptions.")

        # 3. 遍历情绪标注文件来添加情绪标签和时间顺序
        emotion_dir = os.path.join(self.data_root, session, 'dialog', 'EmoEvaluation')
        
        # --- 调试点 F：检查情绪文件目录 ---
        print(f"DEBUG F [{session}]: Checking Emotion Dir: {emotion_dir}") 
        
        if not os.path.exists(emotion_dir):
            print(f"ERROR: Emotion directory not found: {emotion_dir}.")
            return [] 
            
        dialog_emo_files = [f for f in os.listdir(emotion_dir) if f.endswith('.txt')]

        # --- 调试点 G：检查情绪文件数量和文件名 ---
        print(f"DEBUG G [{session}]: Found {len(dialog_emo_files)} emotion files. Files: {dialog_emo_files[:3]}...") 
        
        if not dialog_emo_files:
            print(f"ERROR: No .txt files found in {emotion_dir}. Check file extension.")
            return []
        
        # src/dataset.py 约 255 行

        # 【最终修正的情绪正则】匹配：[TIME] ID LABEL [V,A,D] 这种结构的行
        # 目标是匹配：[8.2904 - 11.9425] Ses01M_script01_3_F000 neu [4.0000, 2.0000, 2.5000]
        # 捕获组 1: UTTERANCE_ID (\w+)
        # 捕获组 2: LABEL (\w+)
        emo_regex = re.compile(
            r'\[[\d\.]+ - [\d\.]+\]\s*(\w+)\s*(\w+)\s*\[[\d\.]+,[\d\.]+,[\d\.]+\]', 
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
                
            
            # --- DEBUG Y (情绪文件内容检查) ---
            # 仅在 Session1 检查第一个文件
            if session == 'Session1' and emo_file_name == dialog_emo_files[0]:
                print(f"DEBUG Y (Emo): Inspecting first file {emo_file_name}")
                for i, line in enumerate(content.splitlines()):
                    if i > 20: break 
                    line = line.strip()
                    if line:
                        match = emo_regex.search(line)
                        print(f"  Line {i} ({bool(match)}): {line[:100]}...")
                        if match:
                            print(f"  --> Captures: ID={match.group(1)}, Label={match.group(2)}")
            # --- DEBUG Y 结束 ---
                
            # 找到所有情绪标注匹配项
            matches = emo_regex.findall(content)
            
            for utt_id, label in matches: # <--- 解包顺序是 ID, LABEL
                label = label.lower()
                
                # 检查是否在我们已经解析的转录文本中 (这就是 Final list size = 0 的原因)
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
        
        # --- 调试点 D：确认最终合并/过滤后的数据量 ---
        print(f"DEBUG D [{session}]: Final utterance list size: {len(final_utterance_list)}")
        
        return final_utterance_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
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
    
    IEMOCAP_ROOT = 'YOUR_IEMOCAP_ROOT_PATH' 

    try:
        train_dataset = IEMOCAPDataset(IEMOCAP_ROOT, target_session='Session5', is_train=True, history_len=3)
        print(f"Train Dataset Size: {len(train_dataset)}")

        test_dataset = IEMOCAPDataset(IEMOCAP_ROOT, target_session='Session5', is_train=False, history_len=3)
        print(f"Test Dataset Size: {len(test_dataset)}")
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print("\n--- Sample 0 ---")
            print(f"History Texts ({len(sample['history_texts'])}): {sample['history_texts']}")
            print(f"Target Text: {sample['target_text']}")
            print(f"Target Label ID: {sample['target_label']}")
            print(f"Audio Path Example: {sample['history_audio_paths'][0]}")

    except Exception as e:
        print(f"\nFATAL ERROR: Dataset loading failed. Ensure IEMOCAP_ROOT is correct and files exist.")
        print(f"Details: {e}")