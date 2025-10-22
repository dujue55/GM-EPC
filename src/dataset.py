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
        :param data_root: IEMOCAP 原始文件的根目录路径，例如 'IEMOCAP_full_release'。
        :param target_session: 当前要作为测试集的 Session 号 (e.g., 'Session5')。
        :param is_train: 是否加载训练集 (True) 或测试集 (False)。
        :param history_len: 模型的上下文长度 L。
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
        
        # 确定需要处理的 Sessions
        sessions_all = [f"Session{i}" for i in range(1, 6)]
        
        if is_train:
            # 训练集：除了 target_session 之外的所有 Sessions
            data_sessions = [s for s in sessions_all if s != target_session]
        else:
            # 测试集：只有 target_session
            data_sessions = [target_session]
        
        print(f"Processing sessions: {data_sessions}")

        for session in data_sessions:
            # --- 步骤 1: 获取当前 Session 内的所有回合（已在 __init__ 中预加载） ---
            utterances = self.raw_utterances_by_session[session]
            
            if not utterances:
                continue

            # --- 步骤 2: 转换成 EPC 任务样本 ---
            # utterances 是一个按时间顺序排列的列表: [u_1, u_2, ..., u_N]
            
            for i in range(len(utterances)):
                # 我们预测回合 i+1 的情绪
                if i + 1 < len(utterances):
                    
                    target_utterance = utterances[i + 1]
                    target_emo_str = target_utterance['label']
                    
                    # 确保目标情绪是要预测的四类之一
                    if target_emo_str in TARGET_EMOS:
                        
                        # 历史回合：[u_{i-L+1}, ..., u_i]
                        history_start_index = max(0, i - self.history_len + 1)
                        history = utterances[history_start_index : i + 1]
                        
                        target_label_id = EMO_MAP[target_emo_str]
                        
                        # 构造最终样本
                        sample = {
                            # 历史回合的文本和音频路径
                            'history_texts': [u['text'] for u in history],
                            'history_audio_paths': [u['audio_path'] for u in history],
                            # 目标情绪标签 (ID)
                            'target_label': target_label_id,
                            # 目标回合的文本和音频路径 (用于特征提取器中的特征对齐和 padding)
                            'target_text': target_utterance['text'], 
                            'target_audio_path': target_utterance['audio_path'] 
                        }
                        all_samples.append(sample)

        return all_samples

    def _collect_session_utterances(self, session):
        """
        【已实现】解析 IEMOCAP 原始文件，收集一个 Session 内所有回合的文本、标签和音频路径。
        
        !!! 注意: IEMOCAP 的文件结构非常复杂，此函数是基于标准预处理流程的框架。
        它假设数据根目录 'IEMOCAP_full_release/' 下有 SessionX 文件夹。
        """
        
        utterances_in_session = []
        
        # 1. 找到该 Session 下的所有对话目录 (Impro/Script)
        session_dir = os.path.join(self.data_root, session, 'dialog', 'transcriptions')
        
        if not os.path.exists(session_dir):
            print(f"ERROR: Transcription directory not found: {session_dir}. Check data_root path.")
            return []

        # 2. 遍历转录文件以获取 Utterance ID, 文本和时间顺序
        dialog_trans_files = [f for f in os.listdir(session_dir) if f.endswith('.txt')]
        
        # 用于存储对话回合，键是 Utterance ID (e.g., Ses01F_impro01_F000)
        dialog_data = {} 
        
        # 正则表达式用于解析标注文件
        # e.g., '[ 0.0000 - 0.9999]: Ses01F_impro01_F000: HEY!'
        utterance_regex = re.compile(r'\[\s*([\d\.]+)\s*-\s*([\d\.]+)]\s*:\s*(\w+)\s*:\s*(.*)', re.S)


        for trans_file_name in dialog_trans_files:
            trans_path = os.path.join(session_dir, trans_file_name)
            
            with open(trans_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 找到所有匹配的回合
            matches = utterance_regex.findall(content)
            
            for start_time, end_time, utt_id, text in matches:
                # 规范化文本，移除首尾空白和多余的换行
                text = text.strip()
                
                # 预先存储转录信息
                dialog_data[utt_id] = {
                    'text': text,
                    'start': float(start_time),
                    'end': float(end_time),
                }
        
        # 3. 遍历情绪标注文件来添加情绪标签和时间顺序
        emotion_dir = os.path.join(self.data_root, session, 'dialog', 'EmoEvaluation')
        
        dialog_emo_files = [f for f in os.listdir(emotion_dir) if f.endswith('.txt')]
        
        
        # 正则表达式用于解析情绪标注文件
        # e.g., '[ 0.0000 - 0.9999 ] - Ses01F_impro01_F000 [neu]'
        emo_regex = re.compile(r'\[.+?\]\s*-\s*(\w+)\s*\[(\w+)\]', re.S)


        final_utterance_list = [] # 最终按时间顺序排列的回合列表

        for emo_file_name in dialog_emo_files:
            emo_path = os.path.join(emotion_dir, emo_file_name)
            
            with open(emo_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 找到所有情绪标注匹配项
            matches = emo_regex.findall(content)
            
            for utt_id, label in matches:
                label = label.lower()
                
                # 检查是否是我们需要的 Utterance ID 并且情绪在目标范围内
                if utt_id in dialog_data:
                    # 获取该回合的详细信息
                    data = dialog_data[utt_id]
                    
                    # 构造音频文件路径
                    # 路径格式：/SessionX/sentences/wav/[对话名称]/[utt_id].wav
                    # 我们需要从 utt_id (e.g., Ses01F_impro01_F000) 中提取对话名称 (e.g., Ses01F_impro01)
                    dialog_name = "_".join(utt_id.split('_')[:-1])
                    
                    audio_sub_path = os.path.join(
                        session, 'sentences', 'wav', dialog_name, f'{utt_id}.wav'
                    )
                    full_audio_path = os.path.join(self.data_root, audio_sub_path)
                    
                    # 如果情绪不在 TARGET_EMOS 内，我们忽略这个回合
                    if label in TARGET_EMOS or label in ['fru', 'sur', 'dis', 'oth', 'xxx']:
                        # 只有在我们关注的四类情绪内才记录下来，否则作为过滤掉的样本
                        final_utterance_list.append({
                            'utt_id': utt_id,
                            'text': data['text'],
                            'audio_path': full_audio_path,
                            'label': label,
                            'start': data['start'],
                            'end': data['end']
                        })


        # 4. 按时间顺序排序 (IEMOCAP 的标注是按对话分组的，但每个对话内部通常是按时间顺序的)
        # 确保同一个对话 (dialog_name) 内部的 Utterances 按时间排序
        # 为了简化，我们假设读取顺序就是时间顺序，并使用 utt_id 的数字后缀进行粗略排序
        final_utterance_list.sort(key=lambda x: x['utt_id'])
        
        # 5. 最后过滤掉非目标情绪的样本 (通常是在 _load_and_split_data 中进行)
        return final_utterance_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 返回文本和音频路径（以及标签），特征提取在另一个文件/阶段处理
        return {
            'history_texts': sample['history_texts'],
            'history_audio_paths': sample['history_audio_paths'],
            # 返回目标回合信息，这对于特征提取器在处理最后的 L 个回合时很重要
            'target_text': sample['target_text'],
            'target_audio_path': sample['target_audio_path'],
            'target_label': sample['target_label']
        }

# ====================================================================
# 本地调试代码 (IEMOCAP 数据下载完成后，再运行)
# ====================================================================
if __name__ == '__main__':
    print("--- Testing IEMOCAP Dataset Initialization (Requires actual data structure) ---")
    
    # ！！！ 运行前，请设置 IEMOCAP_ROOT ！！！
    # 假设您的 IEMOCAP 根目录路径
    # 例如：
    # IEMOCAP_ROOT = 'data/IEMOCAP_raw/IEMOCAP_full_release' 
    IEMOCAP_ROOT = 'YOUR_IEMOCAP_ROOT_PATH' 

    # 警告：此测试仅在您下载了原始 IEMOCAP 文件并将其放在正确的位置时才能运行！

    # 创建一个训练集 (Leave Session5 out)
    try:
        train_dataset = IEMOCAPDataset(IEMOCAP_ROOT, target_session='Session5', is_train=True, history_len=3)
        print(f"Train Dataset Size: {len(train_dataset)}")

        # 创建一个测试集 (仅 Session5)
        test_dataset = IEMOCAPDataset(IEMOCAP_ROOT, target_session='Session5', is_train=False, history_len=3)
        print(f"Test Dataset Size: {len(test_dataset)}")
        
        # 打印一个样本进行检查
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
