# src/trainer.py

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix # 用于评估
import os
import numpy as np
import pandas as pd
import time
import copy 
import sys # 用于添加路径，以防万一

# 确保在导入前，src 路径已被添加到 Python 路径 (通常在 __init__.py 中或使用 -m 运行已解决)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# 从其他模块导入必要的组件
from .model import GatedMultimodalEPC, TextOnlyModel, SpeechOnlyModel, StaticFusionModel, BaseWavLMModel 
from .features import get_dummy_features, get_dummy_labels, TEXT_DIM, SPEECH_DIM 
# from .dataset import IEMOCAPDataset # 导入真实的 Dataset 类 (等待数据)

# --- 占位符：虚拟数据集 (用于本地测试) ---
class DummyConversationDataset(Dataset):
    def __init__(self, num_samples, history_len, num_classes):
        self.num_samples = num_samples
        self.history_len = history_len
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        F_t, F_s = get_dummy_features(batch_size=1, sequence_length=self.history_len)
        labels = get_dummy_labels(batch_size=1, num_classes=self.num_classes)
        
        return F_t.squeeze(0), F_s.squeeze(0), labels.squeeze(0)


class Trainer:
    def __init__(self, model, learning_rate, weight_decay, num_classes, patience=10):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.num_classes = num_classes
        self.patience = patience # 早停参数
        
        # 检查 GPU 可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Trainer initialized. Using device: {self.device}")


    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        # 使用 tqdm 显示进度
        desc = f"Epoch {epoch_idx + 1:02d} | Train"
        for F_t, F_s, labels in tqdm(dataloader, desc=desc):
            F_t, F_s, labels = F_t.to(self.device), F_s.to(self.device), labels.to(self.device)

            # 1. 前向传播
            logits = self.model(F_t, F_s)

            # 2. 计算损失
            loss = self.criterion(logits, labels)

            # 3. 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * F_t.size(0)
            
            # 记录预测结果
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, macro_f1


    def evaluate(self, dataloader, desc="Evaluation"):
        """
        在验证集或测试集上评估模型性能。
        """
        self.model.eval() # 设置为评估模式
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # 评估时禁用梯度计算
            for F_t, F_s, labels in tqdm(dataloader, desc=desc):
                F_t, F_s, labels = F_t.to(self.device), F_s.to(self.device), labels.to(self.device)

                # 1. 前向传播
                logits = self.model(F_t, F_s)

                # 2. 计算损失
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * F_t.size(0)
                
                # 记录预测结果
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # 返回评估指标和原始标签/预测值
        return avg_loss, macro_f1, all_labels, all_preds


# --- 外部运行函数 ---

def run_cross_validation(ModelClass, config):
    """
    运行 Leave-One-Session-Out 交叉验证 (5折)。
    
    :param ModelClass: 要训练的模型类 (如 GatedMultimodalEPC)
    :param config: 包含所有超参数的字典
    """
    
    # IEMOCAP 的五个 Session
    sessions = [f'Session{i}' for i in range(1, 6)] 
    
    all_test_f1s = []
    
    # 结果 DataFrame
    results_df = pd.DataFrame(columns=['Session', 'Test_Loss', 'Test_Macro_F1', 'Train_Time_s', 'Best_Epoch'])

    print(f"\n--- Starting 5-Fold Cross-Validation for {ModelClass.__name__} ---")

    for fold_idx, target_session in enumerate(sessions):
        print(f"\n=======================================================")
        print(f"| FOLD {fold_idx + 1}/5: Test on {target_session} |")
        print(f"=======================================================")
        
        start_time = time.time()

        # --- 1. 数据加载 ---
        # --- 真实数据加载 (TODO: 等待 IEMOCAP 数据下载完成后启用) ---
        # train_dataset = IEMOCAPDataset(config['data_root'], target_session, is_train=True, history_len=config['history_len'])
        # test_dataset = IEMOCAPDataset(config['data_root'], target_session, is_train=False, history_len=config['history_len'])

        # --- 虚拟数据加载 (用于本地调试) ---
        train_dataset = DummyConversationDataset(config['test_samples'] * 4, config['history_len'], config['num_classes'])
        test_dataset = DummyConversationDataset(config['test_samples'], config['history_len'], config['num_classes'])

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        # --- 2. 初始化模型和 Trainer (必须在每次 Fold 中重新初始化) ---
        model_instance = ModelClass(
            text_dim=TEXT_DIM, 
            speech_dim=SPEECH_DIM, 
            hidden_size=config['hidden_size'], 
            num_classes=config['num_classes']
        )
        
        # 实例化 Trainer (Trainer 实例包含了模型和优化器)
        trainer = Trainer(
            model=model_instance, 
            learning_rate=config['learning_rate'], 
            weight_decay=config['weight_decay'], 
            num_classes=config['num_classes'],
            patience=config['patience'] # 确保传递早停参数
        )
        
        # 记录最佳模型和指标
        best_f1 = -1.0
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        best_loss = 0.0

        # --- 3. 训练循环 (带早停) ---
        for epoch in range(config['epochs']):
            train_loss, train_f1 = trainer.train_epoch(train_dataloader, epoch)
            
            # 验证/测试集评估
            test_loss, test_f1, _, _ = trainer.evaluate(test_dataloader, desc="Test/Validation")
            
            print(f"  --> Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Macro F1={test_f1:.4f}")

            # --- 4. 早停和模型保存 ---
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_loss = test_loss
                best_model_state = copy.deepcopy(model_instance.state_dict())
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= config['patience']:
                print(f"Early stopping triggered after {best_epoch} epochs (patience={config['patience']}).")
                break
        
        # 确保使用最佳 F1 记录结果
        
        # --- 5. 记录最终结果 ---
        end_time = time.time()
        train_duration = end_time - start_time
        
        all_test_f1s.append(best_f1)
        
        # 将结果添加到 DataFrame
        new_row = pd.Series({
            'Session': target_session,
            'Test_Loss': best_loss, # 记录最佳 F1 时的损失
            'Test_Macro_F1': best_f1,
            'Train_Time_s': train_duration,
            'Best_Epoch': best_epoch
        })
        results_df.loc[len(results_df)] = new_row
            
    print("\n=======================================================")
    print(f"| Cross-Validation FINISHED for {ModelClass.__name__} |")
    print("=======================================================")
    
    # 计算平均性能
    avg_f1 = np.mean(all_test_f1s)
    std_f1 = np.std(all_test_f1s)
    
    print(f"Average Macro F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    
    return results_df


def run_experiment(config):
    """
    运行完整的模型实验流程。
    """
    
    # 1. 模型选择和初始化
    model_map = {
        "GM-EPC": GatedMultimodalEPC,
        "Text-Only": TextOnlyModel,
        "Speech-Only": SpeechOnlyModel,
        "Static-Fusion": StaticFusionModel,
        "Dynamic-WavLM": BaseWavLMModel 
    }
    
    if config['model_name'] not in model_map:
        raise ValueError(f"Unknown model name: {config['model_name']}. Choose from {list(model_map.keys())}")
        
    ModelClass = model_map[config['model_name']]
    
    # 2. 运行交叉验证
    # 直接调用 run_cross_validation 函数 (这是静态函数)
    final_results = run_cross_validation(ModelClass, config) # <-- 修正：直接调用函数，并传入 ModelClass 和 config
    
    return final_results


# ====================================================================
# 本地测试代码块 (if __name__ == '__main__':)
# 运行 python src/trainer.py 即可测试
# ====================================================================
if __name__ == '__main__':
    print("--- Starting Local Code Logic Test (Full CV Simulation) ---")

    # --- 1. 实验配置 (代替 YAML 文件) ---
    CONFIG = {
        'model_name': 'GM-EPC', # 测试 Gated Multimodal EPC
        'data_root': '/path/to/iemocap', 
        'history_len': 3,
        'num_classes': 4,
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'hidden_size': 64,
        'test_samples': 20, # 虚拟数据测试样本数量
        'patience': 3,      # 虚拟早停耐心值
    }
    
    # --- 2. 运行完整的虚拟实验 ---
    try:
        results_df = run_experiment(CONFIG)
        
        print("\n--- Full CV Simulation Successful ---")
        print("Aggregated Results:")
        print(results_df)
        print(f"\nOverall Mean F1: {results_df['Test_Macro_F1'].mean():.4f} (+/- {results_df['Test_Macro_F1'].std():.4f})")
        print("Training logic and CV loop verified. You are ready for real data!")
        
    except Exception as e:
        print("\n--- Test FAILED ---")
        print(f"An error occurred during the CV loop: {e}")
        # 打印详细 traceback
        import traceback
        traceback.print_exc()
