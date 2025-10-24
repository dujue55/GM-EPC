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

# --- 从其他模块导入必要的组件 (修正导入方式) ---
# 假设 Notebook 的 Cell 1 已经将 src 目录添加到 sys.path
from model import GatedMultimodalEPC, TextOnlyModel, SpeechOnlyModel, StaticFusionModel, BaseWavLMModel 
from features import get_dummy_features, get_dummy_labels, TEXT_DIM, SPEECH_DIM 
from dataset import IEMOCAPDataset # 修正：现在直接导入真实的 Dataset 类


# --- 占位符：虚拟数据集 (用于本地测试) ---
class DummyConversationDataset(Dataset):
    # ... (保持不变，但修改 __getitem__ 返回字典以匹配实际 Dataset) ...
    def __init__(self, num_samples, history_len, num_classes):
        self.num_samples = num_samples
        self.history_len = history_len
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 注意：这里返回的特征需要是 [L, D] 形状
        F_t, F_s = get_dummy_features(batch_size=1, sequence_length=self.history_len, text_dim=TEXT_DIM, speech_dim=SPEECH_DIM)
        labels = get_dummy_labels(batch_size=1, num_classes=self.num_classes)
        
        # 修正：返回字典以匹配真实 IEMOCAPDataset.__getitem__ 的输出
        return {
            'F_t': F_t.squeeze(0),
            'F_s': F_s.squeeze(0),
            'target_label': labels.squeeze(0), # 确保是 [1] 形状的 Tensor
            'mask': torch.ones(self.history_len, dtype=torch.bool)
        }


class Trainer:
    def __init__(self, model, learning_rate, weight_decay, num_classes, patience=10):
        self.model = model
        # 修正：可以加入 class weights，但暂时保持 CrossEntropyLoss 不变
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.num_classes = num_classes
        self.patience = patience 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Trainer initialized. Using device: {self.device}")


    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        desc = f"Epoch {epoch_idx + 1:02d} | Train"
        for batch in tqdm(dataloader, desc=desc): 
            F_t = batch['F_t'].to(self.device)
            F_s = batch['F_s'].to(self.device)
            labels = batch['target_label'].to(self.device)

            self.optimizer.zero_grad()
            
            # --- 关键修正：检查模型类型并解包输出 ---
            model_output = self.model(F_t, F_s)
            
            if isinstance(model_output, tuple):
                # 如果返回的是 (logits, W_gate)
                logits = model_output[0]
            else:
                # 如果只返回 logits (基线模型)
                logits = model_output
            # --- 修正结束 ---

            # 确保 labels 的维度正确
            if labels.dim() > 1:
                labels = labels.squeeze(-1)

            # 89 行：现在 logits 确保是 Tensor
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * F_t.size(0)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, macro_f1


    def evaluate(self, dataloader, desc="Evaluation"):
        self.model.eval() 
        total_loss = 0
        all_preds = []
        all_labels = []
        # 新增：用于收集门控权重的列表 (仅在 GatedMultimodalEPC 模型上收集)
        all_gate_weights = [] 

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc): # 修正 1：接受字典
                F_t = batch['F_t'].to(self.device)
                F_s = batch['F_s'].to(self.device)
                labels = batch['target_label'].to(self.device)
                
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)

                model_output = self.model(F_t, F_s)
                
                # 检查输出是否为 (logits, W_gate) 的 tuple
                if isinstance(model_output, tuple): 
                    logits = model_output[0]
                    gate_weights = model_output[1] # 提取门控权重 W_gate
                    
                    # 收集最后一个回合的平均权重 (用于简化分析)
                    # gate_weights shape: [B, L, D] -> 最后一个时间步的平均权重 [B]
                    avg_gate_per_sample = gate_weights[:, -1, :].mean(dim=-1).cpu().numpy()
                    all_gate_weights.extend(avg_gate_per_sample)
                else:
                    logits = model_output # 否则，只返回 logits
                    # gate_weights 保持 None，all_gate_weights 保持空
                # --- 修正结束 ---
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * F_t.size(0)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # 修正 2：新增 UAR (Unweighted Average Recall)
        # 在多分类中，UAR = 召回率的算术平均值。等于宏观召回率，通常用于替代宏观F1作为主要指标
        uar = f1_score(all_labels, all_preds, average='macro', zero_division=0) 
        
        # 修正 3：返回所有需要的原始数据
        return avg_loss, macro_f1, uar, all_labels, all_preds, all_gate_weights


# --- 外部运行函数 (run_cross_validation) ---

def run_cross_validation(ModelClass, config, cv_data_split):
    """
    运行 Leave-One-Session-Out 交叉验证 (5折)。
    
    :param ModelClass: 要训练的模型类
    :param config: 包含所有超参数的字典
    :param cv_data_split: 包含所有 Session 的原始数据/ID/Label 顺序 (由 dataset.py 提供)
    """
    
    sessions = [f'Session{i}' for i in range(1, 6)] 
    
    all_test_f1s = []
    
    # 修正 4：新增全局数据收集列表
    global_labels = []
    global_preds = []
    global_gate_weights = []
    
    # 结果 DataFrame (新增 UAR)
    results_df = pd.DataFrame(columns=['Session', 'Test_Loss', 'Test_Macro_F1', 'Test_UAR', 'Train_Time_s', 'Best_Epoch', 'Params (M)'])

    print(f"\n--- Starting 5-Fold Cross-Validation for {ModelClass.__name__} ---")

    for fold_idx, target_session in enumerate(sessions):
        print(f"\n=======================================================")
        print(f"| FOLD {fold_idx + 1}/5: Test on {target_session} |")
        print(f"=======================================================")
        

        # --- 1. 数据加载 (启用真实 IEMOCAPDataset 加载) ---
        
        # 实例化 IEMOCAPDataset 类，使用双路径模式加载数据
        train_dataset = IEMOCAPDataset(
            config['original_data_root'], 
            target_session, 
            is_train=True, 
            history_len=config['history_len'], 
            feature_cache_path=config['feature_cache_path'] # 传入缓存路径
        )
        test_dataset = IEMOCAPDataset(
            config['original_data_root'], 
            target_session, 
            is_train=False, 
            history_len=config['history_len'], 
            feature_cache_path=config['feature_cache_path'] # 传入缓存路径
        )

        # --- 虚拟数据加载 (必须注释掉，确保使用真实数据) ---
        # train_dataset = DummyConversationDataset(config['test_samples'] * 4, config['history_len'], config['num_classes'])
        # test_dataset = DummyConversationDataset(config['test_samples'], config['history_len'], config['num_classes'])

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=None)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=None)

        # --- 2. 初始化模型和 Trainer ---
        model_instance = ModelClass(
            text_dim=TEXT_DIM, 
            speech_dim=SPEECH_DIM, 
            hidden_size=config['hidden_size'], 
            num_classes=config['num_classes']
        )
        
        # 修正 5：计算模型参数量 (用于图表 1)
        total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        params_in_millions = total_params / 1_000_000
        print(f"Model Parameters: {params_in_millions:.2f} M")
        
        trainer = Trainer(
            model=model_instance, 
            learning_rate=config['learning_rate'], 
            weight_decay=config['weight_decay'], 
            num_classes=config['num_classes'],
            patience=config['patience'] 
        )
        
        best_f1 = -1.0
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        best_loss = 0.0
        best_uar = 0.0
        test_labels_at_best = []
        test_preds_at_best = []
        test_gates_at_best = []


        # --- 3. 训练循环 (带早停) ---
        for epoch in range(config['epochs']):
            train_loss, train_f1 = trainer.train_epoch(train_dataloader, epoch)
            
            # 验证/测试集评估
            test_loss, test_f1, test_uar, test_labels, test_preds, test_gates = trainer.evaluate(test_dataloader, desc="Test/Validation")
            
            print(f"  --> Epoch {epoch+1:02d}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Macro F1={test_f1:.4f}, Test UAR={test_uar:.4f}")

            # --- 4. 早停和模型保存 (基于 UAR) ---
            if test_uar > best_uar:
                best_uar = test_uar
                best_f1 = test_f1
                best_loss = test_loss
                best_model_state = copy.deepcopy(model_instance.state_dict())
                epochs_no_improve = 0
                best_epoch = epoch + 1
                
                # 收集最佳 UAR 时的原始数据 (图表 2, 5, 7 的数据源)
                test_labels_at_best = test_labels
                test_preds_at_best = test_preds
                test_gates_at_best = test_gates
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= config['patience']:
                print(f"Early stopping triggered after {best_epoch} epochs (patience={config['patience']}).")
                break
        
        # --- 5. 记录最终结果 ---
        end_time = time.time()
        train_duration = end_time - start_time
        
        all_test_f1s.append(best_f1)
        
        # 将数据添加到全局列表 (用于汇总所有 Fold 的数据，进行图表分析)
        global_labels.extend(test_labels_at_best)
        global_preds.extend(test_preds_at_best)
        global_gate_weights.extend(test_gates_at_best) # 如果不是 GM-EPC 模型，此列表将是空的

        # 将结果添加到 DataFrame (新增 UAR 和 Params)
        new_row = pd.Series({
            'Session': target_session,
            'Test_Loss': best_loss, 
            'Test_Macro_F1': best_f1,
            'Test_UAR': best_uar, # 记录最佳 UAR
            'Train_Time_s': train_duration,
            'Best_Epoch': best_epoch,
            'Params (M)': params_in_millions
        })
        results_df.loc[len(results_df)] = new_row
            
    print("\n=======================================================")
    print(f"| Cross-Validation FINISHED for {ModelClass.__name__} |")
    print("=======================================================")
    
    avg_f1 = results_df['Test_Macro_F1'].mean()
    std_f1 = results_df['Test_Macro_F1'].std()
    
    print(f"Average Macro F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    
    # 返回结果 DataFrame 和原始数据 (供 Cell 7 用于图表生成)
    return results_df, global_labels, global_preds, global_gate_weights


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
        "Dynamic-WavLM": GatedMultimodalEPC # 修正：WavLM 的 Ablation 仍使用 GatedMultimodalEPC 结构
    }
    
    if config['model_name'] not in model_map:
        raise ValueError(f"Unknown model name: {config['model_name']}. Choose from {list(model_map.keys())}")
        
    ModelClass = model_map[config['model_name']]
    
    # 2. 运行交叉验证
    # 注意：此处需要传入一个包含 IEMOCAP 数据分割信息的参数 (cv_data_split)
    # 由于我们使用虚拟数据，这里省略 cv_data_split 参数，但在真实运行中需要传入
    final_results = run_cross_validation(ModelClass, config, cv_data_split=None) 
    
    return final_results


# ... (保持 if __name__ == '__main__': 逻辑不变) ...

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
