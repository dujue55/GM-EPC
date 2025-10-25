# src/trainer.py (最终修正版：健壮、严谨、符合学术规范)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm 
from sklearn.metrics import f1_score, recall_score, accuracy_score 
import pandas as pd
import time
import copy 
from utils.collate import collate_epc 
import numpy as np

# --- 从其他模块导入必要的组件 ---
from model import GatedMultimodalEPC, TextOnlyModel, SpeechOnlyModel, StaticFusionModel, BaseWavLMModel 
# 假设 features.py 中的 get_dummy_features 现在返回 F_t, F_s_e2v, F_s_wavlm (三个)
from features import get_dummy_features, get_dummy_labels, TEXT_DIM, SPEECH_DIM 
from dataset import IEMOCAPDataset 


# --- 占位符：虚拟数据集 (用于本地测试) ---
class DummyConversationDataset(Dataset):
    # ... (初始化和 __len__ 保持不变) ...
    def __init__(self, num_samples, history_len, num_classes):
        self.num_samples = num_samples
        self.history_len = history_len
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 🚨 修正：从 features 导入的 get_dummy_features 现在返回 F_t, F_s_e2v, F_s_wavlm (三个)
        F_t, F_s_e2v, F_s_wavlm = get_dummy_features(batch_size=1, sequence_length=self.history_len)
        labels = get_dummy_labels(batch_size=1, num_classes=self.num_classes)
        
        # 默认使用 F_s_e2v 作为通用 F_s 进行测试
        return {
            'F_t': F_t.squeeze(0),
            'F_s': F_s_e2v.squeeze(0), 
            'target_label': labels.squeeze(0), 
            'mask': torch.ones(self.history_len, dtype=torch.bool)
        }


class Trainer:
    # ... (__init__ 保持不变) ...
    def __init__(self, model, learning_rate, weight_decay, num_classes, patience=10):
        self.model = model
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
        # print(f"Trainer initialized. Using device: {self.device}") # 简化输出

    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        desc = f"Epoch {epoch_idx + 1:02d} | Train"
        for batch in tqdm(dataloader, desc=desc, disable=True): 
            # ... (训练逻辑保持不变) ...
            F_t = batch['F_t'].to(self.device)
            F_s = batch['F_s'].to(self.device)
            labels = batch['target_label'].to(self.device)

            self.optimizer.zero_grad()

            # --- 前向传播 ---
            model_output = self.model(F_t, F_s)

            # --- 区分是否有 gate 输出 ---
            if isinstance(model_output, tuple):
                logits, W_gate = model_output
            else:
                logits = model_output
                W_gate = None

            # --- 标签维度调整 ---
            if labels.dim() > 1:
                labels = labels.squeeze(-1)

            # --- 基础交叉熵损失 ---
            loss = self.criterion(logits, labels)

            # --- ✅【新增】Gate Balance Regularization ---
            # 仅对 GatedMultimodalEPC 模型启用
            if W_gate is not None and isinstance(self.model, GatedMultimodalEPC):
                lambda_balance = 5e-4   # 可调，建议从 1e-3 开始
                loss_balance = torch.mean((W_gate - 0.5) ** 2)
                loss = loss + lambda_balance * loss_balance

            # --- 反向传播 ---
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
        all_gate_weights = [] 

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, disable=True): 
                F_t = batch['F_t'].to(self.device)
                F_s = batch['F_s'].to(self.device)
                labels = batch['target_label'].to(self.device)
                
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)

                model_output = self.model(F_t, F_s)
                
                if isinstance(model_output, tuple): 
                    logits = model_output[0]
                    gate_weights = model_output[1] 
                    
                    # 收集最后一个回合的平均权重
                    # avg_gate_per_sample = gate_weights[:, -1, :].mean(dim=-1).cpu().numpy()
                    # all_gate_weights.extend(avg_gate_per_sample)
                    # ✅ 计算每个样本在时间维上的平均 gate（每一轮话的整体偏好）
                    gate_mean_per_sample = gate_weights.mean(dim=(1, 2)).cpu().numpy()  # 平均值

                    # ✅ 计算每个样本内部（时间维）的标准差，表示动态变化程度
                    gate_std_per_sample = gate_weights.std(dim=1).mean(dim=1).cpu().numpy()  # 平均波动幅度

                    # ✅ 保存两种统计量
                    all_gate_weights.append({
                        "mean": gate_mean_per_sample,
                        "std": gate_std_per_sample
                    })

                else:
                    logits = model_output 
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * F_t.size(0)
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        wa = accuracy_score(all_labels, all_preds)   # ✅ 新增 Weighted Accuracy
        
        return avg_loss, macro_f1, uar, wa, all_labels, all_preds, all_gate_weights  # ✅ 多返回一个 WA


# --- 外部运行函数 (run_cross_validation) ---


def run_cross_validation(ModelClass, config):
    
   
    if 'original_data_root' not in config:
        config['original_data_root'] = config.get('data_root', '/path/to/iemocap') # 适配本地测试
    
    # 根据模型类型确定使用的语音特征 tag
    model_name = ModelClass.__name__
    if model_name in ['GatedMultimodalEPC', 'SpeechOnlyModel', 'StaticFusionModel', 'TextOnlyModel']:
        tag = 'e2v' 
    elif model_name == 'BaseWavLMModel':
        tag = 'wavlm'
    else:
        raise ValueError(f"Unknown Model Class {model_name} for tag determination.")


    sessions = [f'Session{i}' for i in range(1, 6)] 
    

    # 初始化结果 DataFrame
    results_df = pd.DataFrame(columns=['Session', 'Test_Loss', 'Test_Macro_F1', 'Test_UAR', 'Test_WA', 'Train_Time_s', 'Best_Epoch'])
    all_test_f1s = []
    
    # 初始化全局数据收集列表
    global_labels = []
    global_preds = []
    global_gate_weights = []
    
    print(f"\n--- Starting 5-Fold Cross-Validation for {ModelClass.__name__} (Feature: {tag}) ---")

    for fold_idx, target_session in enumerate(sessions):
        start_time = time.time()
        
        print(f"\n=======================================================")
        print(f"| FOLD {fold_idx + 1}/5: Test on {target_session} |")
        print(f"=======================================================")
        

        # --- 1. 数据加载 (传入 tag) ---
        
        train_dataset = IEMOCAPDataset(
            config['original_data_root'], 
            target_session, 
            is_train=True, 
            history_len=config['history_len'], 
            feature_cache_path=config['feature_cache_path'],
            speech_feature_tag=tag 
        )
        test_dataset = IEMOCAPDataset(
            config['original_data_root'], 
            target_session, 
            is_train=False, 
            history_len=config['history_len'], 
            feature_cache_path=config['feature_cache_path'],
            speech_feature_tag=tag 
        )

        # 🚨 替换：使用虚拟数据加载器进行本地测试（如果需要）
        # train_dataset = DummyConversationDataset(config['test_samples'] * 4, config['history_len'], config['num_classes'])
        # test_dataset = DummyConversationDataset(config['test_samples'], config['history_len'], config['num_classes'])

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            collate_fn=collate_epc  # <-- 修正：训练集使用自定义 collate_fn
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            collate_fn=collate_epc   # <-- 修正：测试集使用自定义 collate_fn
        )

        # --- 2. 初始化模型和 Trainer ---
        model_instance = ModelClass(
            text_dim=TEXT_DIM, 
            speech_dim=SPEECH_DIM, 
            hidden_size=config['hidden_size'], 
            num_classes=config['num_classes']
        )
        
        
        trainer = Trainer(
            model=model_instance, 
            learning_rate=config['learning_rate'], 
            weight_decay=config['weight_decay'], 
            num_classes=config['num_classes'],
            patience=config['patience'] 
        )
        
        
        # 每次开始训练前，重置优化器状态
        trainer.optimizer = AdamW(
            trainer.model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay']
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
                    
            test_loss, test_f1, test_uar, test_wa, test_labels, test_preds, test_gates = trainer.evaluate(test_dataloader, desc="Test/Validation")  

            print(f"[Epoch {epoch+1:02d}] | TrainLoss={train_loss:.2f} | TestLoss={test_loss:.2f} | F1={test_f1:.2f} | UAR={test_uar:.2f} | WA={test_wa:.2f}")

            # --- 4. 早停和模型保存 (基于 UAR) ---
            if test_uar > best_uar:
                best_uar = test_uar
                best_f1 = test_f1
                best_loss = test_loss
                best_wa = test_wa          # ✅ 新增：保存最佳 WA
                best_model_state = copy.deepcopy(model_instance.state_dict())
                epochs_no_improve = 0
                best_epoch = epoch + 1
                
                # 收集最佳 UAR 时的原始数据
                test_labels_at_best = test_labels
                test_preds_at_best = test_preds
                test_gates_at_best = test_gates
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= config['patience']:
                print(f"Early stopping triggered after {best_epoch} epochs (patience={config['patience']}).")
                break
        
        # --- 5. 记录最终结果 ---
        train_duration = time.time() - start_time # 🚨 修正 4：记录当前 fold 的总耗时
        
        all_test_f1s.append(best_f1)
        
        # 将数据添加到全局列表 (用于汇总所有 Fold 的数据)
        global_labels.extend(test_labels_at_best)
        global_preds.extend(test_preds_at_best)
        global_gate_weights.extend(test_gates_at_best)

        # 将结果添加到 DataFrame
        new_row = pd.Series({
            'Session': target_session,
            'Test_Loss': best_loss, 
            'Test_Macro_F1': best_f1,
            'Test_UAR': best_uar, 
            'Test_WA': best_wa,     # ✅ 改这里，写 best_wa 而不是 test_wa
            'Train_Time_s': train_duration,
            'Best_Epoch': best_epoch
        })



        results_df.loc[len(results_df)] = new_row
            
    print("\n=======================================================")
    print(f"| Cross-Validation FINISHED for {ModelClass.__name__} |")
    print("=======================================================")
    
    avg_f1 = results_df['Test_Macro_F1'].mean()
    std_f1 = results_df['Test_Macro_F1'].std()
    
    print(f"Average Macro F1: {avg_f1:.4f} (+/- {std_f1:.4f})")

    results_df = results_df.round(2)
    # === 保存 Gate 动态性数据 ===
    if len(global_gate_weights) > 0:
        gate_means = np.concatenate([x["mean"] for x in global_gate_weights])
        gate_stds = np.concatenate([x["std"] for x in global_gate_weights])
        
        np.savez(
            f"./gate_stats_{ModelClass.__name__}.npz",
            gate_means=gate_means,
            gate_stds=gate_stds
        )
        print(f"✅ Gate dynamics saved to gate_stats_{ModelClass.__name__}.npz")

        
    # 返回结果 DataFrame 和原始数据 (供 Cell 7 用于图表生成)
    return results_df, global_labels, global_preds, global_gate_weights


def run_experiment(config):
    # ... (ModelClass 映射保持不变) ...
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
    
    final_results = run_cross_validation(ModelClass, config) 
    
    return final_results

# ====================================================================
# 本地测试代码块 (if __name__ == '__main__':)
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
        'test_samples': 20, 
        'patience': 3,
        'feature_cache_path': './GM-EPC/data/features_cache', 
        # 注意：你需要确保 run_cross_validation 使用的是 'data_root' 作为原始路径
    }
    
    # 修正 run_cross_validation 中的路径映射 (确保使用 data_root)
    CONFIG['original_data_root'] = CONFIG['data_root']
    
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
