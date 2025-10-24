# src/model.py (优化版)

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 核心模型 (Model 5: GM-EPC) ---

class GatedMultimodalEPC(nn.Module):
    """
    Gated Multimodal Emotion Prediction in Conversation (GM-EPC) 模型。
    使用线性层对齐特征维度，然后进行动态门控融合。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(GatedMultimodalEPC, self).__init__()
        
        # 1. 维度对齐层：只有在维度不匹配时才使用投影层 (增强鲁棒性)
        self.use_projection = (speech_dim != text_dim)
        if self.use_projection:
            self.speech_projection = nn.Linear(speech_dim, text_dim)
        
        # 维度对齐后的特征维度
        aligned_dim = text_dim
        
        # 2. Gating Unit：输入是拼接后的特征 (2 * aligned_dim)，输出是门控权重 (aligned_dim)
        self.gate_linear = nn.Linear(2 * aligned_dim, aligned_dim) 
        self.sigmoid = nn.Sigmoid()
        
        # 3. GRU 层：输入维度是融合后的特征维度 (aligned_dim)
        self.gru = nn.GRU(
            input_size=aligned_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 4. 分类层：移除最后一层的 ReLU/Dropout，确保输出是 logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3), # 优化：将 Dropout 移到 ReLU 之后
            nn.Linear(hidden_size // 2, num_classes) # 最后一层是 Logits，无激活函数
        )
    
    def forward(self, F_t, F_s):
        # F_t: [B, L, D_t], F_s: [B, L, D_s]
        
        # 1. 维度对齐
        F_s_aligned = self.speech_projection(F_s) if self.use_projection else F_s
        
        # 2. 拼接 (H_t)
        H_t_concat = torch.cat((F_t, F_s_aligned), dim=-1)
        
        # 3. 计算门控权重 (W_gate)
        W_gate = self.sigmoid(self.gate_linear(H_t_concat))
        
        # 4. 动态融合 (F_fused = W_gate ⊙ F_t + (1 - W_gate) ⊙ F_s_aligned)
        F_fused = W_gate * F_t + (1 - W_gate) * F_s_aligned
        
        # 5. GRU 编码
        gru_out, _ = self.gru(F_fused)
        
        # 6. 预测：只取最后一个回合的输出
        final_output = gru_out[:, -1, :]
        
        # 7. 分类
        logits = self.classifier(final_output)
        
        return logits, W_gate


# --- 基线模型 1: 纯文本 (Text-only) ---

class TextOnlyModel(nn.Module):
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(TextOnlyModel, self).__init__()
        self.gru = nn.GRU(text_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s): 
        gru_out, _ = self.gru(F_t)
        return self.classifier(gru_out[:, -1, :])


# --- 基线模型 2: 纯语音 (Speech-only) ---

class SpeechOnlyModel(nn.Module):
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(SpeechOnlyModel, self).__init__()
        # 修正：如果 D_s != D_t，我们仍然需要处理 D_s 的输入维度
        self.gru = nn.GRU(speech_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s): 
        gru_out, _ = self.gru(F_s)
        return self.classifier(gru_out[:, -1, :])


# --- 基线模型 3: 静态融合 (Static Fusion) ---

class StaticFusionModel(nn.Module):
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(StaticFusionModel, self).__init__()
        
        # 1. 对齐层
        self.use_projection = (speech_dim != text_dim)
        if self.use_projection:
            self.speech_projection = nn.Linear(speech_dim, text_dim)
        
        aligned_dim = text_dim
        input_size = text_dim + aligned_dim # 融合后的维度
        
        # 2. GRU
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. 分类层 (与 GM-EPC 相同)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3), # 优化：将 Dropout 移到 ReLU 之后
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, F_t, F_s):
        # 1. 维度对齐
        F_s_aligned = self.speech_projection(F_s) if self.use_projection else F_s
        
        # 2. 静态拼接融合
        F_static_fused = torch.cat((F_t, F_s_aligned), dim=-1)
        
        # 3. GRU 编码
        gru_out, _ = self.gru(F_static_fused)
        
        # 4. 分类
        return self.classifier(gru_out[:, -1, :])


# --- 基线模型 4: Dynamic WavLM (占位符) ---
# 这个类在训练时，我们会使用 GatedMultimodalEPC，但传入 WavLM 的特征。
# 因此，这个占位符类是用于结构完整性的。
class BaseWavLMModel(TextOnlyModel): # 继承TextOnlyModel以保持简单，实际逻辑在trainer中处理特征加载
    pass


# ====================================================================
# 本地测试代码块 (if __name__ == '__main__':)
# ====================================================================
if __name__ == '__main__':
    # 确保您的 features.py 中的文件路径正确
    try:
        from features import get_dummy_features, TEXT_DIM, SPEECH_DIM
    except ImportError:
        from .features import get_dummy_features, TEXT_DIM, SPEECH_DIM

    print("--- Testing All Model Architectures ---")
    
    # 设定测试参数
    # 设定测试参数
    BATCH_SIZE = 8
    HISTORY_LEN = 3 
    GRU_HIDDEN_SIZE = 256
    NUM_CLASSES = 4
    
    # 🚨 修正：TEXT_DIM 和 SPEECH_DIM 应从 features.py 导入，无需重新定义
    # 假设 features.py 中的 TEXT_DIM=768, SPEECH_DIM=768 (我们之前统一了维度)
    
    # 🚨 修正：调用 get_dummy_features 时只传入两个参数，并接收三个返回
    dummy_input_t, dummy_input_s_e2v, dummy_input_s_wavlm = get_dummy_features(BATCH_SIZE, HISTORY_LEN)
    
    # 为了测试单模态和融合，我们统一使用 F_s_e2v 作为语音输入 F_s
    dummy_input_s = dummy_input_s_e2v
    
    models_to_test = {
        "GM-EPC (Core)": GatedMultimodalEPC,
        "Text Only (Baseline 1)": TextOnlyModel,
        "Speech Only (Baseline 2)": SpeechOnlyModel,
        "Static Fusion (Baseline 3)": StaticFusionModel,
        "Dynamic WavLM (Baseline 4)": BaseWavLMModel,
    }

    for name, ModelClass in models_to_test.items():
        print(f"\nTesting {name}...")
        try:
            model = ModelClass(
                # 🚨 修正：使用导入的常量作为维度
                text_dim=TEXT_DIM, 
                speech_dim=SPEECH_DIM, 
                hidden_size=GRU_HIDDEN_SIZE, 
                num_classes=NUM_CLASSES
            )
            output = model(dummy_input_t, dummy_input_s)
            
            if name == "GM-EPC (Core)":
                logits, W_gate = output  # 👈 解包两个返回值
                final_output = logits
                
                # 验证门控权重形状 (可选，但推荐)
                assert W_gate.shape == (BATCH_SIZE, HISTORY_LEN, TEXT_DIM)
            else:
                final_output = output  # 👈 其他模型只返回 logits
            
            # 使用 final_output 验证最终的分类器形状
            assert final_output.shape == (BATCH_SIZE, NUM_CLASSES)

            print(f"  SUCCESS: Output shape {output.shape} verified.")
            
        except Exception as e:
            print(f"  FAILED: Error during test for {name}: {e}")