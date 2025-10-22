# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 辅助模块 (Gating Unit) ---

# 注意：ModalityGatingUnit类在GM-EPC中被内联实现，这里作为历史保留
class ModalityGatingUnit(nn.Module):
    def __init__(self, input_dim):
        super(ModalityGatingUnit, self).__init__()
        # 此类仅用于结构描述，实际逻辑已在 GatedMultimodalEPC 中实现
        pass

    def forward(self, F_t, F_s):
        # 实际代码中不调用此处的forward
        raise NotImplementedError("This class is primarily for structural reference.")


# --- 核心模型 (Model 5: GM-EPC) ---

class GatedMultimodalEPC(nn.Module):
    """
    Gated Multimodal Emotion Prediction in Conversation (GM-EPC) 模型。
    使用线性层对齐特征维度，然后进行动态门控融合。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(GatedMultimodalEPC, self).__init__()
        
        # 1. 维度对齐层：将 Speech 特征投影到 Text 特征的维度 (768)
        # 这是为了确保 Gating Unit 和 GRU 能够处理一致的维度。
        self.speech_projection = nn.Linear(speech_dim, text_dim)
        
        # 2. Gating Unit：输入是拼接后的特征 (2 * text_dim)，输出是门控权重 (text_dim)
        self.gate_linear = nn.Linear(2 * text_dim, text_dim) 
        self.sigmoid = nn.Sigmoid()
        
        # 3. GRU 层：输入维度是融合后的特征维度 (text_dim)
        self.gru = nn.GRU(
            input_size=text_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 4. 分类层：从最后一个隐藏状态到最终的类别预测
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, F_t, F_s):
        # F_t: [B, L, 768], F_s: [B, L, 1024]
        
        # 1. 维度对齐
        F_s_aligned = self.speech_projection(F_s) # [B, L, 768]
        
        # 2. 拼接 (H_t)
        H_t_concat = torch.cat((F_t, F_s_aligned), dim=-1) # [B, L, 1536]
        
        # 3. 计算门控权重 (W_gate)
        W_gate = self.sigmoid(self.gate_linear(H_t_concat)) # [B, L, 768]
        
        # 4. 动态融合 (F_fused = W_gate ⊙ F_t + (1 - W_gate) ⊙ F_s)
        F_fused = W_gate * F_t + (1 - W_gate) * F_s_aligned # [B, L, 768]
        
        # 5. GRU 编码
        gru_out, _ = self.gru(F_fused)
        
        # 6. 预测：只取最后一个回合的输出
        final_output = gru_out[:, -1, :] # [B, hidden_size]
        
        # 7. 分类
        logits = self.classifier(final_output) # [B, num_classes]
        
        return logits


# --- 基线模型 1: 纯文本 (Text-only) ---

class TextOnlyModel(nn.Module):
    """
    基线模型 1: 仅使用文本特征 (BERT)。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(TextOnlyModel, self).__init__()
        # 只关注文本维度 text_dim
        self.gru = nn.GRU(text_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s): # F_s 即使不使用，也必须作为参数接收以保证接口统一
        gru_out, _ = self.gru(F_t)
        return self.classifier(gru_out[:, -1, :])


# --- 基线模型 2: 纯语音 (Speech-only) ---

class SpeechOnlyModel(nn.Module):
    """
    基线模型 2: 仅使用语音特征 (emotion2vec)。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(SpeechOnlyModel, self).__init__()
        # 语音特征维度为 speech_dim (1024)
        self.gru = nn.GRU(speech_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s): # F_t 即使不使用，也必须作为参数接收
        gru_out, _ = self.gru(F_s)
        return self.classifier(gru_out[:, -1, :])


# --- 基线模型 3: 静态融合 (Static Fusion) ---

class StaticFusionModel(nn.Module):
    """
    基线模型 3: 静态融合。对齐维度后直接拼接，然后送入 GRU。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(StaticFusionModel, self).__init__()
        
        # 1. 对齐层：将 speech 投影到 text_dim (768)
        self.speech_projection = nn.Linear(speech_dim, text_dim)
        
        # 2. GRU 的输入维度是 768 (text) + 768 (aligned speech) = 1536
        input_size = text_dim + text_dim
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
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, F_t, F_s):
        # 1. 维度对齐
        F_s_aligned = self.speech_projection(F_s) # [B, L, 768]
        
        # 2. 静态拼接融合
        F_static_fused = torch.cat((F_t, F_s_aligned), dim=-1) # [B, L, 1536]
        
        # 3. GRU 编码
        gru_out, _ = self.gru(F_static_fused)
        
        # 4. 分类
        return self.classifier(gru_out[:, -1, :])


# --- 基线模型 4: Dynamic WavLM (用于与 GM-EPC 比较) ---

class BaseWavLMModel(nn.Module):
    """
    基线模型 4 (占位符): WavLM 特征的动态融合模型。
    在真实实验中，你需要运行两个版本的 GatedMultimodalEPC：
    1. GM-EPC (使用 emotion2vec)
    2. Dynamic WavLM (使用 WavLM 特征，结构与 GM-EPC 相同)
    
    为了测试通过，这里暂时将其定义为 SpeechOnlyModel的结构，
    但逻辑上，它应该与 GatedMultimodalEPC 结构相同，只是在 features.py 中加载 WavLM 特征。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(BaseWavLMModel, self).__init__()
        # 这里使用 TextOnlyModel 的结构作为占位符，
        # 真正的 Dynamic WavLM 结构应该与 GatedMultimodalEPC 相同，但使用 WavLM 特征。
        self.gru = nn.GRU(text_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s):
        # 占位符：仅使用文本特征
        gru_out, _ = self.gru(F_t)
        return self.classifier(gru_out[:, -1, :])


# ====================================================================
# 本地测试代码块 (if __name__ == '__main__':)
# ====================================================================
if __name__ == '__main__':
    # 确保您的 features.py 中的文件路径正确
    try:
        from .features import get_dummy_features, TEXT_DIM, SPEECH_DIM
    except ImportError:
        # 如果直接运行 model.py，需要使用绝对导入
        from features import get_dummy_features, TEXT_DIM, SPEECH_DIM

    print("--- Testing All Model Architectures ---")
    
    # 设定测试参数
    BATCH_SIZE = 8
    HISTORY_LEN = 3 
    GRU_HIDDEN_SIZE = 256
    NUM_CLASSES = 4
    
    dummy_input_t, dummy_input_s = get_dummy_features(BATCH_SIZE, HISTORY_LEN)
    
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
                text_dim=TEXT_DIM, 
                speech_dim=SPEECH_DIM, 
                hidden_size=GRU_HIDDEN_SIZE, 
                num_classes=NUM_CLASSES
            )
            output = model(dummy_input_t, dummy_input_s)
            
            assert output.shape == (BATCH_SIZE, NUM_CLASSES)
            print(f"  SUCCESS: Output shape {output.shape} verified.")
            
        except Exception as e:
            print(f"  FAILED: Error during test for {name}: {e}")
