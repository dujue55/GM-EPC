# ================================================================
# ✅ src/model.py (Clean Edition, Oct 2025)
# 支持以下模型：
# 1. Text-Only
# 2. Speech-Only (E2V)
# 3. Speech-Only (WavLM)
# 4. Gated Fusion (E2V)
# 5. Gated Fusion (WavLM)
# ================================================================

import torch
import torch.nn as nn


# ================================================================
# 🧠 Model 1: Gated Fusion (E2V)
# ================================================================
class GatedMultimodalEPC(nn.Module):
    """
    Gated Multimodal Emotion Prediction in Conversation (GM-EPC)
    用于 Emotion2Vec (E2V) 特征的动态门控融合模型。
    """
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(GatedMultimodalEPC, self).__init__()
        
        # 1️⃣ 维度对齐层
        self.use_projection = (speech_dim != text_dim)
        if self.use_projection:
            self.speech_projection = nn.Linear(speech_dim, text_dim)
        
        aligned_dim = text_dim
        
        # 2️⃣ Gating Unit
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * aligned_dim, aligned_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(aligned_dim // 2),
            nn.Linear(aligned_dim // 2, aligned_dim),
            nn.Sigmoid()
        )

        # 3️⃣ GRU 层
        self.gru = nn.GRU(
            input_size=aligned_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # 4️⃣ 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, F_t, F_s):
        F_s_aligned = self.speech_projection(F_s) if self.use_projection else F_s
        H_t_concat = torch.cat((F_t, F_s_aligned), dim=-1)
        W_gate = self.gate_fc(H_t_concat)                   # [B, L, D]
        F_fused = W_gate * F_t + (1 - W_gate) * F_s_aligned
        gru_out, _ = self.gru(F_fused)
        final_output = gru_out[:, -1, :]
        logits = self.classifier(final_output)
        return logits, W_gate


# ================================================================
# 🗣️ Model 2: Text-Only Baseline
# ================================================================
class TextOnlyModel(nn.Module):
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(TextOnlyModel, self).__init__()
        self.gru = nn.GRU(text_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s): 
        gru_out, _ = self.gru(F_t)
        return self.classifier(gru_out[:, -1, :])


# ================================================================
# 🔊 Model 3: Speech-Only (E2V)
# ================================================================
class SpeechOnlyModel(nn.Module):
    def __init__(self, text_dim, speech_dim, hidden_size, num_classes):
        super(SpeechOnlyModel, self).__init__()
        self.gru = nn.GRU(speech_dim, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, F_t, F_s): 
        gru_out, _ = self.gru(F_s)
        return self.classifier(gru_out[:, -1, :])


# ================================================================
# 🌀 Model 4 & 5: BaseWavLMModel (Gated / Only)
# ================================================================
class BaseWavLMModel(GatedMultimodalEPC):
    """
    ✅ 通用的 WavLM 模型基类：
    - 若传入 F_t=None，则退化为 Speech-Only(WavLM)
    - 若传入 F_t!=None，则执行 Gated Fusion(WavLM)
    """
    def forward(self, F_t, F_s):
        if F_t is None:  # Speech-Only 模式
            F_s_aligned = self.speech_projection(F_s) if self.use_projection else F_s
            gru_out, _ = self.gru(F_s_aligned)
            logits = self.classifier(gru_out[:, -1, :])
            return logits
        
        return super().forward(F_t, F_s)


# ================================================================
# ✅ Local Test (for debugging)
# ================================================================
if __name__ == '__main__':
    try:
        from features import get_dummy_features, TEXT_DIM, SPEECH_DIM
    except ImportError:
        from .features import get_dummy_features, TEXT_DIM, SPEECH_DIM

    print("--- Testing All Model Architectures ---")

    BATCH_SIZE = 8
    HISTORY_LEN = 3 
    GRU_HIDDEN_SIZE = 256
    NUM_CLASSES = 4

    dummy_input_t, dummy_input_s_e2v, dummy_input_s_wavlm = get_dummy_features(BATCH_SIZE, HISTORY_LEN)

    models_to_test = {
        "Text-Only": (TextOnlyModel, dummy_input_t, dummy_input_s_e2v),
        "Speech-Only (E2V)": (SpeechOnlyModel, dummy_input_t, dummy_input_s_e2v),
        "Speech-Only (WavLM)": (BaseWavLMModel, None, dummy_input_s_wavlm),
        "Gated Fusion (E2V)": (GatedMultimodalEPC, dummy_input_t, dummy_input_s_e2v),
        "Gated Fusion (WavLM)": (BaseWavLMModel, dummy_input_t, dummy_input_s_wavlm),
    }

    for name, (ModelClass, Ft, Fs) in models_to_test.items():
        print(f"\nTesting {name}...")
        try:
            model = ModelClass(
                text_dim=TEXT_DIM, 
                speech_dim=SPEECH_DIM, 
                hidden_size=GRU_HIDDEN_SIZE, 
                num_classes=NUM_CLASSES
            )
            output = model(Ft, Fs)
            if isinstance(output, tuple):
                logits, W_gate = output
                print(f"  ✅ logits {logits.shape}, gate {W_gate.shape}")
            else:
                print(f"  ✅ logits {output.shape}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
