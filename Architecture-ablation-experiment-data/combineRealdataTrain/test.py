"""
测试电动助力车 PPO + Transformer 模型（真实数据驱动版）
模型输入：7维状态 [torque, speed, accelerate, accX, accY, accZ, last_motor_torque]
历史帧长度：SEQ_LEN = 7
输出：电机助力力矩 (0~38 Nm)
"""
import torch
import torch.nn as nn
import numpy as np
import os

# ==================== 与训练代码保持一致的超参数 ====================
SEQ_LEN = 7                     # 历史帧长度
D_MODEL = 256                   # Transformer 隐层维度
NHEAD = 4                       # 多头注意力头数
NUM_TF_LAYERS = 1               # Transformer Encoder 层数
DIM_FEEDFORWARD = 512           # FFN 中间层维度
TF_DROPOUT = 0.1                # Transformer Dropout
MOTOR_TORQUE_MAX = 38.0         # 电机最大输出力矩 (Nm)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 定义 Actor 网络（与训练代码完全一致）====================
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ActorTransformer(nn.Module):
    """Actor专用Transformer网络（与训练代码结构一致）"""
    def __init__(self, state_dim=7, action_dim=1, seq_len=SEQ_LEN):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FEEDFORWARD,
            dropout=TF_DROPOUT, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_TF_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.actor_mean = nn.Linear(D_MODEL, action_dim)
        # 可学习 log_std（训练时存在，测试时不需要但为了加载权重必须定义）
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x_seq):
        # x_seq: [batch, seq_len, state_dim]
        x = self.input_proj(x_seq)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)                     # [batch, d_model]
        mean_raw = self.actor_mean(pooled)
        mean = torch.relu(mean_raw)                # 非负
        mean = torch.clamp(mean, 0.0, MOTOR_TORQUE_MAX)
        # 测试时只返回 mean，但为了结构完整，也返回 std（不使用）
        std = torch.exp(self.actor_log_std).expand_as(mean)
        return mean, std, pooled

# ==================== 辅助函数：构造测试输入 ====================
def build_test_input(torque, speed, last_motor=0.0):
    """
    根据当前人力矩和车速，构造完整的7维状态，并扩展为7步历史序列。
    其他固定取值：
        accelerate = 0.0        （假设匀速）
        accX = -0.7              （训练环境中的典型静态值）
        accY = 0.7
        accZ = 0.15
        last_motor_torque = last_motor （测试时通常为0）
    返回: tensor (1, SEQ_LEN, 7)
    """
    state = np.array([
        torque,           # 人力矩 (Nm)
        speed,            # 车速 (m/s)
        0.0,              # accelerate (m/s²)
        -0.7,             # accX
        0.7,              # accY
        0.15,             # accZ
        last_motor        # 上一时刻实际电机力矩
    ], dtype=np.float32)
    # 将当前状态重复 SEQ_LEN 次，构造历史序列
    seq = np.repeat(state[np.newaxis, :], SEQ_LEN, axis=0)   # (SEQ_LEN, 7)
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==================== 测试单个模型 ====================
def test_model(model_path, output_file, torque_range, speed_range):
    print(f"\n测试模型: {model_path} -> {output_file}")
    if not os.path.exists(model_path):
        print(f"模型文件不存在，跳过: {model_path}")
        return

    model = ActorTransformer(state_dim=7, action_dim=1).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    # 训练保存的字典键为 'actor'
    if 'actor' not in checkpoint:
        print(f"警告: 模型文件中没有 'actor' 键，尝试直接加载整个字典")
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint['actor'], strict=False)
    model.eval()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# torque(Nm) speed(m/s) assist_torque(Nm)\n")
        for torque in torque_range:
            for speed in speed_range:
                inp = build_test_input(torque, speed, last_motor=0.0)
                with torch.no_grad():
                    mean, _, _ = model(inp)
                    out = mean.item()
                f.write(f"{torque:.3f} {speed:.3f} {out:.3f}\n")
    print(f"结果已保存至 {output_file}")

# ==================== 主函数 ====================
if __name__ == "__main__":
    # 测试网格
    torque_min, torque_max = 0.0, 100.0
    speed_min, speed_max = 0.0, 13.0
    torque_step = 10.0
    speed_step = 1.0
    torque_range = np.arange(torque_min, torque_max + torque_step/2, torque_step)
    speed_range = np.arange(speed_min, speed_max + speed_step/2, speed_step)

    # 模型文件路径（可根据实际名称修改）
    model_files = {
        "ppo_ebike_model_realdata_v2_best.pth": "test_results_best.txt",
        "ppo_ebike_model_realdata_v2.pth": "test_results_final.txt"
    }

    for model_path, out_file in model_files.items():
        test_model(model_path, out_file, torque_range, speed_range)