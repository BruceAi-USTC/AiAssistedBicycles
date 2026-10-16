#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import socket
import threading
import queue
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
from collections import deque

# ==================== 与 train_ppo_ebike_model.py 完全一致的 Actor 网络（Mamba时序版） ====================
# 训练脚本中的参数（必须一致）
ACTOR_SEQ_LEN = 5               # 历史帧长度（最近5步）
HIDDEN_SIZE = 512               # MLP 特征提取层隐藏单元数
NUM_LAYERS = 3                  # MLP 特征提取层数
MOTOR_TORQUE_MAX = 38.0         # 电机最大输出力矩 (Nm)

# 简化版 Mamba 模块（与训练代码完全一致）
class SimpleMamba(nn.Module):
    """
    简化版 Mamba 模块，仅包含核心的选择性状态空间机制
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
        # 状态空间参数
        self.A_param = nn.Parameter(torch.randn(hidden_dim))
        self.B_param = nn.Parameter(torch.randn(hidden_dim))
        self.C_param = nn.Parameter(torch.randn(hidden_dim))
        self.D_param = nn.Parameter(torch.randn(output_dim))
        
        # 输入门控
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.z_proj = nn.Linear(input_dim, hidden_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()
        
        # 重塑为 [batch*seq_len, input_dim]
        x_reshaped = x.reshape(-1, input_dim)
        
        # 投影到隐藏空间
        x_hid = self.in_proj(x_reshaped)
        x_hid = F.silu(x_hid)  # Swish/SiLU激活
        
        # 重塑回序列形式
        x_hid = x_hid.view(batch_size, seq_len, self.hidden_dim)
        
        # 沿时间维度处理
        outputs = []
        for t in range(seq_len):
            x_t = x_hid[:, t, :]  # [batch, hidden_dim]
            
            # 简化的状态更新
            A = -torch.abs(self.A_param)  # 确保稳定性
            B = self.B_param
            C = self.C_param
            
            # 计算当前时刻的输出
            h_t = x_t * torch.sigmoid(B * x_t)  # 简化的状态变换
            y_t = h_t * C + x_t * self.D_param  # 输出计算
            
            outputs.append(y_t)
        
        # 堆叠输出
        stacked_outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # 最终投影到输出维度
        final_output = self.out_proj(stacked_outputs)
        
        return final_output  # 返回整个序列 [batch, seq_len, output_dim]


class ActorNetwork(nn.Module):
    """
    Mamba时序策略网络，与 train_ppo_ebike_model.py 中的 ActorNetwork 完全一致。
    输入: (batch, seq_len, state_dim)  其中 seq_len=5, state_dim=3 (人力矩, 车速, 上一时刻实际电机力矩)
    输出: (batch, action_dim) 电机助力力矩均值 (Nm)
    """
    def __init__(self, state_dim=3, action_dim=1):
        super().__init__()
        self.seq_len = ACTOR_SEQ_LEN
        self.state_dim = state_dim
        
        # 单步特征提取 MLP
        layers = []
        in_dim = state_dim
        for i in range(NUM_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_SIZE))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(HIDDEN_SIZE))
            in_dim = HIDDEN_SIZE
        self.feature_net = nn.Sequential(*layers)
        
        # Mamba 模块处理时序，隐藏层减半以控制参数量
        mamba_hidden = HIDDEN_SIZE // 2
        self.mamba = SimpleMamba(HIDDEN_SIZE, mamba_hidden, mamba_hidden)
        
        # 输出层
        self.mean_head = nn.Linear(mamba_hidden, action_dim)
        # 初始化（与训练一致：偏置 32.0）
        nn.init.orthogonal_(self.mean_head.weight, gain=0.5)
        nn.init.constant_(self.mean_head.bias, 32.0)
        
        # 可学习对数标准差（测试时不会用到，但为兼容加载保留）
        self.log_std = nn.Parameter(torch.ones(action_dim) * 2.0)

    def forward(self, x_seq):
        """
        x_seq: 输入形状 (batch, seq_len, state_dim)
        返回动作均值（期望的助力力矩），形状 (batch, action_dim)
        """
        batch_size, seq_len, state_dim = x_seq.shape
        
        # 提取每步特征
        x_flat = x_seq.view(batch_size * seq_len, state_dim)
        features = self.feature_net(x_flat)
        features = features.view(batch_size, seq_len, -1)
        
        # Mamba 时序融合
        mamba_out = self.mamba(features)            # [batch, seq_len, mamba_hidden]
        last_hidden = mamba_out[:, -1, :]           # [batch, mamba_hidden]
        
        mean_raw = self.mean_head(last_hidden)
        # 硬限幅到 [0, MOTOR_TORQUE_MAX]
        mean = torch.clamp(mean_raw, 0.0, MOTOR_TORQUE_MAX)
        return mean


# ==================== 序列填充工具函数 ====================
def pad_sequence(seq_list, length, pad_value=None):
    """将序列填充到固定长度，不足时头部重复填充第一帧"""
    seq = list(seq_list)
    if len(seq) == 0:
        raise ValueError("序列不能为空")
    if pad_value is None:
        pad_value = seq[0].copy()
    while len(seq) < length:
        seq.insert(0, pad_value.copy())
    return np.array(seq[-length:], dtype=np.float32)


# ==================== TCP GUI 主类 ====================
class TCPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STM32 WiFi 控制端（TCP 通信）- Mamba时序模型（历史5帧，3维观测）")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # TCP 连接参数
        self.server_ip = "192.168.1.211"
        self.server_port = 8080
        self.sock = None
        self.is_connected = False
        self.rx_thread = None
        self.rx_queue = queue.Queue()

        # 模型加载
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 时序缓冲区：保存最近 ACTOR_SEQ_LEN 个观测 [人力矩(Nm), 车速(m/s), 上一时刻实际电机力矩(Nm)]
        self.actor_seq_buffer = deque(maxlen=ACTOR_SEQ_LEN)
        # 上一次输出的电机力矩（用于下一帧的第三个输入）
        self.last_motor_torque = 0.0
        self.load_model()

        # 手动控制变量
        self.manual_control_var = tk.BooleanVar(value=False)
        self.manual_voltage_var = tk.StringVar(value="0.0")

        # 日志文件
        self.log_file = None
        self.setup_logging()

        self.create_widgets()
        self.update_receive_area()

    def load_model(self):
        """加载训练好的 Mamba 模型（优先使用 ppo_ebike_model_gru_architecture.pth，兼容其他命名）"""
        # 根据用户要求，优先使用 ppo_ebike_model_gru_architecture.pth
        model_path = "ppo_ebike_model_mamba_architecture_best.pth"
        
        try:
            self.model = ActorNetwork(state_dim=3, action_dim=1).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            # 训练时保存的字典键为 'actor'
            if 'actor' in checkpoint:
                self.model.load_state_dict(checkpoint['actor'])
            else:
                # 兼容直接保存整个模型的情况
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"✅ Mamba 模型加载成功，使用设备: {self.device}，历史帧长度: {ACTOR_SEQ_LEN}")
            print(f"   模型文件: {model_path}")
        except FileNotFoundError:
            self.model = None
            error_msg = f"模型文件未找到: {model_path}\n请确保文件存在于当前目录。"
            print(error_msg)
            messagebox.showerror("模型错误", error_msg)
        except Exception as e:
            self.model = None
            error_msg = f"模型加载失败: {e}\n请检查模型文件是否正确。"
            print(error_msg)
            messagebox.showerror("模型错误", error_msg)

    def setup_logging(self):
        """创建log目录并打开日志文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "log_best")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            print(f"[LOG] 创建日志目录: {log_dir}")

        now = datetime.now()
        try:
            filename = now.strftime("%Y%m%d-%-H.%-M-%S.txt")
        except ValueError:
            filename = now.strftime("%Y%m%d-%H.%M-%S.txt")
        log_path = os.path.join(log_dir, filename)

        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            print(f"[LOG] 日志文件已创建: {log_path}")
            self.log_file.write(f"日志开始时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 50 + "\n")
            self.log_file.flush()
        except Exception as e:
            print(f"[LOG] 无法创建日志文件: {e}")
            self.log_file = None

    def write_log(self, message):
        if self.log_file:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_file.write(f"[{timestamp}] {message}\n")
                self.log_file.flush()
            except Exception as e:
                print(f"[LOG] 写入日志失败: {e}")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=2)

        ttk.Label(ctrl_frame, text="目标 IP:").pack(side=tk.LEFT, padx=5)
        self.ip_entry = ttk.Entry(ctrl_frame, width=15)
        self.ip_entry.insert(0, self.server_ip)
        self.ip_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(ctrl_frame, text="端口:").pack(side=tk.LEFT, padx=5)
        self.port_entry = ttk.Entry(ctrl_frame, width=6)
        self.port_entry.insert(0, str(self.server_port))
        self.port_entry.pack(side=tk.LEFT, padx=5)

        self.connect_btn = ttk.Button(ctrl_frame, text="连接", command=self.toggle_connection, width=8)
        self.connect_btn.pack(side=tk.LEFT, padx=10)

        model_status = "✅ 模型已加载(Mamba)" if self.model else "❌ 模型未加载"
        ttk.Label(ctrl_frame, text=model_status, foreground="green" if self.model else "red").pack(side=tk.LEFT, padx=10)

        # 手动控制区域
        manual_frame = ttk.Frame(main_frame)
        manual_frame.pack(fill=tk.X, pady=5)
        self.manual_checkbox = ttk.Checkbutton(manual_frame, text="手动控制电压", variable=self.manual_control_var,
                                               command=self.on_manual_toggle)
        self.manual_checkbox.pack(side=tk.LEFT, padx=5)
        ttk.Label(manual_frame, text="电压 (0~3.3 V):").pack(side=tk.LEFT, padx=5)
        self.manual_entry = ttk.Entry(manual_frame, textvariable=self.manual_voltage_var, width=8, state='disabled')
        self.manual_entry.pack(side=tk.LEFT, padx=5)

        rx_frame = ttk.LabelFrame(main_frame, text="接收区", padding="5")
        rx_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.rx_text = scrolledtext.ScrolledText(rx_frame, wrap=tk.WORD, height=15)
        self.rx_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(rx_frame, text="清空接收", command=self.clear_rx).pack(pady=2)

        tx_frame = ttk.LabelFrame(main_frame, text="手动发送（调试）", padding="5")
        tx_frame.pack(fill=tk.X, pady=5)

        self.tx_entry = ttk.Entry(tx_frame)
        self.tx_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.tx_entry.bind('<Return>', self.send_data)

        ttk.Button(tx_frame, text="发送", command=self.send_data, width=8).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("未连接")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=2)

    def on_manual_toggle(self):
        if self.manual_control_var.get():
            self.manual_entry.config(state='normal')
        else:
            self.manual_entry.config(state='disabled')

    def toggle_connection(self):
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        ip = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            self.status_var.set("端口号无效")
            return
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2)
            self.sock.connect((ip, port))
            self.sock.settimeout(None)   # 阻塞模式
            self.is_connected = True
            self.connect_btn.config(text="断开")
            self.status_var.set(f"已连接 {ip}:{port}")
            # 重置时序缓冲区和上一时刻力矩
            self.actor_seq_buffer.clear()
            self.last_motor_torque = 0.0
            self.rx_thread = threading.Thread(target=self.rx_worker, daemon=True)
            self.rx_thread.start()
        except Exception as e:
            self.status_var.set(f"连接失败: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None

    def disconnect(self):
        self.is_connected = False
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        self.connect_btn.config(text="连接")
        self.status_var.set("未连接")

    def rx_worker(self):
        """接收线程：读取TCP数据，按行解析"""
        buffer = ""
        while self.is_connected and self.sock:
            try:
                data = self.sock.recv(1024).decode('utf-8', errors='ignore')
                if not data:
                    # 连接已关闭
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        print(f"[DEBUG] 收到: {line}")
                        self.rx_queue.put(f"[收] {line}")
                        if line.startswith("SENSOR_DATA_"):
                            data_part = line[len("SENSOR_DATA_"):]
                            if self.model:
                                self.process_and_reply(data_part)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[DEBUG] 接收异常: {e}")
                self.rx_queue.put(f"[错误] {e}")
                break
        # 线程结束，断开连接
        self.disconnect()

    def process_and_reply(self, data_part):
        """
        处理传感器数据并发送控制指令（格式：AIRESULT_电压值）
        优先判断手动模式：若开启手动且电压有效，则直接发送手动电压。
        否则进行传感器数据校验和模型推理（使用历史5帧 [人力矩, 车速, 上一时刻实际电机力矩]）。
        """
        # ----------------- 手动模式优先 -----------------
        if self.manual_control_var.get():
            try:
                manual_val = float(self.manual_voltage_var.get())
                if 0.0 <= manual_val <= 3.6:
                    dac_voltage = np.clip(manual_val, 0.0, 3.3)
                    cmd = f"AIRESULT_{dac_voltage:.3f}\r\n"
                    self.send_tcp(cmd)
                    # 手动模式下不更新 last_motor_torque，因为未使用模型
                    print(f"[DEBUG] 手动模式：使用用户设定电压 {dac_voltage:.3f} V")
                    self.rx_queue.put(f"[发] {cmd.strip()}")
                    return
                else:
                    print(f"[DEBUG] 手动电压超出范围 (0~3.6)，本次不发送")
                    return
            except ValueError:
                print(f"[DEBUG] 手动电压无效，本次不发送")
                return

        # ----------------- 自动模式（模型推理） -----------------
        try:
            parts = data_part.split(',')
            if len(parts) < 2:
                raise ValueError("格式错误，应为 '力矩电压,速度电压'")

            torque_voltage = float(parts[0])
            speed_voltage = float(parts[1])

            # 数据有效性检查（力矩电压1.45~3.3，速度电压0~3.3）
            if not (1.45 <= torque_voltage <= 3.3):
                print(f"[DEBUG] 非法力矩电压: {torque_voltage:.3f} V (允许 1.45~3.3)，忽略")
                return
            speed_voltage = np.clip(speed_voltage, 0.0, 3.3)

            # 电压转物理量（与训练环境一致）
            torque_nm = (torque_voltage - 1.45) / (3.3 - 1.45) * 100.9
            speed_mps = speed_voltage / 3.3 * 13.0
            torque_nm = np.clip(torque_nm, 0.0, 100.9)
            speed_mps = np.clip(speed_mps, 0.0, 13.0)

            print(f"[DEBUG] 映射: 力矩={torque_nm:.3f} Nm, 速度={speed_mps:.3f} m/s, 上一时刻电机力矩={self.last_motor_torque:.3f} Nm")

            # 构造当前观测 [人力矩, 车速, 上一时刻实际电机力矩] (3维)
            current_obs = np.array([torque_nm, speed_mps, self.last_motor_torque], dtype=np.float32)

            # 更新时序缓冲区
            self.actor_seq_buffer.append(current_obs)

            # 填充到固定长度 ACTOR_SEQ_LEN，不足时复制第一帧
            actor_seq = pad_sequence(self.actor_seq_buffer, ACTOR_SEQ_LEN)  # (L, 3)

            # 转换为 (batch=1, seq_len=L, state_dim=3) 形状
            inp = torch.tensor(actor_seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, L, 3)

            with torch.no_grad():
                motor_torque_nm = self.model(inp).item()   # 输出标量

            motor_torque_nm = np.clip(motor_torque_nm, 0.0, MOTOR_TORQUE_MAX)
            print(f"[DEBUG] 模型输出扭矩: {motor_torque_nm:.3f} Nm")

            # 扭矩转电压（0~38Nm -> 0~3.3V）
            dac_voltage = motor_torque_nm / MOTOR_TORQUE_MAX * 3.3
            dac_voltage = np.clip(dac_voltage, 0.0, 3.3)

            cmd = f"AIRESULT_{dac_voltage:.3f}\r\n"
            self.send_tcp(cmd)
            print(f"[DEBUG] 发送命令: {cmd.strip()}")
            self.rx_queue.put(f"[发] {cmd.strip()}")

            # 更新上一时刻电机力矩，供下一帧使用
            self.last_motor_torque = motor_torque_nm

        except Exception as e:
            print(f"[DEBUG] 处理数据异常: {e}")
            self.rx_queue.put(f"[推理错误] {e}")

    def send_tcp(self, data):
        """通过 TCP 发送数据"""
        if self.is_connected and self.sock:
            try:
                self.sock.sendall(data.encode('utf-8'))
            except Exception as e:
                self.status_var.set(f"发送失败: {e}")
                self.disconnect()

    def send_data(self, event=None):
        """手动发送：支持纯数字（自动加 AIRESULT_ 前缀）或开关命令 A/B 原样发送"""
        if not self.is_connected:
            self.status_var.set("未连接，无法发送")
            return
        raw = self.tx_entry.get().strip()
        if not raw:
            return

        # 如果输入是数字（可带小数点），自动加上 AIRESULT_ 前缀
        try:
            float(raw)   # 尝试转换为浮点数
            cmd = f"AIRESULT_{raw}\r\n"
        except ValueError:
            # 不是纯数字，保持原样（如 A, B 等），并自动补换行
            cmd = raw
            if not cmd.endswith('\n'):
                cmd += '\n'

        self.send_tcp(cmd)
        self.tx_entry.delete(0, tk.END)
        msg = f"[手动发送] {cmd.strip()}"
        self.rx_text.insert(tk.END, msg + '\n')
        self.rx_text.see(tk.END)
        self.write_log(msg)

    def update_receive_area(self):
        try:
            while True:
                line = self.rx_queue.get_nowait()
                self.rx_text.insert(tk.END, line + '\n')
                self.rx_text.see(tk.END)
                self.write_log(line)
        except queue.Empty:
            pass
        finally:
            self.root.after(5, self.update_receive_area)

    def clear_rx(self):
        self.rx_text.delete(1.0, tk.END)

    def on_closing(self):
        if self.log_file:
            try:
                self.log_file.write("\n" + "=" * 50 + "\n")
                self.log_file.write(f"日志结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.close()
                print("[LOG] 日志文件已关闭")
            except Exception as e:
                print(f"[LOG] 关闭日志文件错误: {e}")
        self.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = TCPGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()