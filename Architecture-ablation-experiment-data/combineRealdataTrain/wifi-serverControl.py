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
import numpy as np
import os
from datetime import datetime
from collections import deque

# ==================== 与 train_ppo_ebike_model.py 完全一致的网络参数 ====================
SEQ_LEN = 7                     # 历史帧长度（7步）
D_MODEL = 256                   # Transformer 隐层维度
NHEAD = 4                       # 多头注意力头数
NUM_TF_LAYERS = 1               # Transformer Encoder 层数（与训练一致）
DIM_FEEDFORWARD = 512           # FFN 中间层维度
TF_DROPOUT = 0.1                # Dropout
MOTOR_TORQUE_MAX = 38.0         # 电机最大输出力矩 (Nm)

# 传感器量程转换参数
TORQUE_MAX_SENSOR = 100.9       # 踏板力矩传感器最大值 (Nm)
TORQUE_MIN_VOLT = 1.45          # 力矩传感器电压下限 (V)
TORQUE_MAX_VOLT = 3.3           # 力矩传感器电压上限 (V)
SPEED_MAX_SENSOR = 13.0         # 最大车速 (m/s)
SPEED_MIN_VOLT = 0.0            # 速度电压下限 (V)
SPEED_MAX_VOLT = 3.3            # 速度电压上限 (V)

# 加速度传感器范围（单位：g，直接使用）
ACC_MIN = -10.0
ACC_MAX = 10.0

# ==================== 网络定义（与训练代码完全一致） ====================
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
    """Actor专用Transformer网络（与训练代码一致）"""
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
        # 以下为训练时需要但推理时不用的参数（保证加载权重不报错）
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x_seq):
        # x_seq: [batch, seq_len, state_dim]
        x = self.input_proj(x_seq)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        mean_raw = self.actor_mean(pooled)
        mean = torch.relu(mean_raw)                # 非负
        mean = torch.clamp(mean, 0.0, MOTOR_TORQUE_MAX)
        return mean                                 # 推理时只返回动作均值

# ==================== 序列填充工具 ====================
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
        self.root.title("电动助力车 AI 控制器（新7维模型） - TCP 通信")
        self.root.geometry("700x600")
        self.root.resizable(True, True)

        # TCP 连接参数（默认 IP 改为 172.20.10.11）
        self.server_ip = "172.20.10.11"
        self.server_port = 8080
        self.sock = None
        self.is_connected = False
        self.rx_thread = None
        self.rx_queue = queue.Queue()

        # 模型相关
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 时序缓冲区：保存最近 SEQ_LEN 个 7 维状态
        self.obs_seq_buffer = deque(maxlen=SEQ_LEN)

        # 状态变量（用于构造观测中的 last_motor_torque）
        self.last_motor_torque = 0.0          # 上一时刻实际电机力矩 (Nm)

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
        """加载训练好的 PPO+Transformer 模型（7维输入）"""
        model_path = "ppo_ebike_model_realdata_v2.pth"
        if not os.path.exists(model_path):
            # 尝试备选名称
            alt_path = "ppo_ebike_model_realdata_v2_best.pth"
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                messagebox.showerror("模型错误", f"模型文件未找到: {model_path}\n请将训练好的模型放置于当前目录。")
                return
        try:
            self.model = ActorTransformer(state_dim=7, action_dim=1, seq_len=SEQ_LEN).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            # 训练保存的键为 'actor'
            if 'actor' in checkpoint:
                self.model.load_state_dict(checkpoint['actor'], strict=True)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            print(f"✅ 模型加载成功: {model_path}, 设备: {self.device}, 历史帧数: {SEQ_LEN}")
        except Exception as e:
            self.model = None
            messagebox.showerror("模型错误", f"加载模型失败: {e}")

    def setup_logging(self):
        """创建日志目录并打开日志文件"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        now = datetime.now()
        try:
            filename = now.strftime("%Y%m%d-%H%M%S.txt")
        except ValueError:
            filename = now.strftime("%Y%m%d-%H.%M.%S.txt")
        log_path = os.path.join(log_dir, filename)
        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_file.write(f"日志开始时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write("=" * 50 + "\n")
            self.log_file.flush()
        except Exception as e:
            print(f"无法创建日志文件: {e}")
            self.log_file = None

    def write_log(self, message):
        if self.log_file:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log_file.write(f"[{timestamp}] {message}\n")
                self.log_file.flush()
            except:
                pass

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

        model_status = "✅ 模型已加载(7维Transformer)" if self.model else "❌ 模型未加载"
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
            self.sock.settimeout(None)
            self.is_connected = True
            self.connect_btn.config(text="断开")
            self.status_var.set(f"已连接 {ip}:{port}")
            # 重置时序缓冲区和状态变量
            self.obs_seq_buffer.clear()
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
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        self.rx_queue.put(f"[收] {line}")
                        if line.startswith("SENSOR_DATA_"):
                            data_part = line[len("SENSOR_DATA_"):]
                            if self.model:
                                self.process_and_reply(data_part)
            except socket.timeout:
                continue
            except Exception as e:
                self.rx_queue.put(f"[错误] {e}")
                break
        self.disconnect()

    def process_and_reply(self, data_part):
        """
        处理传感器数据并发送控制指令（格式：AIRESULT_电压值）
        优先判断手动模式；否则使用模型推理。
        """
        # 手动模式优先
        if self.manual_control_var.get():
            try:
                manual_val = float(self.manual_voltage_var.get())
                if 0.0 <= manual_val <= 3.6:
                    dac_voltage = np.clip(manual_val, 0.0, 3.3)
                    cmd = f"AIRESULT_{dac_voltage:.3f}\r\n"
                    self.send_tcp(cmd)
                    self.rx_queue.put(f"[发] {cmd.strip()} (手动)")
                    return
                else:
                    self.rx_queue.put(f"[警告] 手动电压超出范围，不发送")
                    return
            except ValueError:
                self.rx_queue.put(f"[警告] 手动电压无效，不发送")
                return

        # 自动模式：模型推理
        try:
            parts = data_part.split(',')
            if len(parts) < 5:
                raise ValueError("数据字段不足，应为: 力矩电压,速度电压,accX,accY,accZ")

            torque_voltage = float(parts[0])
            speed_voltage = float(parts[1])
            accX = float(parts[2])
            accY = float(parts[3])
            accZ = float(parts[4])

            # 数据有效性检查
            if not (TORQUE_MIN_VOLT <= torque_voltage <= TORQUE_MAX_VOLT):
                self.rx_queue.put(f"[警告] 力矩电压越界: {torque_voltage:.3f}V，忽略")
                return
            speed_voltage = np.clip(speed_voltage, SPEED_MIN_VOLT, SPEED_MAX_VOLT)
            accX = np.clip(accX, ACC_MIN, ACC_MAX)
            accY = np.clip(accY, ACC_MIN, ACC_MAX)
            accZ = np.clip(accZ, ACC_MIN, ACC_MAX)

            # 电压转物理量
            torque_nm = (torque_voltage - TORQUE_MIN_VOLT) / (TORQUE_MAX_VOLT - TORQUE_MIN_VOLT) * TORQUE_MAX_SENSOR
            torque_nm = np.clip(torque_nm, 0.0, TORQUE_MAX_SENSOR)

            speed_mps = speed_voltage / SPEED_MAX_VOLT * SPEED_MAX_SENSOR
            speed_mps = np.clip(speed_mps, 0.0, SPEED_MAX_SENSOR)

            # 注意：accelerate 直接使用 accX（与训练要求一致）
            accelerate = accX

            # 构造当前7维状态
            current_state = np.array([
                torque_nm,           # 人力矩 (Nm)
                speed_mps,           # 车速 (m/s)
                accelerate,          # 纵向加速度 (m/s²)
                accX,                # X轴加速度 (g)
                accY,                # Y轴加速度 (g)
                accZ,                # Z轴加速度 (g)
                self.last_motor_torque   # 上一时刻实际电机力矩 (Nm)
            ], dtype=np.float32)

            # 更新历史缓冲区
            self.obs_seq_buffer.append(current_state)

            # 填充至固定长度 SEQ_LEN
            obs_seq = pad_sequence(self.obs_seq_buffer, SEQ_LEN)   # shape: (SEQ_LEN, 7)

            # 转换为模型输入 (batch=1, seq_len, 7)
            inp = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                motor_torque_nm = self.model(inp).item()

            motor_torque_nm = np.clip(motor_torque_nm, 0.0, MOTOR_TORQUE_MAX)

            # 扭矩转电压 (0~38Nm -> 0~3.3V)
            dac_voltage = motor_torque_nm / MOTOR_TORQUE_MAX * 3.3
            dac_voltage = np.clip(dac_voltage, 0.0, 3.3)

            # 发送命令
            cmd = f"AIRESULT_{dac_voltage:.3f}\r\n"
            self.send_tcp(cmd)

            # 更新上一时刻电机力矩（用于下一帧）
            self.last_motor_torque = motor_torque_nm

            # 记录调试信息
            debug_info = (f"[推理] 人力矩={torque_nm:.2f}Nm, 速度={speed_mps:.2f}m/s, "
                          f"accX={accX:.2f}g, 输出扭矩={motor_torque_nm:.2f}Nm -> 电压={dac_voltage:.3f}V")
            self.rx_queue.put(debug_info)
            self.write_log(debug_info)

        except Exception as e:
            err_msg = f"[推理错误] {e}"
            self.rx_queue.put(err_msg)
            self.write_log(err_msg)

    def send_tcp(self, data):
        """通过 TCP 发送数据"""
        if self.is_connected and self.sock:
            try:
                self.sock.sendall(data.encode('utf-8'))
            except Exception as e:
                self.status_var.set(f"发送失败: {e}")
                self.disconnect()

    def send_data(self, event=None):
        """手动发送（调试用）"""
        if not self.is_connected:
            self.status_var.set("未连接，无法发送")
            return
        raw = self.tx_entry.get().strip()
        if not raw:
            return
        try:
            float(raw)
            cmd = f"AIRESULT_{raw}\r\n"
        except ValueError:
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
            self.log_file.write("\n" + "=" * 50 + "\n")
            self.log_file.write(f"日志结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.close()
        self.disconnect()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TCPGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()