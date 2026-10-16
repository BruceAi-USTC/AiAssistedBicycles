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

# ==================== 与 train_ppo_ebike_model.py 完全一致的 Actor 网络（全连接版）====================
# 训练脚本中的参数（必须一致）
HIDDEN_SIZE = 512             # 隐藏层大小（与训练代码 HIDDEN_SIZE 一致）
NUM_LAYERS = 4                # 隐藏层数量（与训练代码 NUM_LAYERS 一致）
MOTOR_TORQUE_MAX = 38.0       # 电机最大输出力矩 (Nm)

class ActorNetwork(nn.Module):
    """
    独立 Actor 网络（全连接，无时序）
    输入: (batch, state_dim)   state_dim=2 (人力矩, 车速)
    输出: (batch, action_dim) 电机助力力矩均值 (Nm)，范围 [0, MOTOR_TORQUE_MAX]
    结构与 train_ppo_ebike_model.py 中的 ActorNetwork 完全一致。
    """
    def __init__(self, state_dim=2, action_dim=1):
        super().__init__()
        layers = []
        in_dim = state_dim
        for _ in range(NUM_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_SIZE))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(HIDDEN_SIZE))
            in_dim = HIDDEN_SIZE
        self.feature_net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(HIDDEN_SIZE, action_dim)

        # 初始化（与训练代码一致）
        nn.init.orthogonal_(self.mean_head.weight, gain=0.5)
        nn.init.constant_(self.mean_head.bias, 19.0)
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # 可学习对数标准差（测试时不会用到，但为兼容加载保留）
        self.log_std = nn.Parameter(torch.ones(action_dim) * 1.609)

    def forward(self, x):
        """
        x: 输入形状 (batch, state_dim)
        返回动作均值 (batch, action_dim)
        """
        features = self.feature_net(x)          # (batch, HIDDEN_SIZE)
        mean_raw = self.mean_head(features)     # (batch, action_dim)
        mean = torch.clamp(mean_raw, 0.0, MOTOR_TORQUE_MAX)
        return mean

# ==================== TCP GUI 主类 ====================
class TCPGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("STM32 WiFi 控制端（TCP 通信）- 全连接模型（当前帧输入）")
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
        """加载训练好的全连接 Actor 模型（最佳模型或最终模型）"""
        # 根据训练脚本 train_ppo_ebike_model.py 保存的文件名
        model_path = "ppo_ebike_model_architecture_fixed_best.pth"   # 训练生成的最佳模型
        if not os.path.exists(model_path):
            model_path = "ppo_ebike_model_architecture_fixed.pth"    # 最终模型
        if not os.path.exists(model_path):
            # 兼容旧命名
            model_path = "ppo_ebike_model_architecture_fixed.pth"
        try:
            self.model = ActorNetwork(state_dim=2, action_dim=1).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            # 训练时保存的字典键为 'actor'
            if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                self.model.load_state_dict(checkpoint['actor'])
            else:
                # 兼容直接保存 state_dict 的情况
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"✅ 全连接模型加载成功，使用设备: {self.device}")
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
        log_dir = os.path.join(script_dir, "log")
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

        model_status = "✅ 模型已加载(全连接)" if self.model else "❌ 模型未加载"
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
        否则进行传感器数据校验和模型推理（使用当前时刻 [人力矩, 车速]）。
        """
        # ----------------- 手动模式优先 -----------------
        if self.manual_control_var.get():
            try:
                manual_val = float(self.manual_voltage_var.get())
                if 0.0 <= manual_val <= 3.6:
                    dac_voltage = np.clip(manual_val, 0.0, 3.3)
                    cmd = f"AIRESULT_{dac_voltage:.3f}\r\n"
                    self.send_tcp(cmd)
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

            print(f"[DEBUG] 映射: 力矩={torque_nm:.3f} Nm, 速度={speed_mps:.3f} m/s")

            # 构造当前观测 [人力矩, 车速] (2维)
            current_obs = np.array([[torque_nm, speed_mps]], dtype=np.float32)  # shape (1,2)
            inp = torch.tensor(current_obs, dtype=torch.float32).to(self.device)

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