#ifndef __ESP8266_H
#define __ESP8266_H	 

#include "led.h"
#include "string.h"
#include "usart.h"

// ========== WiFi 配置（请根据实际情况修改）==========
//#define SSID       "YP-DL"          // 要连接的路由器名称
//#define PASS       "18655192192"    // 路由器密码

#define SSID       "iPhone15mac"          // 要连接的路由器名称
#define PASS       "12345678"    // 路由器密码

// ========== 静态 IP 配置（固定 IP 地址）==========
#define STATIC_IP   "172.20.10.11" // 固定 IP 地址
#define GATEWAY     "172.20.10.15"   // 网关（通常为路由器 IP）
#define NETMASK     "255.255.255.240" // 子网掩码

// ========== TCP 客户端模式配置（本代码未使用，仅作预留）==========
#define Sever_IP   "192.168.1.103"  // 目标服务器 IP（仅客户端模式使用）
#define Sever_Port "8080"

#define ESP_Receive_Buff   Usart1_RxBuff
#define ESP8266_USART      Usart1_Init
#define ESP8266_Rx_OK      Usart1_Rx_ok

#define RESET_IO(x)    GPIO_WriteBit(GPIOA, GPIO_Pin_5, (BitAction)x)

void ESP_RST_Init(void);
void Handle_ESPSever_Task(void);
void Handle_ESPClient_Task(void);
char ESP8266_TcpClient(void);
char ESP8266_TcpSever(void);
void ESP8266_SendToClient(uint8_t id, char *data);

#endif