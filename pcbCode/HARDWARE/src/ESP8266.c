#include "esp8266.h"
#include "usart3.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* 引用 main.c 中的全局变量和函数 */
extern volatile float g_motor_target;
extern volatile uint32_t g_last_cmd_time;
extern volatile uint8_t system_enabled;
extern uint32_t sys_tick;
extern void SetDACVoltage(float voltage);   // 声明外部函数

/* 去除字符串首尾空白字符 */
static void trim(char *str)
{
    char *start = str;
    char *end;
    while(isspace((unsigned char)*start)) start++;
    if(*start == 0) {
        str[0] = '\0';
        return;
    }
    end = start + strlen(start) - 1;
    while(end > start && isspace((unsigned char)*end)) end--;
    *(end+1) = '\0';
    memmove(str, start, end - start + 2);
}

void ESP_RST_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_5;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    GPIO_SetBits(GPIOA, GPIO_Pin_5);
}

/* 通过WiFi发送数据到指定客户端（多连接模式下） */
void ESP8266_SendToClient(uint8_t id, char *data)
{
    char cmd[32];
    uint16_t len = strlen(data);
    sprintf(cmd, "AT+CIPSEND=%d,%d\r\n", id, len);
    my_printf(USART1, "%s", cmd);
    my_printf(USART3, "[WiFi Send] %s", cmd);
    delay_ms(80);
    my_printf(USART1, "%s", data);
    my_printf(USART3, "[WiFi Data] %s", data);
    delay_ms(20);
}


/* 发送AT指令并等待指定应答 */
unsigned char ESP8266_send_cmd(char *cmd, int wait, char *ack)
{
    delay_ms(10);
    memset(ESP_Receive_Buff, 0, sizeof(ESP_Receive_Buff));
    my_printf(USART1, "%s\r\n", cmd);
    while(wait--)
    {
        if(strstr((char *)ESP_Receive_Buff, (char *)ack) != NULL)
            break;
        delay_ms(500);
    }
    my_printf(USART3, "\r\n");
    if(wait <= 0)
        return 1;
    else
        return 0;
}

/* 处理WiFi接收的命令（只处理 +IPD 的TCP数据） */
void Handle_ESPSever_Task(void)
{
    char cmd_buf[128];
    float cmd_val;
    char *colon;
    char *start;
    char reply[32];      // 提前声明，避免C89错误
    
    if(ESP8266_Rx_OK)
    {
        // 关键过滤：只处理 +IPD 消息，忽略 ESP8266 的 AT 响应（如 OK, >, SEND OK 等）
        if (strstr(ESP_Receive_Buff, "+IPD") == NULL) {
            // 不是TCP接收数据，直接清空并返回，不回复任何消息
            memset(ESP_Receive_Buff, 0, sizeof(ESP_Receive_Buff));
            ESP8266_Rx_OK = 0;
            return;
        }
        
        // 查找最后一个冒号（ESP8266 数据格式通常为 +IPD,<id>,<len>:<data>）
        colon = strrchr(ESP_Receive_Buff, ':');
        if(colon == NULL) {
            // 格式错误，忽略
            memset(ESP_Receive_Buff, 0, sizeof(ESP_Receive_Buff));
            ESP8266_Rx_OK = 0;
            return;
        }
        
        // 跳过冒号，复制后续内容
        start = colon + 1;
        while(*start == ' ') start++;
        strncpy(cmd_buf, start, sizeof(cmd_buf)-1);
        cmd_buf[sizeof(cmd_buf)-1] = '\0';
        
        // 去除首尾空白（包括 \r\n）
        trim(cmd_buf);
        
        // 如果提取出的命令为空，忽略
        if (strlen(cmd_buf) == 0) {
            memset(ESP_Receive_Buff, 0, sizeof(ESP_Receive_Buff));
            ESP8266_Rx_OK = 0;
            return;
        }
       
        // 1. 检查开关命令 A / B
        if (strlen(cmd_buf) == 1 && (cmd_buf[0] == 'A' || cmd_buf[0] == 'a')) {
            system_enabled = 1;
            ESP8266_SendToClient(0, "System enabled (ADC/DAC ON)\n");
            SetDACVoltage(g_motor_target);
            my_printf(USART3, "[WiFi CMD] System enabled\n");
        }
        else if (strlen(cmd_buf) == 1 && (cmd_buf[0] == 'B' || cmd_buf[0] == 'b')) {
            system_enabled = 0;
            ESP8266_SendToClient(0, "System disabled (ADC/DAC OFF)\n");
            SetDACVoltage(0.0f);
            g_motor_target = 0.0f;
            my_printf(USART3, "[WiFi CMD] System disabled\n");
        }
        // 2. 检查 AIRESULT_ 前缀
        else if (strncmp(cmd_buf, "AIRESULT_", 9) == 0) {
            if (sscanf(cmd_buf + 9, "%f", &cmd_val) == 1) {
                if (!system_enabled) {
                    ESP8266_SendToClient(0, "System disabled, send 'A' to enable.\n");
                } else {
                    g_motor_target = cmd_val;
                    if(g_motor_target > 3.3f) g_motor_target = 3.3f;
                    if(g_motor_target < 0.0f) g_motor_target = 0.0f;
                    g_last_cmd_time = sys_tick;
                    sprintf(reply, "Voltage set to %.2fV\n", g_motor_target);
                    ESP8266_SendToClient(0, reply);
                    SetDACVoltage(g_motor_target);
                    my_printf(USART3, "[WiFi CMD] Set voltage: %.2f V\n", g_motor_target);
                }
            } else {
                ESP8266_SendToClient(0, "Invalid AIRESULT format. Use AIRESULT_<value>\n");
            }
        }
        // 3. 兼容旧格式：尝试解析为浮点数（纯数字）
        else if (sscanf(cmd_buf, "%f", &cmd_val) == 1) {
            if (!system_enabled) {
                ESP8266_SendToClient(0, "System disabled, send 'A' to enable.\n");
            } else {
                g_motor_target = cmd_val;
                if(g_motor_target > 3.3f) g_motor_target = 3.3f;
                if(g_motor_target < 0.0f) g_motor_target = 0.0f;
                g_last_cmd_time = sys_tick;
                sprintf(reply, "Voltage set to %.2fV\n", g_motor_target);
                ESP8266_SendToClient(0, reply);
                SetDACVoltage(g_motor_target);
                my_printf(USART3, "[WiFi CMD] Set voltage: %.2f V\n", g_motor_target);
            }
        }
        // 4. 无法识别的命令：静默忽略，不回复（避免给电脑端发送大量无意义错误）
        else {
            // 可选调试打印到串口3，但不回复WiFi
            my_printf(USART3, "[WiFi] Unknown command ignored: %s\n", cmd_buf);
        }
        
        // 清空缓冲区
        memset(ESP_Receive_Buff, 0, sizeof(ESP_Receive_Buff));
        ESP8266_Rx_OK = 0;
    }
}

/* ESP8266 TCP服务器模式（多连接，端口8080），并设置静态IP为192.168.1.211 */
char ESP8266_TcpSever(void)
{
    char ip_cmd[80];
    
    ESP_RST_Init();
    RESET_IO(0);
    delay_ms(1000);
    RESET_IO(1);

    ESP8266_send_cmd("AT+RST\r\n", 1, "ready");
    my_printf(USART3, "复位完成，进行设置上电不自动连接路由器\r\n");

    if(ESP8266_send_cmd("AT+CWAUTOCONN=0\r\n", 10, "OK"))
    {
        my_printf(USART3, "WIFI上电不自动连接热点失败\r\n");
        return 10;
    }
    my_printf(USART3, "设置成功，进行设置模式\r\n");

    ESP8266_send_cmd("ATE0\r\n", 1, "OK");

    if(ESP8266_send_cmd("AT+CWMODE=1\r\n", 15, "OK"))
    {
        my_printf(USART3, "wifi模式设置失败\r\n");
        return 1;
    }
    my_printf(USART3, "模式设置成功，准备连接路由器\r\n");

    if(ESP8266_send_cmd("AT+CWJAP=\"" SSID "\",\"" PASS "\"\r\n", 25, "OK"))
    {
        my_printf(USART3, "路由器连接失败\r\n");
        return 2;
    }
    my_printf(USART3, "路由器连接成功,准备设置静态IP\r\n");

    sprintf(ip_cmd, "AT+CIPSTA=\"%s\",\"%s\",\"%s\"\r\n", STATIC_IP, GATEWAY, NETMASK);
    if(ESP8266_send_cmd(ip_cmd, 15, "OK"))
    {
        my_printf(USART3, "设置静态IP失败\r\n");
        return 4;
    }
    my_printf(USART3, "静态IP设置成功: %s\r\n", STATIC_IP);

    my_printf(USART3, "准备使能多连接\r\n");
    if(ESP8266_send_cmd("AT+CIPMUX=1\r\n", 15, "OK"))
    {
        my_printf(USART3, "使能多连接失败\r\n");
        return 3;
    }
    my_printf(USART3, "使能多连接成功,准备建立TCP服务器\r\n");

    if(ESP8266_send_cmd("AT+CIPSERVER=1,8080\r\n", 15, "OK"))
    {
        my_printf(USART3, "建立TCP服务器失败\r\n");
        return 3;
    }
    my_printf(USART3, "建立TCP服务器成功,准备查询ESP8266设备的IP地址\r\n");

    if(ESP8266_send_cmd("AT+CIFSR\r\n", 15, "OK"))
    {
        my_printf(USART3, "查询ESP8266设备的IP地址失败\r\n");
        return 3;
    }
    my_printf(USART3, "可以开始传输数据\r\n");
    return 0;
}

/* 以下函数保留但不使用，仅为满足接口 */
char ESP8266_TcpClient(void) { return 0; }
void Handle_ESPClient_Task(void) { }