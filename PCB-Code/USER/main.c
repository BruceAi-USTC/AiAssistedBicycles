/**
 * @file    main.c
 * @brief   电动助力控制器 + WiFi远程监控与控制 (10Hz 传感器发送)
 *          - ADC1通道2（扭矩PA2），ADC1通道3（速度PA3）
 *          - DAC通道1（PA4）输出控制电机
 *          - ESP8266 TCP服务器（USART1），WiFi远程接收命令和发送传感器数据
 *          - ADXL345加速度传感器（I2C PB6/PB7），输出原始加速度值（g）
 *          - USART3（PC10/PC11）保留，支持本地串口命令和调试打印
 *          - LED（PD2）运行指示
 *          - 开关控制：发送 'A' 开启ADC/DAC功能，发送 'B' 关闭
 *          - 传感器发送格式：SENSOR_DATA_<扭矩电压>,<速度电压>,<X轴加速度>,<Y轴加速度>,<Z轴加速度>
 */

#include "stm32f10x.h"
#include "usart3.h"
#include "delay.h"
#include "led.h"
#include "esp8266.h"
#include "adxl345.h"          // 新增 ADXL345 驱动
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* 宏定义 */
#define DEBUG_ENABLE        1
#define MOTOR_TIMEOUT_MS    3000      // 无命令超时时间（毫秒）
#define RX_BUF_SIZE         1024
#define ADC_SEND_INTERVAL_MS 100       // 传感器发送间隔（毫秒） -> 10Hz

/* 全局变量 */
volatile float g_motor_target = 0.0f;      // 目标电机电压（0~3.3V）
volatile uint32_t g_last_cmd_time = 0;     // 最后收到命令的时间（ms）
volatile uint8_t system_enabled = 1;       // 1: 开启ADC/DAC收发, 0: 关闭

char rx_buf[RX_BUF_SIZE];                  // USART3接收缓冲区
uint16_t rx_index = 0;
uint8_t rx_complete = 0;

uint32_t sys_tick = 0;                     // 系统时间计数器（毫秒）

/* 函数声明 */
void SendString(char *str);
void DebugPrint(const char *fmt, ...);
void ParseCommand(void);
void ADC_Config(void);
uint16_t ReadADC(uint8_t channel);
void DAC_Config(void);
void SetDACVoltage(float voltage);
void SystemClock_Config(void);
void SendSensorDataViaWiFi(float torque, float speed, float ax, float ay, float az);  // 修改：增加加速度参数

/* 串口3发送字符串 */
void SendString(char *str) {
    Usart3_SendString(str);
}

/* 调试打印，输出到串口3 */
void DebugPrint(const char *fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);
    SendString(buf);
}

/* USART3 中断处理（接收本地命令） */
void USART3_IRQHandler(void) {
    if (USART_GetITStatus(USART3, USART_IT_RXNE) != RESET) {
        char c = USART_ReceiveData(USART3);
        if (rx_complete) return;
        if (c == '\n' || rx_index >= RX_BUF_SIZE - 1) {
            rx_buf[rx_index] = '\0';
            rx_complete = 1;
            rx_index = 0;
        } else if (c != '\r') {
            rx_buf[rx_index++] = c;
        }
    }
}

/* 解析串口3接收到的命令：A/B开关、数字电压值 */
void ParseCommand(void) {
    float val;
    if (rx_complete) {
        DebugPrint("\r\n[UART3] Received: %s\r\n", rx_buf);
        // 开关命令 A（开启）
        if (strlen(rx_buf) == 1 && (rx_buf[0] == 'A' || rx_buf[0] == 'a')) {
            system_enabled = 1;
            DebugPrint("[UART3] System enabled (ADC/DAC ON)\r\n");
            SetDACVoltage(g_motor_target);   // 恢复当前目标电压
        }
        // 开关命令 B（关闭）
        else if (strlen(rx_buf) == 1 && (rx_buf[0] == 'B' || rx_buf[0] == 'b')) {
            system_enabled = 0;
            DebugPrint("[UART3] System disabled (ADC/DAC OFF)\r\n");
            SetDACVoltage(0.0f);
            g_motor_target = 0.0f;
        }
        // 数字电压命令（仅在开启状态下生效）
        else if (system_enabled && sscanf(rx_buf, "%f", &val) == 1) {
            g_motor_target = val;
            if (g_motor_target > 3.3f) g_motor_target = 3.3f;
            if (g_motor_target < 0.0f) g_motor_target = 0.0f;
            g_last_cmd_time = sys_tick;
            DebugPrint("[UART3] Set motor voltage: %.2f V\r\n", g_motor_target);
            SetDACVoltage(g_motor_target);
        }
        else if (!system_enabled) {
            DebugPrint("[UART3] System disabled, ignore voltage command.\r\n");
        }
        else {
            DebugPrint("[UART3] Invalid command. Send 'A' to enable, 'B' to disable, or a number (0-5).\r\n");
        }
        rx_complete = 0;
        rx_index = 0;
    }
}

/* ADC1 初始化（PA3速度，PA2扭矩） */
void ADC_Config(void) {
    GPIO_InitTypeDef GPIO_InitStructure;
    ADC_InitTypeDef ADC_InitStructure;
    
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_ADC1, ENABLE);
    RCC_ADCCLKConfig(RCC_PCLK2_Div6);
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2 | GPIO_Pin_3;  // PA2扭矩, PA3速度
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    
    ADC_InitStructure.ADC_Mode = ADC_Mode_Independent;
    ADC_InitStructure.ADC_ScanConvMode = DISABLE;
    ADC_InitStructure.ADC_ContinuousConvMode = DISABLE;
    ADC_InitStructure.ADC_ExternalTrigConv = ADC_ExternalTrigConv_None;
    ADC_InitStructure.ADC_DataAlign = ADC_DataAlign_Right;
    ADC_InitStructure.ADC_NbrOfChannel = 1;
    ADC_Init(ADC1, &ADC_InitStructure);
    
    ADC_Cmd(ADC1, ENABLE);
    ADC_ResetCalibration(ADC1);
    while (ADC_GetResetCalibrationStatus(ADC1));
    ADC_StartCalibration(ADC1);
    while (ADC_GetCalibrationStatus(ADC1));
}

/* 读取指定ADC通道（软件触发，30次平均） */
uint16_t ReadADC(uint8_t channel) {
    uint32_t sum = 0;
    uint16_t result;
    int i;
    for (i = 0; i < 30; i++) {
        ADC_RegularChannelConfig(ADC1, channel, 1, ADC_SampleTime_239Cycles5);
        ADC_SoftwareStartConvCmd(ADC1, ENABLE);
        while (ADC_GetFlagStatus(ADC1, ADC_FLAG_EOC) == RESET);
        sum += ADC_GetConversionValue(ADC1);
    }
    result = (uint16_t)(sum / 30);
    return result;
}

/* DAC 初始化（PA4） */
void DAC_Config(void) {
    GPIO_InitTypeDef GPIO_InitStructure;
    DAC_InitTypeDef DAC_InitStructure;
    
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_DAC, ENABLE);
    
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AIN;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    
    DAC_InitStructure.DAC_Trigger = DAC_Trigger_None;
    DAC_InitStructure.DAC_WaveGeneration = DAC_WaveGeneration_None;
    DAC_InitStructure.DAC_LFSRUnmask_TriangleAmplitude = DAC_TriangleAmplitude_1;
    DAC_InitStructure.DAC_OutputBuffer = DAC_OutputBuffer_Enable;
    DAC_Init(DAC_Channel_1, &DAC_InitStructure);
    DAC_Cmd(DAC_Channel_1, ENABLE);
}

/* 设置DAC输出电压（0~3.3V对应0~4095） */
void SetDACVoltage(float voltage) {
    uint16_t val;
    
    if (voltage > 3.3f) voltage = 3.3f;
    if (voltage < 0) voltage = 0;
    val = (uint16_t)(voltage * 4095.0f / 3.3f);
    DAC_SetChannel1Data(DAC_Align_12b_R, val);
}

/* 系统时钟配置（72MHz） */
void SystemClock_Config(void) {
    ErrorStatus HSEStartUpStatus;
    RCC_DeInit();
    RCC_HSEConfig(RCC_HSE_ON);
    HSEStartUpStatus = RCC_WaitForHSEStartUp();
    if (HSEStartUpStatus == SUCCESS) {
        FLASH_PrefetchBufferCmd(FLASH_PrefetchBuffer_Enable);
        FLASH_SetLatency(FLASH_Latency_2);
        RCC_HCLKConfig(RCC_SYSCLK_Div1);
        RCC_PCLK2Config(RCC_HCLK_Div1);
        RCC_PCLK1Config(RCC_HCLK_Div2);
        RCC_PLLConfig(RCC_PLLSource_HSE_Div1, RCC_PLLMul_9);
        RCC_PLLCmd(ENABLE);
        while (RCC_GetFlagStatus(RCC_FLAG_PLLRDY) == RESET);
        RCC_SYSCLKConfig(RCC_SYSCLKSource_PLLCLK);
        while (RCC_GetSYSCLKSource() != 0x08);
    } else {
        while (1);
    }
    SystemCoreClockUpdate();
}

/* 通过WiFi发送传感器数据到客户端（ID=0），数据格式：扭矩电压,速度电压,X轴加速度(g),Y轴加速度(g),Z轴加速度(g) */
void SendSensorDataViaWiFi(float torque, float speed, float ax, float ay, float az) {
    char buf[96];
    sprintf(buf, "SENSOR_DATA_%.2f,%.2f,%.2f,%.2f,%.2f\n", torque, speed, ax, ay, az);
    ESP8266_SendToClient(0, buf);
}

/* 主函数 */
int main(void) {
    uint32_t last_adc_time = 0;
    uint32_t last_led_time = 0;
    uint16_t adc_torque, adc_speed;
    float torque_voltage, speed_voltage;
    short ax_raw, ay_raw, az_raw;          // ADXL345 原始值
    float ax_g, ay_g, az_g;                // 转换后的加速度(g)
    char sensor_buf[128];
    uint32_t now;
    
    SystemClock_Config();
    delay_init();
    Usart3_Init(115200);
    ADC_Config();
    DAC_Config();
    LED_Init();
    
    // 初始化 ADXL345 加速度传感器
    ADXL345_Init();
    DebugPrint("[ADXL345] Initialized.\r\n");
    
    Usart1_Init(115200);
    ESP_RST_Init();
    
    SendString("\r\n===== Electric Assist Controller + WiFi Server Started (10Hz Sensor) =====\r\n");
    DebugPrint("SystemCoreClock: %d Hz\r\n", SystemCoreClock);
    
    if (ESP8266_TcpSever() != 0) {
        DebugPrint("ESP8266 TCP Server init failed!\r\n");
        while(1);
    }
    DebugPrint("ESP8266 Server ready. Connect to port 8080.\r\n");
    DebugPrint("Send 'A' to enable ADC/DAC, 'B' to disable. Send number (0-5) to set voltage.\r\n");
    DebugPrint("Sensor data sending rate: 10Hz (every 100ms). Format: SENSOR_DATA_<torqueV>,<speedV>,<ax_g>,<ay_g>,<az_g>\r\n");
    
    while (1) {
        delay_ms(10);
        sys_tick += 10;
        now = sys_tick;
        
        /* 1. 处理本地串口命令 */
        ParseCommand();
        
        /* 2. 处理WiFi命令 */
        Handle_ESPSever_Task();
        
        /* 3. 定期读取ADC和加速度传感器并发送（独立于system_enabled，始终执行） */
        if (now - last_adc_time >= ADC_SEND_INTERVAL_MS) {
            last_adc_time = now;
            
            // 读取ADC（扭矩PA2，速度PA3）
            adc_torque = ReadADC(ADC_Channel_2);   // PA2 -> 扭矩
            adc_speed  = ReadADC(ADC_Channel_3);   // PA3 -> 速度
            torque_voltage = adc_torque * 3.3f / 4096.0f;
            speed_voltage  = adc_speed * 3.3f / 4096.0f;
            
            // 读取加速度原始值并转换为g（量程±16g，灵敏度约256 LSB/g）
            adxl345_read_data(&ax_raw, &ay_raw, &az_raw);
            ax_g = (float)ax_raw / 256.0f;
            ay_g = (float)ay_raw / 256.0f;
            az_g = (float)az_raw / 256.0f;
            
            // 调试打印到串口3
            if (DEBUG_ENABLE) {
                DebugPrint("Torque=%.2fV Speed=%.2fV Accel: X=%.2fg Y=%.2fg Z=%.2fg\r\n",
                           torque_voltage, speed_voltage, ax_g, ay_g, az_g);
            }
            
            // 构造并发送传感器数据（串口3 + WiFi）
            sprintf(sensor_buf, "SENSOR_DATA_%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    torque_voltage, speed_voltage, ax_g, ay_g, az_g);
            SendString(sensor_buf);
            SendSensorDataViaWiFi(torque_voltage, speed_voltage, ax_g, ay_g, az_g);
        }
        
        /* 4. 电机控制逻辑（受system_enabled控制） */
        if (system_enabled) {
            /* 超时控制：长时间无命令则清零输出电压 */
            if (g_last_cmd_time != 0 && (now - g_last_cmd_time) >= MOTOR_TIMEOUT_MS) {
                g_motor_target = 0.0f;
                g_last_cmd_time = 0;
                DebugPrint("[TIMEOUT] Motor voltage set to 0V\r\n");
                SetDACVoltage(g_motor_target);
            } else {
                SetDACVoltage(g_motor_target);
            }
        } else {
            /* 系统关闭时确保DAC输出0V */
            static uint8_t dac_off_done = 0;
            if (!dac_off_done) {
                SetDACVoltage(0.0f);
                dac_off_done = 1;
            }
        }
        
        /* 5. LED闪烁（每500ms，指示系统运行） */
        if (now - last_led_time >= 500) {
            last_led_time = now;
            LED = !LED;
        }
    }
}