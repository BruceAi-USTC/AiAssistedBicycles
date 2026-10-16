#include "stm32f10x.h"
#include "usart3.h"

void Usart3_Init(unsigned int bound)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    USART_InitTypeDef USART_InitStructure;

    // 1. 使能时钟
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOC | RCC_APB2Periph_AFIO, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART3, ENABLE);

    // 2. 重映射：部分重映射到 PC10/PC11
    GPIO_PinRemapConfig(GPIO_PartialRemap_USART3, ENABLE);

    // 3. 配置 PC10 为 TX (复用推挽)
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOC, &GPIO_InitStructure);

    // 4. 配置 PC11 为 RX (浮空输入)
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOC, &GPIO_InitStructure);

    // 5. USART 初始化参数
    USART_InitStructure.USART_BaudRate = bound;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Tx;   // 调试只需要发送，可省略 Rx
    USART_Init(USART3, &USART_InitStructure);

    // 6. 使能 USART3
    USART_Cmd(USART3, ENABLE);
}

// 简单发送字符串函数（不依赖中断）
void Usart3_SendString(char *str)
{
    while(*str)
    {
        USART_SendData(USART3, (u8)*str++);
        while(USART_GetFlagStatus(USART3, USART_FLAG_TXE) == RESET);
    }
}