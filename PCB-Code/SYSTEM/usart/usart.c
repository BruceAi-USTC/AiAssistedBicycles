#include "stm32f10x.h"
#include "usart.h"

void Usart1_Init(unsigned int bound)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    USART_InitTypeDef USART_InitStructure;
#if USART1_RX_ENABLE
    NVIC_InitTypeDef NVIC_InitStructure;
#endif

#if USART1_RX_ENABLE
    NVIC_PriorityGroupConfig(NVIC_PriorityGroup_2);
#endif

    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    USART_InitStructure.USART_BaudRate = bound;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
#if USART1_RX_ENABLE
    USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
#else
    USART_InitStructure.USART_Mode = USART_Mode_Tx;
#endif
    USART_Init(USART1, &USART_InitStructure);

#if USART1_RX_ENABLE
    USART_ClearFlag(USART1, USART_FLAG_RXNE);
    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
    USART_ITConfig(USART1, USART_IT_IDLE, ENABLE);
    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_Init(&NVIC_InitStructure);
#endif

    USART_Cmd(USART1, ENABLE);
}

char Usart_buff[1024];

int fputc(int ch, FILE *p)
{
    USART_SendData(USART1, (u8)ch);
    while(USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
    return ch;
}

void send_byte(USART_TypeDef *USARTx, u8 byte)
{
    while(!(USART_GetFlagStatus(USARTx, USART_FLAG_TC)));
    USART_SendData(USARTx, byte);
}

void send_str(USART_TypeDef *USARTx, u8 *str)
{
    int len = strlen((const char *)str);
    while(len--)
        send_byte(USARTx, *str++);
}

void send_str_len(USART_TypeDef *USARTx, u8 *str, u8 len)
{
    while(len--)
        send_byte(USARTx, *str++);
}

void my_printf(USART_TypeDef *USARTx, char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsprintf(Usart_buff, fmt, ap);
    va_end(ap);
    send_str(USARTx, (u8 *)Usart_buff);
    memset(Usart_buff, 0, sizeof(Usart_buff));
}

u8 USART_Receive_byte(USART_TypeDef *USARTx, u8 byte)
{
    while(!USART_GetFlagStatus(USARTx, USART_FLAG_RXNE));
    return USART_ReceiveData(USARTx);
}

char Usart1_RxBuff[500];
unsigned char Usart1_Rx_ok = 0;
u8 Usart1_RxCounter = 0;

void USART1_IRQHandler(void)
{
    if(USART_GetITStatus(USART1, USART_IT_RXNE))
    {
        Usart1_RxBuff[Usart1_RxCounter++] = USART_ReceiveData(USART1);
        USART_ClearITPendingBit(USART1, USART_IT_RXNE);
    }
    if(USART_GetITStatus(USART1, USART_IT_IDLE))
    {
        USART1->SR;
        USART1->DR;
        Usart1_Rx_ok = 1;
        Usart1_RxCounter = 0;
    }
}