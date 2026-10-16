#ifndef __USART_H
#define __USART_H
#include "sys.h" 
#include "stdio.h"
#include "stdarg.h"
#include "string.h"
#include "usart3.h"      // ¸ÄÓÃ usart3

#define USART1_RX_ENABLE 1

void Usart1_Init(unsigned int bound);
void my_printf(USART_TypeDef *USARTx, char *fmt, ...);
void send_str_len(USART_TypeDef *USARTx, u8 *str, u8 len);
void send_str(USART_TypeDef *USARTx, u8 *str);

extern char Usart1_RxBuff[500];
extern unsigned char Usart1_Rx_ok;

#endif