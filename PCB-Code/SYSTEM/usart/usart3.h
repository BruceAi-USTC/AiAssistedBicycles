#ifndef __USART3_H
#define __USART3_H


extern unsigned int Usart3_RxCounter;
extern char Usart3_RxBuff[500];
extern unsigned char Usart3_rx_ok;

void Usart3_Init(unsigned int);

#endif