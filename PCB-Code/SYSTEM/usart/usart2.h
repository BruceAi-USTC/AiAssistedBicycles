#ifndef __USART2_H
#define __USART2_H

#include "led.h"

extern unsigned int Usart2_RxCounter;          //外部声明，其他文件可以调用该变量
extern char Usart2_RxBuff[500]; //外部声明，其他文件可以调用该变量
extern unsigned char Usart2_rx_ok;											//外部声明，其他文件可以调用该变量

void Usart2_Init(unsigned int);             
 

#endif


