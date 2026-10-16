#ifndef __LED_H
#define __LED_H	 
#include "sys.h"
#include "usart.h"
#include "delay.h"

#define LED PAout(8)      // ¸ÄÎª PA8

void LED_Init(void);

#endif