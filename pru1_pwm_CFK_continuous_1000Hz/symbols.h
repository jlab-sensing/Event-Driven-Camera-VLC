#ifndef SYMBOLS_H
#define SYMBOLS_H

// Note: at 10 us PWM period, the TBPRD == 1000

#define REDOFF 1
#define GREENOFF 1
#define BLUEOFF 1

#define REDON 750
#define GREENON 750
#define BLUEON 750

#define RED000   1000
#define GREEN000 615 // 8679
#define BLUE000  152 // 1322

#define RED001   442
#define GREEN001 263
#define BLUE001  65 // 1000

#define RED010   3
#define GREEN010 182
#define BLUE010  381

#define RED011   7 // 9212
#define GREEN011 425
#define BLUE011  888 // 787

#define RED100   1
#define GREEN100 1000 // 8679
#define BLUE100  445 // 1322

#define RED101   1
#define GREEN101 429
#define BLUE101  191 // 1000

#define RED110   543
#define GREEN110 17
#define BLUE110  255

#define RED111   1000 // 9212
#define GREEN111 40
#define BLUE111  595 // 787


#endif
