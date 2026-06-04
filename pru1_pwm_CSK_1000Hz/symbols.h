#ifndef SYMBOLS_H
#define SYMBOLS_H

// Note: at 10 us PWM period, the TBPRD == 1000


#define REDOFF 1
#define GREENOFF 1
#define BLUEOFF 1

#define REDON 1000
#define GREENON 1000
#define BLUEON 1000

#define RED00   97
#define GREEN00 81 // 8679
#define BLUE00  56 // 1322

#define RED01   1
#define GREEN01 131
#define BLUE01  1 // 1000

#define RED10   501
#define GREEN10 1
#define BLUE10  1

#define RED11   1 // 9212
#define GREEN11 47
#define BLUE11  1000 // 787


// Power matrix proposed
//#define RED00   0
//#define GREEN00 868 // 8679
//#define BLUE00  132 // 1322

//#define RED01   0
//#define GREEN01 0
//#define BLUE01  1000 // 1000

//#define RED10   382 // 3822
//#define GREEN10 293 // 2928
//#define BLUE10  325 // 3250

//#define RED11   921 // 9212
//#define GREEN11 0
//#define BLUE11  79 // 787


#endif
