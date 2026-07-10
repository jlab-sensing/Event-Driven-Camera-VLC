#ifndef GPIO_PRU0_PRU1_H
#define GPIO_PRU0_PRU1_H

// R30: pruout
// R31: pruin

// PRU0 R30 R31
#define P9_31  (1 << 0)
#define P9_29  (1 << 1)
#define P9_30  (1 << 2)
#define P9_28  (1 << 3)
#define P9_42B (1 << 4)
#define P9_27  (1 << 5)
#define P9_41B (1 << 6)
#define P9_25  (1 << 7)

// PRU0 R30
#define P8_12  (1 << 14)
#define P8_11  (1 << 15)

// PRU0 R31
#define P9_16  (1 << 14)
#define P8_15  (1 << 15)
#define P9_24  (1 << 16)
#define P9_41A  (1 << 16)

// PRU1 R30 R31
#define P8_45  (1 << 0)
#define P8_46  (1 << 1)
#define P8_43  (1 << 2)
#define P8_44  (1 << 3)
#define P8_41  (1 << 4)
#define P8_42  (1 << 5)
#define P8_39  (1 << 6)
#define P8_40  (1 << 7)
#define P8_27  (1 << 8)
#define P8_29  (1 << 9)
#define P8_28  (1 << 10)
#define P8_30  (1 << 11)
#define P8_21  (1 << 12)
#define P8_20  (1 << 13)

// PRU1 R31
#define P8_26  (1 << 16)

#endif