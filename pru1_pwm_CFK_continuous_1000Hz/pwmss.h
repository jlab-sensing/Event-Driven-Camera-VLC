#ifndef PWMSS_H
#define PWMSS_H

// Note: ePWM is on the L4 peripheral clock (L4_PER_CLK)

// The reusable PWM class. It encapsulates all the low-level file I/O.
// TBCTL (CLKDIV, HSPCLKDIV) register presets
// Choose TBCLK depending on the range of periods (frequencies) the chirp occurs on.
// Choosing too fast of a TBCLK means that 65535 (16 bit max) cycles at that divided clock
// requency will be insufficient to obtain the desired low frequency.
// Likewise, choosing too slow of a TBCLK means that a single TBCLK will exceed the desired high frequency.
/*
#define TBCLK_10_NS    0b000000 // fastest TBCLK
#define TBCLK_20_NS    0b000001
#define TBCLK_40_NS    0b000011
#define TBCLK_80_NS    0b000100
#define TBCLK_100_NS   0b000101
#define TBCLK_200_NS   0b001101
#define TBCLK_400_NS   0b010101
#define TBCLK_800_NS   0b011101
#define TBCLK_1600_NS  0b100101
#define TBCLK_3200_NS  0b101101
#define TBCLK_6400_NS  0b110101
#define TBCLK_12800_NS 0b111101
#define TBCLK_17920_NS 0b111111 // slowest TBCLK
*/

#define TBCLK_BASE_FREQ_HZ 100000000
#define TBCLK_BASE_PERIOD_NS 10

#define CLKDIV_DIV1   0b000
#define CLKDIV_DIV2   0b001
#define CLKDIV_DIV4   0b010
#define CLKDIV_DIV8   0b011
#define CLKDIV_DIV16  0b100
#define CLKDIV_DIV32  0b101
#define CLKDIV_DIV64  0b110
#define CLKDIV_DIV128 0b111

#define HSPCLKDIV_DIV1  0b000
#define HSPCLKDIV_DIV2  0b001
#define HSPCLKDIV_DIV4  0b010
#define HSPCLKDIV_DIV6  0b011
#define HSPCLKDIV_DIV8  0b100
#define HSPCLKDIV_DIV10 0b101
#define HSPCLKDIV_DIV12 0b110
#define HSPCLKDIV_DIV14 0b111

// Common presets

#define TBCLK    TBCLK_100_NS  // TODO: this should be changed if chirps do not come out properly
#define TBCLK_NS 100           // TODO: change this to match the TBCLK timing above this line

#define PWMSS0_BASE 0x48300000
#define PWMSS1_BASE 0x48302000
#define PWMSS2_BASE 0x48304000
#define PWMSSx_LEN  0x00002000
// #define PWMSSx_LEN  0x00000260

#define PWMSSx_ECAP_OFFSET 0x100
#define PWMSSx_ECAP_LEN    0x080
#define PWMSSx_EQEP_OFFSET 0x180
#define PWMSSx_EQEP_LEN    0x020
#define PWMSSx_EPWM_OFFSET 0x200
#define PWMSSx_EPWM_LEN    0x060

// Interrupts
// 39 ePWM2INT eHRPWM2 (PWM Subsystem) epwm_intr_intr_pend
// 86 ePWM0INT eHRPWM0 (PWM Subsystem) epwm_intr_intr_pend
// 87 ePWM1INT eHRPWM1 (PWM Subsystem) epwm_intr_intr_pend

// 17 GPIO 0 GPIOINT0A
// 18 GPIO 0 GPIOINT0B
// 19 GPIO 1 GPIOINT1A
// 20 GPIO 1 GPIOINT1B
// 21 GPIO 2 GPIOINT2A
// 22 GPIO 2 GPIOINT2B
// 23 GPIO 3 GPIOINT3A
// 24 GPIO 3 GPIOINT3B

/* SYS_PWMSS register set */
typedef struct {
    /***************************/
    /* PWM Subsystem Registers */
    /***************************/
    /* SYS_PWMSS_IDVER register bit field */
    union {
        volatile uint32_t IDVER;

        volatile struct {
            unsigned Y_MINOR : 6;   // 5:0
            unsigned CUSTOM  : 2;   // 7:6
            unsigned X_MAJOR : 3;   // 10:8
            unsigned R_RTL   : 5;   // 15:11
            unsigned FUNC    : 12;  // 27:16
            unsigned rsvd28  : 2;   // 29:28
            unsigned SCHEME  : 2;   // 31:30
        } IDVER_bit;
    };  // 0x0

    /* SYS_PWMSS_SYSCONFIG register bit field */
    union {
        volatile uint32_t SYSCONFIG;

        volatile struct {
            unsigned SOFTRESET   : 1;   // 0
            unsigned FREEEMU     : 1;   // 1
            unsigned IDLEMODE    : 2;   // 3:2
            unsigned STANDBYMODE : 2;   // 5:4
            unsigned rsvd6       : 26;  // 31:6
        } SYSCONFIG_bit;
    };  // 0x4

    /* SYS_PWMSS_CLKCONFIG register bit field */
    union {
        volatile uint32_t CLKCONFIG;

        volatile struct {
            unsigned ECAPCLK_EN      : 1;   // 0
            unsigned ECAPCLKSTOP_REQ : 1;   // 1
            unsigned rsvd2           : 2;   // 3:2
            unsigned EQEPCLK_EN      : 1;   // 4
            unsigned EQEPCLKSTOP_REQ : 1;   // 5
            unsigned rsvd6           : 2;   // 7:6
            unsigned EPWMCLK_EN      : 1;   // 8
            unsigned EPWMCLKSTOP_REQ : 1;   // 9
            unsigned rsvd10          : 22;  // 31:10
        } CLKCONFIG_bit;
    };  // 0x8

    /* SYS_PWMSS_CLKSTATUS register bit field */
    union {
        volatile uint32_t CLKSTATUS;

        volatile struct {
            unsigned ECAPCLK_EN_ACK  : 1;   // 0
            unsigned ECAPCLKSTOP_ACK : 1;   // 1
            unsigned rsvd2           : 2;   // 3:2
            unsigned EQEPCLK_EN_ACK  : 1;   // 4
            unsigned EQEPCLKSTOP_ACK : 1;   // 5
            unsigned rsvd6           : 2;   // 7:6
            unsigned EPWMCLK_EN_ACK  : 1;   // 8
            unsigned EPWMCLKSTOP_ACK : 1;   // 9
            unsigned rsvd10          : 22;  // 31:10
        } CLKSTATUS_bit;
    };  // 0xC

    uint32_t rsvd10[60];  // 0x10 - 0xFC

    /*************************/
    /* eCAP Module Registers */
    /*************************/
    /* SYS_PWMSS_ECAP_TSCTR register bit field */
    union {
        volatile uint32_t ECAP_TSCTR;

        volatile struct {
            unsigned TSCTR : 32;  // 31:0
        } ECAP_TSCTR_bit;
    };  // 0x100

    /* SYS_PWMSS_ECAP_CTRPHS register bit field */
    union {
        volatile uint32_t ECAP_CTRPHS;

        volatile struct {
            unsigned CTRPHS : 32;  // 31:0
        } ECAP_CTRPHS_bit;
    };  // 0x104

    /* SYS_PWMSS_ECAP_CAP1 register bit field */
    union {
        volatile uint32_t ECAP_CAP1;

        volatile struct {
            unsigned CAP1 : 32;  // 31:0
        } ECAP_CAP1_bit;
    };  // 0x108

    /* SYS_PWMSS_ECAP_CAP2 register bit field */
    union {
        volatile uint32_t ECAP_CAP2;

        volatile struct {
            unsigned CAP2 : 32;  // 31:0
        } ECAP_CAP2_bit;
    };  // 0x10C

    /* SYS_PWMSS_ECAP_CAP3 register bit field */
    union {
        volatile uint32_t ECAP_CAP3;

        volatile struct {
            unsigned CAP3 : 32;  // 31:0
        } ECAP_CAP3_bit;
    };  // 0x110

    /* SYS_PWMSS_ECAP_CAP4 register bit field */
    union {
        volatile uint32_t ECAP_CAP4;

        volatile struct {
            unsigned CAP4 : 32;  // 31:0
        } ECAP_CAP4_bit;
    };  // 0x114

    uint32_t rsvd118[4];  // 0x118 - 0x124

    /* SYS_PWMSS_ECAP_ECCTL1 register bit field */
    volatile uint16_t ECAP_ECCTL1;  // 0x128

    /* SYS_PWMSS_ECAP_ECCTL2 register bit field */
    volatile uint16_t ECAP_ECCTL2;  // 0x12A

    /* SYS_PWMSS_ECAP_ECEINT register bit field */
    volatile uint16_t ECAP_ECEINT;  // 0x12C

    /* SYS_PWMSS_ECAP_ECFLG register bit field */
    volatile uint16_t ECAP_ECFLG;  // 0x12E

    /* SYS_PWMSS_ECAP_ECCLR register bit field */
    volatile uint16_t ECAP_ECCLR;  // 0x130

    /* SYS_PWMSS_ECAP_ECFRC register bit field */
    volatile uint16_t ECAP_ECFRC;  // 0x132

    uint32_t rsvd134[10];  // 0x134 - 0x158

    /* SYS_PWMSS_ECAP_REVID register bit field */
    union {
        volatile uint32_t ECAP_REVID;

        volatile struct {
            unsigned REV : 32;  // 31:0
        } ECAP_REVID_bit;
    };  // 0x15C

    uint32_t rsvd160[8];  // 0x160 - 0x17C

    /*************************/
    /* eQEP Module Registers */
    /*************************/
    /* SYS_PWMSS_EQEP_QPOSCNT register bit field */
    union {
        volatile uint32_t EQEP_QPOSCNT;

        volatile struct {
            unsigned QPOSCNT : 32;  // 31:0
        } EQEP_QPOSCNT_bit;
    };  // 0x180

    /* SYS_PWMSS_EQEP_QPOSINIT register bit field */
    union {
        volatile uint32_t EQEP_QPOSINIT;

        volatile struct {
            unsigned QPOSINIT : 32;  // 31:0
        } EQEP_QPOSINIT_bit;
    };  // 0x184

    /* SYS_PWMSS_EQEP_QPOSMAX register bit field */
    union {
        volatile uint32_t EQEP_QPOSMAX;

        volatile struct {
            unsigned QPOSMAX : 32;  // 31:0
        } EQEP_QPOSMAX_bit;
    };  // 0x188

    /* SYS_PWMSS_EQEP_QPOSCMP register bit field */
    union {
        volatile uint32_t EQEP_QPOSCMP;

        volatile struct {
            unsigned QPOSCMP : 32;  // 31:0
        } EQEP_QPOSCMP_bit;
    };  // 0x18C

    /* SYS_PWMSS_EQEP_QPOSILAT register bit field */
    union {
        volatile uint32_t EQEP_QPOSILAT;

        volatile struct {
            unsigned QPOSILAT : 32;  // 31:0
        } EQEP_QPOSILAT_bit;
    };  // 0x190

    /* SYS_PWMSS_EQEP_QPOSSLAT register bit field */
    union {
        volatile uint32_t EQEP_QPOSSLAT;

        volatile struct {
            unsigned QPOSSLAT : 32;  // 31:0
        } EQEP_QPOSSLAT_bit;
    };  // 0x194

    /* SYS_PWMSS_EQEP_QPOSLAT register bit field */
    union {
        volatile uint32_t EQEP_QPOSLAT;

        volatile struct {
            unsigned QPOSLAT : 32;  // 31:0
        } EQEP_QPOSLAT_bit;
    };  // 0x198

    /* SYS_PWMSS_EQEP_QUTMR register bit field */
    union {
        volatile uint32_t EQEP_QUTMR;

        volatile struct {
            unsigned QUTMR : 32;  // 31:0
        } EQEP_QUTMR_bit;
    };  // 0x19C

    /* SYS_PWMSS_EQEP_QUPRD register bit field */
    union {
        volatile uint32_t EQEP_QUPRD;

        volatile struct {
            unsigned QUPRD : 32;  // 31:0
        } EQEP_QUPRD_bit;
    };  // 0x1A0

    /* SYS_PWMSS_EQEP_QWDTMR register bit field */
    volatile uint16_t EQEP_QWDTMR;  // 0x1A4

    /* SYS_PWMSS_EQEP_QWDPRD register bit field */
    volatile uint16_t EQEP_QWDPRD;  // 0x1A6

    /* SYS_PWMSS_EQEP_QDECCTL register bit field */
    volatile uint16_t EQEP_QDECCTL;  // 0x1A8

    /* SYS_PWMSS_EQEP_QEPCTL register bit field */
    volatile uint16_t EQEP_QEPCTL;  // 0x1AA

    /* SYS_PWMSS_EQEP_QCAPCTL register bit field */
    volatile uint16_t EQEP_QCAPCTL;  // 0x1AC

    /* SYS_PWMSS_EQEP_QPOSCTL register bit field */
    volatile uint16_t EQEP_QPOSCTL;  // 0x1AE

    /* SYS_PWMSS_EQEP_QEINT register bit field */
    volatile uint16_t EQEP_QEINT;  // 0x1B0

    /* SYS_PWMSS_EQEP_QFLG register bit field */
    volatile uint16_t EQEP_QFLG;  // 0x1B2

    /* SYS_PWMSS_EQEP_QCLR register bit field */
    volatile uint16_t EQEP_QCLR;  // 0x1B4

    /* SYS_PWMSS_EQEP_QFRC register bit field */
    volatile uint16_t EQEP_QFRC;  // 0x1B6

    /* SYS_PWMSS_EQEP_QEPSTS register bit field */
    volatile uint16_t EQEP_QEPSTS;  // 0x1B8

    /* SYS_PWMSS_EQEP_QCTMR register bit field */
    volatile uint16_t EQEP_QCTMR;  // 0x1BA

    /* SYS_PWMSS_EQEP_QCPRD register bit field */
    volatile uint16_t EQEP_QCPRD;  // 0x1BC

    /* SYS_PWMSS_EQEP_QCTMRLAT register bit field */
    volatile uint16_t EQEP_QCTMRLAT;  // 0x1BE

    /* SYS_PWMSS_EQEP_QCPRDLAT register bit field */
    volatile uint16_t EQEP_QCPRDLAT;  // 0x1C0

    uint16_t rsvd1C2[1];  // 0x1C2 - 0x1C3
    uint32_t rsvd1C4[6];  // 0x1C4 - 0x1D8

    /* SYS_PWMSS_EQEP_REVID register bit field */
    union {
        volatile uint32_t EQEP_REVID;

        volatile struct {
            unsigned REVID : 32;  // 31:0
        } EQEP_REVID_bit;
    };  // 0x1DC

    uint32_t rsvd1E0[8];  // 0x1E0 - 0x1FC

    /*************************/
    /* ePWM Module Registers */
    /*************************/
    /* SYS_PWMSS_EPWM_TBCTL register bit field */
    // volatile uint16_t EPWM_TBCTL;	// 0x200
    union {
        volatile uint16_t EPWM_TBCTL;

        volatile struct {
            uint16_t CTRMODE   : 2;  // 1:0
            uint16_t PHSEN     : 1;  // 2
            uint16_t PRDLD     : 1;  // 3
            uint16_t SYNCOSEL  : 2;  // 5:4
            uint16_t SWFSYNC   : 1;  // 6
            uint16_t HSPCLKDIV : 3;  // 9:7
            uint16_t CLKDIV    : 3;  // 12:10
            uint16_t PHSDIR    : 1;  // 13
            uint16_t FREE_SOFT : 2;  // 15:14
        } EPWM_TBCTL_bit;
    };  // 0x200

    /* SYS_PWMSS_EPWM_TBSTS register bit field */
    // volatile uint16_t EPWM_TBSTS;  // 0x202
    union {
        volatile uint16_t EPWM_TBSTS;

        volatile struct {
            uint16_t CTRDIR : 1;   // 0
            uint16_t SYNCI  : 1;   // 1
            uint16_t CTRMAX : 1;   // 2
            uint16_t rsvd3  : 13;  // 15:3
        } EPWM_TBSTS_bit;
    };  // 0x202

    /* SYS_PWMSS_EPWM_TBPHSHR register bit field */
    // volatile uint16_t EPWM_TBPHSHR;  // 0x204
    union {
        volatile uint16_t EPWM_TBPHSHR;

        volatile struct {
            uint16_t rsvd0  : 8;  // 7:0
            uint16_t TBPHSH : 8;  // 15:8
        } EPWM_TBPHSHR_bit;
    };  // 0x204

    /* SYS_PWMSS_EPWM_TBPHS register bit field */
    volatile uint16_t EPWM_TBPHS;  // 0x206

    /* SYS_PWMSS_EPWM_TBCNT register bit field */
    volatile uint16_t EPWM_TBCNT;  // 0x208

    /* SYS_PWMSS_EPWM_TBPRD register bit field */
    volatile uint16_t EPWM_TBPRD;  // 0x20A

    uint16_t rsvd20C[1];  // 0x20C - 0x20D

    /* SYS_PWMSS_EPWM_CMPCTL register bit field */
    // volatile uint16_t EPWM_CMPCTL;	// 0x20E
    union {
        volatile uint16_t EPWM_CMPCTL;

        volatile struct {
            uint16_t LOADAMODE : 2;  // 1:0
            uint16_t LOADBMODE : 2;  // 3:2
            uint16_t SHDWAMODE : 1;  // 4
            uint16_t rsvd5     : 1;  // 5
            uint16_t SHDWBMODE : 1;  // 6
            uint16_t rsvd7     : 1;  // 7
            uint16_t SHDWAFULL : 1;  // 8
            uint16_t SHDWBFULL : 1;  // 9
            uint16_t rsvd10    : 6;  // 15:10
        } EPWM_CMPCTL_bit;
    };  // 0x20E

    /* SYS_PWMSS_EPWM_CMPAHR register bit field */
    // volatile uint16_t EPWM_CMPAHR;	// 0x210
    union {
        volatile uint16_t EPWM_CMPAHR;

        volatile struct {
            uint16_t rsvd0  : 8;  // 7:0
            uint16_t CMPAHR : 8;  // 15:8
        } EPWM_CMPAHR_bit;
    };  // 0x210

    /* SYS_PWMSS_EPWM_CMPA register bit field */
    volatile uint16_t EPWM_CMPA;  // 0x212

    /* SYS_PWMSS_EPWM_CMPB register bit field */
    volatile uint16_t EPWM_CMPB;  // 0x214

    /* SYS_PWMSS_EPWM_AQCTLA register bit field */
    volatile uint16_t EPWM_AQCTLA;  // 0x216

    /* SYS_PWMSS_EPWM_AQCTLB register bit field */
    volatile uint16_t EPWM_AQCTLB;  // 0x218

    /* SYS_PWMSS_EPWM_AQSFRC register bit field */
    volatile uint16_t EPWM_AQSFRC;  // 0x21A

    /* SYS_PWMSS_EPWM_AQCSFRC register bit field */
    volatile uint16_t EPWM_AQCSFRC;  // 0x21C

    /* SYS_PWMSS_EPWM_DBCTL register bit field */
    volatile uint16_t EPWM_DBCTL;  // 0x21E

    /* SYS_PWMSS_EPWM_DBRED register bit field */
    volatile uint16_t EPWM_DBRED;  // 0x220

    /* SYS_PWMSS_EPWM_DBFED register bit field */
    volatile uint16_t EPWM_DBFED;  // 0x222

    /* SYS_PWMSS_EPWM_TZSEL register bit field */
    volatile uint16_t EPWM_TZSEL;  // 0x224

    uint16_t rsvd226[1];  // 0x226 - 0x227

    /* SYS_PWMSS_EPWM_TZCTL register bit field */
    volatile uint16_t EPWM_TZCTL;  // 0x228

    /* SYS_PWMSS_EPWM_TZEINT register bit field */
    volatile uint16_t EPWM_TZEINT;  // 0x22A

    /* SYS_PWMSS_EPWM_TZFLG register bit field */
    volatile uint16_t EPWM_TZFLG;  // 0x22C

    /* SYS_PWMSS_EPWM_TZCLR register bit field */
    volatile uint16_t EPWM_TZCLR;  // 0x22E

    /* SYS_PWMSS_EPWM_TZFRC register bit field */
    volatile uint16_t EPWM_TZFRC;  // 0x230

    /* SYS_PWMSS_EPWM_ETSEL register bit field */
    // volatile uint16_t EPWM_ETSEL;  // 0x232
    union {
        volatile uint16_t EPWM_ETSEL;

        volatile struct {
            uint16_t INTSEL : 3;   // 2:0
            uint16_t INTEN  : 1;   // 3
            uint16_t rsvd4  : 12;  // 15:4
        } EPWM_ETSEL_bit;
    };  // 0x232

    /* SYS_PWMSS_EPWM_ETPS register bit field */
    // volatile uint16_t EPWM_ETPS;  // 0x234
    union {
        volatile uint16_t EPWM_ETPS;

        volatile struct {
            uint16_t INTPRD : 2;   // 1:0
            uint16_t INTCNT : 2;   // 3:2
            uint16_t rsvd4  : 12;  // 15:4
        } EPWM_ETPS_bit;
    };  // 0x234

    /* SYS_PWMSS_EPWM_ETFLG register bit field */
    // volatile uint16_t EPWM_ETFLG;  // 0x236
    union {
        volatile uint16_t EPWM_ETFLG;

        volatile struct {
            uint16_t INT   : 1;   // 0
            uint16_t rsvd1 : 15;  // 15:1
        } EPWM_ETFLG_bit;
    };  // 0x236

    /* SYS_PWMSS_EPWM_ETCLR register bit field */
    // volatile uint16_t EPWM_ETCLR;  // 0x238
    union {
        volatile uint16_t EPWM_ETCLR;

        volatile struct {
            uint16_t INT   : 1;   // 0
            uint16_t rsvd1 : 15;  // 15:1
        } EPWM_ETCLR_bit;
    };  // 0x238

    /* SYS_PWMSS_EPWM_ETFRC register bit field */
    // volatile uint16_t EPWM_ETFRC;  // 0x23A
    union {
        volatile uint16_t EPWM_ETFRC;

        volatile struct {
            uint16_t INT   : 1;   // 0
            uint16_t rsvd1 : 15;  // 15:1
        } EPWM_ETFRC_bit;
    };  // 0x23A

    /* SYS_PWMSS_EPWM_PCCTL register bit field */
    volatile uint16_t EPWM_PCCTL;  // 0x23C

    uint16_t rsvd23E[1];   // 0x23E - 0x23F
    uint32_t rsvd240[32];  // 0x240 - 0x2BC

    /* SYS_PWMSS_EPWM_HRCNGF register bit field */
    volatile uint16_t EPWM_HRCNGF;  // 0x2C0

} sysPwmss;

// volatile __far sysPwmss PWMSS0 __attribute__((cregister("PWMSS0", far), peripheral));
// volatile __far sysPwmss PWMSS1 __attribute__((cregister("PWMSS1", far), peripheral));
// volatile __far sysPwmss PWMSS2 __attribute__((cregister("PWMSS2", far), peripheral));

#endif
