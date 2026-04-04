/*
 * Usage: make program && sleep 1 && echo "go" > /dev/rpmsg_pru31 && cat /dev/rpmsg_pru31
 */

/* ----- INCLUDES ----- */

/* ## STANDARD INCLUDES ## */

#include <stdint.h>
// #include <stdio.h>
#include <stdlib.h>
// #include <string.h>
// #include <unistd.h>
// #include <math.h>

/* ## PRU CORE INCLUDES ## */

#include <pru_rpmsg.h>
// #include <pru_types.h>
// #include <pru_virtio_ids.h>
// #include <pru_virtio_ring.h>
// #include <pru_virtqueue.h>
// #include <rsc_types.h>
// #include <types.h>

/* ## PRU PERIPHERAL INCLUDES ## */

#include <pru_cfg.h>
#include <pru_ctrl.h>
// #include <pru_ecap.h>
#include <pru_iep.h>
#include <pru_intc.h>
// #include <pru_uart.h>
// #include <sys_mailbox.h>
// #include <sys_mcspi.h>
// #include <sys_pwmss.h>
// #include <sys_tscAdcSs.h>

/* ## LOCAL INCLUDES ## */

// #include "gpio_pru0_pru1.h"
// #include "intc_map.h"
#include "pwmss.h"  // modification of <sys_pwmss.h> for more bitfields
#include "resource_table_1.h"
#include "symbols.h"

/* ----- DEFINES ----- */

/* ## DEBUG DEFINES ## */

// #define PRU_RPMSG_LOOPBACK_ENABLE

/* ## SHARED MEMORY DEFINES ## */

#define PING_ADDR		0x9FFC0000
#define PONG_ADDR		0x9FFE0000
#define INDICATOR_ADDR  0x9FFB0000
#define PING			0
#define PONG			1
#define SHARED_BUFFER_LEN 0x00020000

/* ## PIN DEFINES ## */

// for PRU1 (remoteproc2)
// #define P8_45 (1 << 0) /* R30 at 0x1 = pru1_pru0_pru_r30_0 = ball R1 = P8_45 */
#define P8_46 (1 << 1) /* R30 at 0x2 = pru1_pru0_pru_r30_1 = ball R2 = P8_46 */

#define OUTPUT_PIN P8_46

/* ## OTHER DEFINES ## */

#define CYCLES_PER_SECOND     200000000 /* PRU has 200 MHz clock */
#define NANOSECONDS_PER_CYCLE 5
// Calibrated down from 200 cycles to offset loop/shared-memory overhead.
#define SYMBOL_DELAY_CYCLES   130

/* ## RPMSG DEFINES ## */

/* Host-1 Interrupt sets bit 31 in register R31 */
#define HOST_INT ((uint32_t)1 << 31)

/* The PRU-ICSS system events used for RPMsg are defined in the Linux device tree
 * PRU0 uses system event 16 (To ARM) and 17 (From ARM)
 * PRU1 uses system event 18 (To ARM) and 19 (From ARM)
 */
#define TO_ARM_HOST   18
#define FROM_ARM_HOST 19

/*
 * Using the name 'rpmsg-client-sample' will probe the RPMsg sample driver
 * found at linux-x.y.z/samples/rpmsg/rpmsg_client_sample.c
 *
 * Using the name 'rpmsg-pru' will probe the rpmsg_pru driver found
 * at linux-x.y.z/drivers/rpmsg/rpmsg_pru.c
 */
// #define CHAN_NAME			"rpmsg-client-sample"
#define CHAN_NAME "rpmsg-pru"

#define CHAN_DESC "Channel 31"
#define CHAN_PORT 31

#define RPMSG_BUF_HEADER_SIZE 16

/*
 * Used to make sure the Linux drivers are ready for RPMsg communication
 * Found at linux-x.y.z/include/uapi/linux/virtio_config.h
 */
#define VIRTIO_CONFIG_S_DRIVER_OK 4

/* ----- MACROS ----- */

#define PRU_PRINT_LITERAL(x)   pru_rpmsg_send(&transport, 31, 1024, x, sizeof(x))
#define PRU_PRINT_STRING(x, y) pru_rpmsg_send(&transport, 31, 1024, x, y)
#define PRU_PRINT_INT(x)                                                        \
    do {                                                                        \
        transmit_length = itoa(x, (char *)transmit_buffer);                     \
        pru_rpmsg_send(&transport, 31, 1024, transmit_buffer, transmit_length); \
    } while (0)
#define PRU_PRINT_UNSIGNED_INT(x)                                               \
    do {                                                                        \
        transmit_length = itoa_unsigned(x, (char *)transmit_buffer);            \
        pru_rpmsg_send(&transport, 31, 1024, transmit_buffer, transmit_length); \
    } while (0)

/* ----- TYPEDEFS ----- */

/* ----- GLOBAL VARIABLES ----- */

// Modified (expanded) sysPwmss structs, originally found in <sys_pwmss.h>
// volatile __far sysPwmss PWMSS0 __attribute__((cregister("PWMSS0", far), peripheral)); // unused
volatile __far sysPwmss PWMSS1 __attribute__((cregister("PWMSS1", far), peripheral));
volatile __far sysPwmss PWMSS2 __attribute__((cregister("PWMSS2", far), peripheral));

volatile register uint32_t __R30; /* output register for PRU */
volatile register uint32_t __R31; /* output register for PRU */

uint8_t payload[RPMSG_BUF_SIZE - RPMSG_BUF_HEADER_SIZE];

struct pru_rpmsg_transport transport;
uint8_t transmit_buffer[RPMSG_BUF_SIZE - RPMSG_BUF_HEADER_SIZE];
uint32_t transmit_length = 0;

/* ----- FUNCTION PROTOTYPES ----- */

void pwm_register_dump();

// itoa and reverse from K&R
int itoa(int n, char s[]);
int itoa_unsigned(unsigned int n, char s[]);
void reverse(char s[], int len);

/* ----- FUNCTION DEFINITIONS ----- */

void main(void) {
    // struct pru_rpmsg_transport transport;
    uint16_t src, dst, len;
    volatile uint8_t *status;
    uint32_t i;
    char* ping_symbols;
    char* pong_symbols;
    volatile uint8_t* indicator;
    uint32_t chunk_len;
    uint32_t symbols_remaining;
    uint8_t curr_buffer;
    char* curr_symbols;
    uint32_t symbols_length = 0;
    volatile char c;
    
    ping_symbols = (char*)PING_ADDR;
    pong_symbols = (char*)PONG_ADDR;
    indicator = (volatile uint8_t*)INDICATOR_ADDR;

    /* Allow OCP master port access by the PRU so the PRU can read external memories */
    CT_CFG.SYSCFG_bit.STANDBY_INIT = 0;

    // PRU1_CTRL.CTRL_bit.CTR_EN = 1;  // Enable PRU system cycle counter (200 MHz)

    /* IEP timer
        4.4.3.2.2 Basic Programming Model
        Follow these basic steps to configure the IEP Timer.
        1. Initialize timer to known state (default values)
            a. Disable counter (GLB_CFG.CNT_ENABLE)
            b. Reset Count Register (CNT) by writing 0xFFFFFFFF to clear
            c. Clear overflow status register (GLB_STS.CNT_OVF)
            d. Clear compare status (CMP_STS)
        2. Set compare values (CMP0-CMPx)
        3. Enable compare event (CMP_CFG.CMP_EN)
        4. Set increment value (GLB_CFG.DEFAULT_INC)
        5. Set compensation value (COMPEN.COMPEN_CNT)
        6. Enable counter (GLB_CFG.CNT_ENABLE)
    */
    /*
    // CT_IEP.TMR_GLB_CFG_bit.CNT_EN = 0; // Disable IEP timer counter
    // CT_IEP.TMR_CNT = 0xFFFFFFFF;
    // CT_IEP.TMR_GLB_STS_bit.CNT_OVF = 0;
    // CT_IEP.TMR_CMP_STS_bit.CMP_HIT = 0;

    // CT_IEP.TMR_CMP_CFG_bit.CMP_EN = 0;

    CT_IEP.TMR_GLB_CFG_bit.DEFAULT_INC = 1;  // default 5

    // Compensation?
    // CT_IEP.TMR_GLB_CFG_bit.CMP_INC = 0;
    // CT_IEP.TMR_COMPEN_bit.COMPEN.CNT;

    CT_IEP.TMR_GLB_CFG_bit.CNT_EN = 1;  // Enable
    */

    /* Clear the status of the PRU-ICSS system event that the ARM will use to 'kick' us */
    CT_INTC.SICR_bit.STS_CLR_IDX = FROM_ARM_HOST;

    /* Make sure the Linux drivers are ready for RPMsg communication */
    status = &resourceTable.rpmsg_vdev.status;
    while (!(*status & VIRTIO_CONFIG_S_DRIVER_OK))
        ;

    /* Initialize the RPMsg transport structure */
    pru_rpmsg_init(&transport, &resourceTable.rpmsg_vring0, &resourceTable.rpmsg_vring1, TO_ARM_HOST, FROM_ARM_HOST);

    /* Create the RPMsg channel between the PRU and ARM user space using the transport structure. */
    while (pru_rpmsg_channel(RPMSG_NS_CREATE, &transport, CHAN_NAME, CHAN_DESC, CHAN_PORT) != PRU_RPMSG_SUCCESS)
        ;

    /* flash pin to indicate setup complete
    // __R30 &= ~OUTPUT_PIN;
    // uint8_t i;
    // for (i = 0; i < 6; i++){
        // __R30 ^= OUTPUT_PIN;
        // __delay_cycles(CYCLES_PER_SECOND / 4); // 0.5 sec
    // }
    */

    /* shared memory test
    __R30 &= ~OUTPUT_PIN;
    __delay_cycles(CYCLES_PER_SECOND * 5);
    for (i = 0; i < 0x20000; i++){
        __R30 |= OUTPUT_PIN; // HIGH
        c = *(symbols+i);    // Predict 27 cycles (200 MHz) = 135 ns to access OCMC RAM per TI doc sprace8a. 265 ns by experimentation. Periodically 345 ns. Once 510 ns. Accesses outside of PRU local memory are not deterministic.
        __R30 &= ~OUTPUT_PIN; // LOW
        
        // PRU_PRINT_STRING((symbols+i),1);
        // __delay_cycles(CYCLES_PER_SECOND / 10);
    }
    PRU_PRINT_LITERAL("Finished printing shared memory\r\n");
    while(1);
    */

    /* // rand timing test
    // __R30 &= ~OUTPUT_PIN;
    // while(1){
        // i = rand();
        // __R30 |= OUTPUT_PIN;
        // i = rand();
        // __R30 &= ~OUTPUT_PIN;

        // __delay_cycles(10000);
    // }
    */

    // The base clock of the PWM is 100 MHz (10 ns).
    // PWMCLK = (100 MHz) / (CLKDIV * HSPCLKDIV)
    // See "pwmss.h" for CLKDIV_DIVx and HSPCLKDIV_DIVx options.
    // PWMSS1.EPWM_TBCTL_bit.CLKDIV    = CLKDIV_DIV1;
    // PWMSS1.EPWM_TBCTL_bit.HSPCLKDIV = HSPCLKDIV_DIV2;
    // PWMSS1.EPWM_CMPA                = 0;
    // PWMSS1.EPWM_TBPRD               = 50000;
    // PWMSS1.EPWM_CMPA = PWMSS1.EPWM_TBPRD / 2;  // Red, P9.14, pwmchip4-0
    // PWMSS1.EPWM_CMPB                = 0;
    // PWMSS1.EPWM_TBPRD               = 50000;
    // PWMSS1.EPWM_CMPB = PWMSS1.EPWM_TBPRD / 2;  // Green, P9.16, pwmchip4-1

    // PWMSS2.EPWM_TBCTL_bit.CLKDIV    = CLKDIV_DIV1;
    // PWMSS2.EPWM_TBCTL_bit.HSPCLKDIV = HSPCLKDIV_DIV2;
    // PWMSS2.EPWM_CMPB                = 0;
    // PWMSS2.EPWM_TBPRD               = 50000;
    // PWMSS2.EPWM_CMPB = PWMSS1.EPWM_TBPRD / 2;  // Blue, P8.13, pwmchip7-1

    // Phase correction
    PWMSS1.EPWM_TBCTL_bit.SYNCOSEL = 0b01;  // TBCNT = 0 triggers sync output signal
    // Note: SYNCOSEL == 0b01 means that PWMSS2 will lag by 1 cycle,
    // which is 20 ns under DIV1 * DIV2 clock dividers.
    PWMSS2.EPWM_TBCTL_bit.PHSEN    = 0b1;

    // pwm_register_dump();
    
    while(1){
    
        PRU_PRINT_LITERAL("Waiting for message from Userspace to indicate shared memory length.\r\n");
        symbols_length = 0;
        while (1) {
            // Check bit 30 of register R31 to see if the ARM has kicked us
            if (__R31 & HOST_INT) {
                // Clear the event status
                CT_INTC.SICR_bit.STS_CLR_IDX = FROM_ARM_HOST;
                // Receive all available messages, multiple messages can be sent per kick
                while (pru_rpmsg_receive(&transport, &src, &dst, payload, &len) == PRU_RPMSG_SUCCESS) {
                    payload[len] = '\0'; // Null append to create string.
                    symbols_length = atoi((char*)payload);
                }
                if (symbols_length != 0) break;
            }
        }
        
        PRU_PRINT_LITERAL("Specified length is ");
        PRU_PRINT_UNSIGNED_INT(symbols_length);
        PRU_PRINT_LITERAL("\r\n");
        
        __delay_cycles(CYCLES_PER_SECOND);
        
        PRU_PRINT_LITERAL("CSK transmission start.\r\n");
        
        curr_buffer = PING;
        symbols_remaining = symbols_length;
        while (symbols_remaining > 0) {
            curr_symbols = (curr_buffer == PING) ? ping_symbols : pong_symbols;
            indicator[0] = curr_buffer;
            chunk_len = (symbols_remaining > SHARED_BUFFER_LEN) ? SHARED_BUFFER_LEN : symbols_remaining;

            for (i = 0; i < chunk_len; i++) {
                __R30 |= OUTPUT_PIN;
                c = curr_symbols[i];

                switch (c) {
                case '0': // treat legacy '0' as OFF for compatibility
                case 'o':
                    PWMSS1.EPWM_CMPA = REDOFF;
                    PWMSS1.EPWM_CMPB = GREENOFF;
                    PWMSS2.EPWM_CMPB = BLUEOFF;
                    break;
                case 'w':
                    PWMSS1.EPWM_CMPA = REDON;
                    PWMSS1.EPWM_CMPB = GREENON;
                    PWMSS2.EPWM_CMPB = BLUEON;
                    break;
                case '1': //00
                    PWMSS1.EPWM_CMPA = RED00;
                    PWMSS1.EPWM_CMPB = GREEN00;
                    PWMSS2.EPWM_CMPB = BLUE00;
                    break;
                case '2': //01
                    PWMSS1.EPWM_CMPA = RED01;
                    PWMSS1.EPWM_CMPB = GREEN01;
                    PWMSS2.EPWM_CMPB = BLUE01;
                    break;
                case '3': //10
                    PWMSS1.EPWM_CMPA = RED10;
                    PWMSS1.EPWM_CMPB = GREEN10;
                    PWMSS2.EPWM_CMPB = BLUE10;
                    break;
                case '4': //11
                    PWMSS1.EPWM_CMPA = RED11;
                    PWMSS1.EPWM_CMPB = GREEN11;
                    PWMSS2.EPWM_CMPB = BLUE11;
                    break;
                default:
                    break;
            }
                // PWMSS1.EPWM_TBCNT = PWMSS1.EPWM_TBPRD - 1;
                // PWMSS2.EPWM_TBCNT = PWMSS2.EPWM_TBPRD - 1;

                // PWMSS1.EPWM_TBCTL_bit.CTRMODE = 0b00;  // counter enable
                // PWMSS2.EPWM_TBCTL_bit.CTRMODE = 0b00;  // counter enable
                // PRU_PRINT_UNSIGNED_INT(buffer1[i]);
                __delay_cycles(SYMBOL_DELAY_CYCLES); // The rest of the loop adds the remaining symbol time.
                // PWMSS1.EPWM_TBCTL_bit.CTRMODE = 0b11;  // counter disable
                // PWMSS2.EPWM_TBCTL_bit.CTRMODE = 0b11;  // counter disable

                // // Rather than set the duty cycle to 0, consider modifying the AQ or CTL (freeze)
                // // PWMSS1.EPWM_AQ
                __R30 &= ~OUTPUT_PIN;
            }

            symbols_remaining -= chunk_len;
            curr_buffer = (curr_buffer == PING) ? PONG : PING;
            indicator[0] = curr_buffer; // Release the buffer just consumed so userspace can refill it.
        }

        PWMSS1.EPWM_CMPA = REDOFF;
        PWMSS1.EPWM_CMPB = GREENOFF;
        PWMSS2.EPWM_CMPB = BLUEOFF;
        PRU_PRINT_LITERAL("CSK transmission end.\r\n");
        
    }
    
    /* IEP timer example
    __delay_cycles(CYCLES_PER_SECOND);
    CT_IEP.TMR_GLB_STS_bit.CNT_OVF = 1;
    previousCycle                  = 0;
    while (1) {
        cycle = CT_IEP.TMR_CNT;

        // PRU_PRINT_UNSIGNED_INT(cycle);
        // PRU_PRINT_LITERAL("\t");
        if (CT_IEP.TMR_GLB_STS_bit.CNT_OVF) {
            CT_IEP.TMR_GLB_STS_bit.CNT_OVF = 1;  // write 1 to clear overflow
            PRU_PRINT_UNSIGNED_INT(cycle + (0xFFFFFFFF - previousCycle));
        } else {
            PRU_PRINT_UNSIGNED_INT((cycle - previousCycle));
        }

        PRU_PRINT_LITERAL("\r\n");

        previousCycle = cycle;
        __delay_cycles(CYCLES_PER_SECOND);
    }
    */

    /* PRU CPU cycle example
    while (1) {
        PRU1_CTRL.CYCLE = 0;
        PRU_PRINT_UNSIGNED_INT(PRU1_CTRL.CYCLE);
        PRU_PRINT_LITERAL("\r\n");
        __delay_cycles(10);
        PRU_PRINT_UNSIGNED_INT(PRU1_CTRL.CYCLE);
        PRU_PRINT_LITERAL("\r\n");
        __delay_cycles(100);
        PRU_PRINT_UNSIGNED_INT(PRU1_CTRL.CYCLE);
        PRU_PRINT_LITERAL("\r\n");

        __delay_cycles(CYCLES_PER_SECOND);
        PRU_PRINT_UNSIGNED_INT(PRU1_CTRL.CYCLE);
        PRU_PRINT_LITERAL("\r\n");
    }
    */

    /* Old LED output code
    while (1) {
        // Check bit 30 of register R31 to see if the ARM has kicked us
        if (__R31 & HOST_INT) {
            // Clear the event status
            CT_INTC.SICR_bit.STS_CLR_IDX = FROM_ARM_HOST;
            // Receive all available messages, multiple messages can be sent per kick
            while (pru_rpmsg_receive(&transport, &src, &dst, payload, &len) == PRU_RPMSG_SUCCESS) {
#ifdef PRU_RPMSG_LOOPBACK_ENABLE
                pru_rpmsg_send(&transport, dst, src, "Received ", sizeof("Received "));
                transmit_length                    = itoa(len, (char *)transmit_buffer);
                transmit_buffer[transmit_length++] = ':';
                pru_rpmsg_send(&transport, dst, src, transmit_buffer, transmit_length);
                for (i = 0, transmit_length = 0; i < len; i++) {
                    transmit_length += itoa(payload[i], (char *)(transmit_buffer + transmit_length));
                    transmit_buffer[transmit_length++] = ',';
                }
                transmit_buffer[transmit_length++] = ':';
                pru_rpmsg_send(&transport, dst, src, transmit_buffer, transmit_length);

                pru_rpmsg_send(&transport, dst, src, payload, len);

                transmit_length = itoa(PRU1_CTRL.CYCLE, (char *)transmit_buffer);
                pru_rpmsg_send(&transport, dst, src, transmit_buffer, transmit_length);
#endif

                // Process the symbols

                // startCycleCount  = PRU1_CTRL.CYCLE;
                // PWMSS1.EPWM_CMPA = PWMSS1.EPWM_TBPRD / 2;
                // PWMSS1.EPWM_TBCTL_bit.CTRMODE = 0b00;

                // Each cycle is 5 ns
                // Symbol duration is 0.0001 s = 0.1 ms == 100 us == 100,000 ns
                // Each PWM period is 10,000 ns == 10 us == 0.01 ms
                // To wait for one symbol (100,000 ns), 100000/5 = 20000 cycles

                // current_symbol = 0;

                // while (current_symbol < sizeof(symbols_to_send)){
                // symbol = symbols_to_send[current_symbol];
                // __R30 |= OUTPUT_PIN;
                // switch (symbol) {
                // case 0:
                // PWMSS1.EPWM_CMPA = REDOFF;
                // PWMSS1.EPWM_CMPB = GREENOFF;
                // PWMSS2.EPWM_CMPB = BLUEOFF;
                // break;
                // case 1:
                // PWMSS1.EPWM_CMPA = RED00;
                // PWMSS1.EPWM_CMPB = GREEN00;
                // PWMSS2.EPWM_CMPB = BLUE00;
                // break;
                // case 2:
                // PWMSS1.EPWM_CMPA = RED01;
                // PWMSS1.EPWM_CMPB = GREEN01;
                // PWMSS2.EPWM_CMPB = BLUE01;
                // break;
                // case 3:
                // PWMSS1.EPWM_CMPA = RED10;
                // PWMSS1.EPWM_CMPB = GREEN10;
                // PWMSS2.EPWM_CMPB = BLUE10;
                // break;
                // case 4:
                // PWMSS1.EPWM_CMPA = RED11;
                // PWMSS1.EPWM_CMPB = GREEN11;
                // PWMSS2.EPWM_CMPB = BLUE11;
                // break;
                // default:
                // break;
                // }
                // // PWMSS1.EPWM_TBCNT = PWMSS1.EPWM_TBPRD - 1;
                // // PWMSS2.EPWM_TBCNT = PWMSS2.EPWM_TBPRD - 1;

                // // PWMSS1.EPWM_TBCTL_bit.CTRMODE = 0b00;  // counter enable
                // // PWMSS2.EPWM_TBCTL_bit.CTRMODE = 0b00;  // counter enable
                // __delay_cycles(20000);                 // 20000 * 5 ns == 100000 ns == 100 us
                // // PWMSS1.EPWM_TBCTL_bit.CTRMODE = 0b11;  // counter disable
                // // PWMSS2.EPWM_TBCTL_bit.CTRMODE = 0b11;  // counter disable

                // // // Rather than set the duty cycle to 0, consider modifying the AQ or CTL (freeze)
                // // // PWMSS1.EPWM_AQ
                // __R30 &= ~OUTPUT_PIN;
                // current_symbol++;
                // }

                // PWMSS1.EPWM_CMPA = REDOFF;
                // PWMSS1.EPWM_CMPB = GREENOFF;
                // PWMSS2.EPWM_CMPB = BLUEOFF;
            }
        }
    }
    */
}

void pwm_register_dump() {
    PRU_PRINT_LITERAL("\r\n\r\nPRU register dump start");

    PRU_PRINT_LITERAL("\r\n\r\nPWMSS1");
    PRU_PRINT_LITERAL("\r\n\tTBCTL_bit.CLKDIV ");
    PRU_PRINT_INT(PWMSS1.EPWM_TBCTL_bit.CLKDIV);
    PRU_PRINT_LITERAL("\r\n\tTBCTL_bit.HSPCLKDIV ");
    PRU_PRINT_INT(PWMSS1.EPWM_TBCTL_bit.HSPCLKDIV);
    PRU_PRINT_LITERAL("\r\n\tTBPRD ");
    PRU_PRINT_INT(PWMSS1.EPWM_TBPRD);
    PRU_PRINT_LITERAL("\r\n\tCMPA (RED) ");
    PRU_PRINT_INT(PWMSS1.EPWM_CMPA);
    PRU_PRINT_LITERAL("\r\n\tCMPB (GREEN) ");
    PRU_PRINT_INT(PWMSS1.EPWM_CMPB);

    __delay_cycles(CYCLES_PER_SECOND);

    PRU_PRINT_LITERAL("\r\n\r\nPWMSS2");
    PRU_PRINT_LITERAL("\r\n\tTBCTL_bit.CLKDIV ");
    PRU_PRINT_INT(PWMSS2.EPWM_TBCTL_bit.CLKDIV);
    PRU_PRINT_LITERAL("\r\n\tTBCTL_bit.HSPCLKDIV ");
    PRU_PRINT_INT(PWMSS2.EPWM_TBCTL_bit.HSPCLKDIV);
    PRU_PRINT_LITERAL("\r\n\tTBPRD ");
    PRU_PRINT_INT(PWMSS2.EPWM_TBPRD);
    PRU_PRINT_LITERAL("\r\n\tCMPA (UNUSED) ");
    PRU_PRINT_INT(PWMSS2.EPWM_CMPA);
    PRU_PRINT_LITERAL("\r\n\tCMPB (BLUE) ");
    PRU_PRINT_INT(PWMSS2.EPWM_CMPB);

    PRU_PRINT_LITERAL("\r\n\r\nPRU register dump end\r\n\r\n");
}

/* itoa:  convert n to characters in s, returning number of digits written (not including null) */
int itoa(int n, char s[]) {
    int i, sign;

    if ((sign = n) < 0) /* record sign */
        n = -n;         /* make n positive */
    i = 0;
    do {                       /* generate digits in reverse order */
        s[i++] = n % 10 + '0'; /* get next digit */
    } while ((n /= 10) > 0);   /* delete it */
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    reverse(s, i);
    return i;
}

/* unsigned version of itoa() */
int itoa_unsigned(unsigned int n, char s[]) {
    int i;

    i = 0;
    do {                       /* generate digits in reverse order */
        s[i++] = n % 10 + '0'; /* get next digit */
    } while ((n /= 10) > 0);   /* delete it */
    s[i] = '\0';
    reverse(s, i);
    return i;
}

/* reverse:  reverse the first len characters of string s in place */
void reverse(char s[], int len) {
    int i, j;
    char c;

    if (len < 2) return;

    for (i = 0, j = len - 1; i < j; i++, j--) {
        c    = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}
