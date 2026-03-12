#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

//#define DATA_FILENAME "CFK_symbols_interleaved_translated_1000Hzhope.txt"
// Deterministic OOK-like pattern for Section 3.1 illumination sweeps.
#define DATA_FILENAME "s31_lux_sweep_1000Hz_symbols.txt"
#define DEV_MEM_FILENAME "/dev/mem"

#define PING_ADDR		0x9FFC0000
#define PONG_ADDR		0x9FFE0000
#define INDICATOR_ADDR  0x9FFB0000
#define PING			0
#define PONG			1

int main(int argc, char** argv) {
    FILE* f_data = NULL;
    // FILE* f_output_copy = NULL;
    int f_mem = 0;
    char *ping_ptr, *pong_ptr, *indicator;
    char c;
	// uint8_t curr_buff = PING;
    
    f_data = fopen(DATA_FILENAME, "r");
    if (!f_data) {
        printf("Failed to open \"%s\" in mode \"r\", exiting.\r\n", DATA_FILENAME);
        return EXIT_FAILURE;
    }
    
    // f_output_copy = fopen(OUTPUT_COPY_FILENAME, "w+");
    // if (!f_output_copy) {
        // printf("Failed to open \"%s\" in mode \"w+\", exiting.\r\n", OUTPUT_COPY_FILENAME);
        // return EXIT_FAILURE;
    // }

    f_mem = open(DEV_MEM_FILENAME,O_RDWR|O_SYNC);
    if(f_mem < 0)
    {
        printf("Can't open \"%s\"\r\n", DEV_MEM_FILENAME);
        return EXIT_FAILURE;
    }
	/* mmap 128kB ping and pong buffers */
    ping_ptr  = (char *) mmap(0, getpagesize()*32, PROT_READ|PROT_WRITE, MAP_SHARED, f_mem, PING_ADDR);
    pong_ptr  = (char *) mmap(0, getpagesize()*32, PROT_READ|PROT_WRITE, MAP_SHARED, f_mem, PONG_ADDR);
    indicator = (char *) mmap(0, getpagesize(),    PROT_READ|PROT_WRITE, MAP_SHARED, f_mem, INDICATOR_ADDR);
    if(ping_ptr == NULL || pong_ptr == NULL || indicator == NULL)
    {
        printf("Failed to mmap().\r\n");
        return EXIT_FAILURE;
    }
    
    printf("Pre-filling both buffers.\r\n");
    
    printf("ping buffer\r\n");
    for (int i = 0; i < (getpagesize()*32); i++){
        c = fgetc(f_data);
        if (c == EOF || c == 0xFF){
            printf("Reached EOF on \"%s\".\r\n", DATA_FILENAME);
            // fclose(f_output_copy);
            fclose(f_data);
            
            printf("holding open (CTRL+C to close and release held memory)\r\n");
            while(1); // Keep process alive so PRU can keep reading the preloaded buffer.
            
            return EXIT_FAILURE;
        }
        // fputc(c, f_output_copy);
        ping_ptr[i] = c;
    }
    
    printf("pong buffer\r\n");
    for (int i = 0; i < (getpagesize()*32); i++){
        c = fgetc(f_data);
        if (c == EOF || c == 0xFF){
            printf("Reached EOF on \"%s\".\r\n", DATA_FILENAME);
            // fclose(f_output_copy);
            fclose(f_data);
            return EXIT_FAILURE;
        }
        // fputc(c, f_output_copy);
        pong_ptr[i] = c;
    }
    
    // Indicator points to which buffer is currently in use by the PRU.
    // The PRU starts with the ping buffer.
    indicator[0] = PING;
    
    printf("pre-fill done\r\n");
        
    while(1) {
        
        while(indicator[0] == PING); // Wait for the ping buffer to be released.
        
        for (int i = 0; i < (getpagesize()*32); i++){
            c = fgetc(f_data);
            if (c == EOF || c == 0xFF){
                printf("Reached EOF on \"%s\".\r\n", DATA_FILENAME);
                // fclose(f_output_copy);
                fclose(f_data);
                return EXIT_FAILURE;
            }
            // fputc(c, f_output_copy);
            ping_ptr[i] = c;
        }
        
        while(indicator[0] == PONG); // Wait for the pong buffer to be released.
        
        for (int i = 0; i < (getpagesize()*32); i++){
            c = fgetc(f_data);
            if (c == EOF || c == 0xFF){
                printf("Reached EOF on \"%s\".\r\n", DATA_FILENAME);
                // fclose(f_output_copy);
                fclose(f_data);
                return EXIT_FAILURE;
            }
            // fputc(c, f_output_copy);
            pong_ptr[i] = c;
        }
    }

    printf("Exiting program.\r\n");
    return EXIT_SUCCESS;
}

