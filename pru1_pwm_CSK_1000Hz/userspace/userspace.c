#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

//#define DATA_FILENAME "CFK_symbols_interleaved_translated_1000Hzhope.txt"
// Default symbol file. You can override from CLI: ./userspace <symbol_file>.
#define DEFAULT_DATA_FILENAME "s31_replication_1500Hz_30s_symbols.txt"
#define DEV_MEM_FILENAME "/dev/mem"

#define PING_ADDR		0x9FFC0000
#define PONG_ADDR		0x9FFE0000
#define INDICATOR_ADDR  0x9FFB0000
#define PING			0
#define PONG			1

int fill_buffer_from_file(FILE* f_data, char* buffer_ptr, int buffer_len) {
    int c;

    for (int i = 0; i < buffer_len; i++) {
        c = fgetc(f_data);
        if (c == EOF || c == 0xFF) {
            // Pad the remainder with OFF so the PRU never reads stale data.
            for (int j = i; j < buffer_len; j++) {
                buffer_ptr[j] = '0';
            }
            return 1;
        }
        buffer_ptr[i] = (char)c;
    }

    return 0;
}

int main(int argc, char** argv) {
    FILE* f_data = NULL;
    // FILE* f_output_copy = NULL;
    int f_mem = 0;
    char *ping_ptr, *pong_ptr, *indicator;
    const char* data_filename = DEFAULT_DATA_FILENAME;
    const int buffer_size = getpagesize() * 32;
	// uint8_t curr_buff = PING;

    if (argc > 1 && argv[1] && argv[1][0] != '\0') {
        data_filename = argv[1];
    }
    
    printf("Using symbol file: %s\r\n", data_filename);
    f_data = fopen(data_filename, "r");
    if (!f_data) {
        printf("Failed to open \"%s\" in mode \"r\", exiting.\r\n", data_filename);
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
    ping_ptr  = (char *) mmap(0, buffer_size, PROT_READ|PROT_WRITE, MAP_SHARED, f_mem, PING_ADDR);
    pong_ptr  = (char *) mmap(0, buffer_size, PROT_READ|PROT_WRITE, MAP_SHARED, f_mem, PONG_ADDR);
    indicator = (char *) mmap(0, getpagesize(),    PROT_READ|PROT_WRITE, MAP_SHARED, f_mem, INDICATOR_ADDR);
    if(ping_ptr == NULL || pong_ptr == NULL || indicator == NULL)
    {
        printf("Failed to mmap().\r\n");
        return EXIT_FAILURE;
    }
    
    printf("Pre-filling both buffers.\r\n");
    
    printf("ping buffer\r\n");
    if (fill_buffer_from_file(f_data, ping_ptr, buffer_size)) {
        printf("Reached EOF on \"%s\".\r\n", data_filename);
        fclose(f_data);

        printf("holding open (CTRL+C to close and release held memory)\r\n");
        while(1); // Keep process alive so PRU can keep reading the preloaded buffer.

        return EXIT_FAILURE;
    }
    
    printf("pong buffer\r\n");
    if (fill_buffer_from_file(f_data, pong_ptr, buffer_size)) {
        printf("Reached EOF on \"%s\".\r\n", data_filename);
        fclose(f_data);

        printf("holding open (CTRL+C to close and release held memory)\r\n");
        while(1); // Keep process alive so PRU can keep reading the preloaded buffers.

        return EXIT_FAILURE;
    }
    
    // Indicator points to which buffer is currently in use by the PRU.
    // The PRU starts with the ping buffer.
    indicator[0] = PING;
    
    printf("pre-fill done\r\n");
        
    while(1) {
        
        while(indicator[0] == PING); // Wait for the ping buffer to be released.
        
        if (fill_buffer_from_file(f_data, ping_ptr, buffer_size)) {
            printf("Reached EOF on \"%s\".\r\n", data_filename);
            fclose(f_data);

            printf("holding open (CTRL+C to close and release held memory)\r\n");
            while(1); // Keep process alive after padding the final partial buffer with OFF symbols.

            return EXIT_FAILURE;
        }
        
        while(indicator[0] == PONG); // Wait for the pong buffer to be released.
        
        if (fill_buffer_from_file(f_data, pong_ptr, buffer_size)) {
            printf("Reached EOF on \"%s\".\r\n", data_filename);
            fclose(f_data);

            printf("holding open (CTRL+C to close and release held memory)\r\n");
            while(1); // Keep process alive after padding the final partial buffer with OFF symbols.

            return EXIT_FAILURE;
        }
    }

    printf("Exiting program.\r\n");
    return EXIT_SUCCESS;
}

