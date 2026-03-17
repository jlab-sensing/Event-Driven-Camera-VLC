SYMBOL_FILE ?= s31_lux_sweep_1000Hz_symbols.txt

all: userspace

program: userspace
	sudo ./userspace $(SYMBOL_FILE)

userspace: userspace.c
	gcc -Wall userspace.c -o userspace

clean:
	rm userspace
