SYMBOL_FILE ?= s31_replication_1500Hz_30s_symbols.txt

all: userspace

program: userspace
	sudo ./userspace $(SYMBOL_FILE)

userspace: userspace.c
	gcc -Wall userspace.c -o userspace

clean:
	rm userspace
