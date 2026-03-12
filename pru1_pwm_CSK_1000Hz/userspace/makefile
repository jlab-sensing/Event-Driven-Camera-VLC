all: userspace

program: userspace
	sudo ./userspace

userspace: userspace.c
	gcc -Wall userspace.c -o userspace

clean:
	rm userspace
