# About

This folder contains code for two programs, both of which are necessary to transmit a series of data symbols over an RGB LED.

# File Listing

`pru1_pwm/`
- `gen/` -- build files and PRU firmware
  - `...`
- `userspace/`
  - `data.txt` -- symbols to transmit, encoded as ASCII characters
  - `makefile`
  - `userspace.c` -- userspace program that copies `data.txt` into shared memory that the PRU can access
- `AM335x_PRU.cmd` -- linker command file for the PRU build process.
- `gpio_pru0_pru1.h` -- defines for PRU GPIO
- `main.c` -- PRU program that outputs three PWM channels that correspond with the symbols shared by the userspace program
- `makefile`
- `pwmss.h` -- modified version of `<sys_pwmss.h>` to provide complete bitfields for all registers
- `README.md`
- `resource_table_1.h` -- resource table for coordinating Rpmsg mailbox resources between the PRU and ARM
- `symbols.h` -- defines for symbol RGB combinations

# Prerequisites (Do Once)

1. `sudo ln -s /usr/bin/ /usr/share/ti/cgt-pru/bin`
2. Append `export PRU_CGT=/usr/share/ti/cgt-pru` to the end of `~/.bashrc`.
    - `source ~/.bashrc` (Note: This step is only needed if you do not restart the BBB after modifying `~/.bashrc`. The `~/.bashrc` file is sourced upon BBB startup.)
3. ~~Rename the pre-existing `/usr/lib/ti/pru-software-support-package/` folder to something else, and clone the v5.9.0 version. `sudo git clone --depth=1 --branch v5.9.0 git://git.ti.com/pru-software-support-package/pru-software-support-package.git`~~

# Usage

Terminal 1:
1. Navigate to the `userspace/` folder.
2. `make userspace`
3. Run userspace with a chosen symbol file:
   - Default: `sudo ./userspace`
   - Explicit file: `sudo ./userspace s31_pw_1p00ms_symbols.txt`
   - `make` shortcut: `make program SYMBOL_FILE=s31_pw_1p00ms_symbols.txt`
4. Default userspace symbol file is `s31_lux_sweep_1000Hz_symbols.txt` (14,000 symbols):
   - 2,000 OFF symbols (`0`)
   - 10,000 symbols alternating `4`/`0`
   - 2,000 OFF symbols (`0`)
5. Pulse-width sweep symbol files (all 56,000 symbols, built for 0.25 ms PRU symbol timing):
   - `s31_pw_1p00ms_symbols.txt`
   - `s31_pw_0p75ms_symbols.txt`
   - `s31_pw_0p50ms_symbols.txt`
   - `s31_pw_0p25ms_symbols.txt`
6. Keep this terminal open. When you are finished with sending all of your transmissions, CTRL+C to halt the program.

Terminal 2:
1. Navigate to the `pru1_pwm/` folder.
2. Build/load PRU firmware (symbol timing currently 0.25 ms): `make program`
3. Start one transmission (example uses 1.00 ms pulse file):
   - `sleep 1 && SYMLEN=$(wc -c < userspace/s31_pw_1p00ms_symbols.txt) && echo -n "$SYMLEN" > /dev/rpmsg_pru31 && cat /dev/rpmsg_pru31`
4. Observe diagnostic messages on the terminal, and use CTRL+C to cancel the `cat` program after the current transmission finishes.
5. To repeat with another pulse-width file, update only the filename in `SYMLEN=$(wc -c < userspace/<file>.txt)`.

# Miscellaneous

`cut -c1-100 data.txt | tr -d $'\n' > /dev/rpmsg_pru31 && cat /dev/rpmsg_pru31`

TIDA01555: example PRU-ARM shared memory with ping pong buffers

SPRACE8A: PRU read latencies
