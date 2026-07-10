# Prerequisites (Do Once)
1. `sudo ln -s /usr/bin/ /usr/share/ti/cgt-pru/bin`
2. Modify `~/.bashrc` and append `export PRU_CGT=/usr/share/ti/cgt-pru` to the end of the file.
    - `source ~/.bashrc` (Note: This step is only needed if you do not restart the BBB after modifying `~/.bashrc`. The `~/.bashrc` file is sourced upon BBB startup.)
3. Rename the pre-existing `/usr/lib/ti/pru-software-support-package/` folder to something else, and clone the v5.9.0 version. `sudo git clone --depth=1 --branch v5.9.0 git://git.ti.com/pru-software-support-package/pru-software-support-package.git`
# Usage

1. `make program`
2. `echo "your_message_here" > /dev/rpmsg_pru31 && cat /dev/rpmsg_pru31`
3. CTRL+C to cancel the `cat` program and repeat step 2 to continue sending more messages to the PRU.
