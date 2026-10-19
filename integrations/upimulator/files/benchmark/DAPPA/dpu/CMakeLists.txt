# task.c is replaced by the device.c of the DaPPA recipe being simulated
# (see integrations/upimulator/run.sh). The committed file is a placeholder.

set(CMAKE_C_COMPILER "/root/upmem-2023.2.0-Linux-x86_64/bin/dpu-upmem-dpurte-clang")
set(CMAKE_C_FLAGS "-w -O2 -S -DNR_TASKLETS=${NR_TASKLETS} -DSTACK_SIZE_DEFAULT=512")

file(GLOB_RECURSE SRCS *.c)

add_executable(DAPPA_device ${SRCS})
