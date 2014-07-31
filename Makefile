# Change this to point to wherever CUDA is installed on your local system.
CUDA_PATH := /opt/net/apps/cuda
NVCC=${CUDA_PATH}/bin/nvcc
CXX=g++

CC_FLAGS := -I./hemi -I${CUDA_PATH}/include

NVCC_FLAGS := ${CC_FLAGS} -arch=sm_21 
ARCH := $(shell getconf LONG_BIT)

LIB_FLAGS_32 := -L$(CUDA_PATH)/lib
LIB_FLAGS_64 := -L$(CUDA_PATH)/lib64

LIB_FLAGS := $(LIB_FLAGS_$(ARCH)) -lcudart

.PHONY: cuckoo clean
all: cuckoo

clean: cuckoo_host_g++ cuckoo_host_nvcc cuckoo_device 
	rm -f $^
cuckoo: cuckoo_host_g++ cuckoo_host_nvcc cuckoo_device 
	@for prog in $^; do \
		echo "---------\nRunning $$prog\n---------"; \
		./$$prog; \
	done

cuckoo_device: cuckoo_test.cpp cuckoo.h
	@${NVCC} ${LIB_FLAGS} -x cu ${CC_FLAGS} ${NVCC_FLAGS} $< -o $@

cuckoo_host_g++: cuckoo_test.cpp cuckoo.h
	@${CXX} ${LIB_FLAGS} ${CC_FLAGS} $< -o $@

cuckoo_host_nvcc: cuckoo_test.cpp cuckoo.h
	@${NVCC} ${LIB_FLAGS} ${CC_FLAGS} ${NVCC_FLAGS} $< -o $@
