#include <cuda.h>
#include <cstdio>
bool* rebuild_device;
bool* rebuild_host;

// need a way to check and abort the rest of a kernel with a flag. The bool
// should show up in L1 or L2 w/o any issues

__global__ void quickAbort(volatile bool* err)
{
  if (*err) 
    return; 
  *err = true;
}

int main() 
{
  cudaError_t status = cudaHostAlloc((void**)&rebuild_host, sizeof(bool),cudaHostAllocMapped);

  cudaHostGetDevicePointer((void **)&rebuild_device, (void *)rebuild_host, 0);
  if (status != cudaSuccess) 
    printf("Error.\n");
  memset(rebuild_device, 0, sizeof(bool));
  printf("%s\n", (rebuild_host ? "Flag" : "No Flag"));
  quickAbort<<<128,128>>>(rebuild_device);
  cudaDeviceSynchronize();

  printf("%s\n", (rebuild_host ? "Flag" : "No Flag"));
  cudaMemset(rebuild_device, 0, sizeof(bool));
  printf("%s\n", (rebuild_host ? "Flag" : "No Flag"));
  quickAbort<<<128,20128>>>(rebuild_device);
  cudaDeviceSynchronize();

  printf("%s\n", (rebuild_host ? "Flag" : "No Flag"));
  return 0;
}
