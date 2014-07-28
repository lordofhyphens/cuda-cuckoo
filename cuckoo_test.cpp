#include <hemi/hemi.h>
#include <hemi/array.h>
#include "cuckoo.h"

template <class T> HEMI_KERNEL(testCuckoo)(T* array, T item)
{
  
}
#define HT_SIZE 500
int main() 
{
  hemi::Array<int> hashTable(HT_SIZE, false);
  int testInsert = 5;
  HEMI_KERNEL_LAUNCH(testCuckoo, gridDim, blockDim, 0, 0, hashTable.writeOnlyPtr(), testInsert);
  return 0;
}
