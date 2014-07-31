#include <hemi/hemi.h>
#include <hemi/array.h>
#include "cuckoo.h"

template <class T> HEMI_KERNEL(testCuckoo)(T* array, T item, HashFuncs hf)
{
  
}
#define HT_SIZE 500
int main() 
{
  hemi::Array<int> hashTable(HT_SIZE, false);
  // create an obj for the hash functions, pointing at
  // the correct location.
  HashFuncs hf = HashFuncs(hash_a.readOnlyPtr(), hash_b.readOnlyPtr()); 

  unsigned gridDim = 1, blockDim = 1;
  int testInsert = 5;
  HEMI_KERNEL_LAUNCH(testCuckoo, gridDim, blockDim, 0, 0, hashTable.writeOnlyPtr(), testInsert, hf);
  return 0;
}
