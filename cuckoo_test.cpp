#include <hemi/hemi.h>
#include <hemi/array.h>
#include "cuckoo.h"

#define HT_SIZE 500
#define DEBUG

HEMI_KERNEL(testCuckooInit)(const unsigned long long* table, unsigned SIZE) 
{
  for (int i = 0; i < SIZE; i++)
  {
    assert(table[i] == HEMI_CONSTANT(EMPTY_SLOT));
  }
  printf("Array is empty.\n");
}

// short test function to 
HEMI_KERNEL(testCuckooInsert)(unsigned key, unsigned item, CuckooTable hf)
{
  bool test;
  unsigned long long testEntry;
  unsigned long long origEntry = make_entry(key, item);
  test = hf.insert_hash(key, item);
  testEntry = hf.retrieve_hash(key, item);
  //printf("Insert of (%d, %d): %s\n", key, item, (test == true ? "Success" : "Failure"));
  //printf("Hash value retrieved: %lx - (%x, %x) -- %lx\n", testEntry, get_key(testEntry), get_value(testEntry), origEntry);
  assert(testEntry == origEntry);
}
HEMI_KERNEL(testCuckooInsertTooMany)(unsigned key, unsigned item, CuckooTable hf)
{
  bool test;
  unsigned long long testEntry;
  unsigned long long origEntry = make_entry(key, item);
  test = hf.insert_hash(key, item);
  testEntry = hf.retrieve_hash(key, item);
  if (!test)
  {
    printf("Insert of (%d, %d): %s\n", key, item, (test == true ? "Success" : "Failure"));
  }
 // printf("Insert of (%d, %d): %s\n", key, item, (test == true ? "Success" : "Failure"));
 // printf("Hash value retrieved: %lx - (%u, %u) -- %lx\n", testEntry, get_key(testEntry), get_value(testEntry), origEntry);
}

int main() 
{
  hemi::Array<unsigned long long> hashTable(HT_SIZE, false);
  // create an obj for the hash functions, pointing at
  // the correct location.
  // Apparently, you have to load data into the host before calling the generic ptr() functions.
  // This probably has something to do with the lazy update.
  printf("Initializing data...\n");
  load_hashvals();


  unsigned gridDim = 1, blockDim = 1;
  int testInsert = 5;
  int key = 23423;
  initCuckooArray(hashTable.writeOnlyHostPtr(), HT_SIZE);
  CuckooTable hf = CuckooTable(hash_a.readOnlyPtr(), hash_b.readOnlyPtr(),hashTable.ptr(), HT_SIZE, rebuild.ptr(),rebuild.ptr(hemi::host)); 
  printf("Data initialized.\n");
  HEMI_KERNEL_LAUNCH(testCuckooInit, gridDim, blockDim, 0, 0, hashTable.readOnlyPtr(), HT_SIZE);
    #ifdef HEMI_CUDA_COMPILER
      cudaDeviceSynchronize();
    #endif
  printf("Testing insert.\n");
  HEMI_KERNEL_LAUNCH(testCuckooInsert, gridDim, blockDim, 0, 0, key, testInsert, hf);
    #ifdef HEMI_CUDA_COMPILER
      cudaDeviceSynchronize();
    #endif


  printf("Testing too many inserts (rebuild flag).\n");
  for (int i = 0; i < hf.size+1; i++) {
    HEMI_KERNEL_LAUNCH(testCuckooInsertTooMany, gridDim, blockDim, 0, 0, i * 73, i, hf);
    
    #ifdef HEMI_CUDA_COMPILER
      cudaDeviceSynchronize();
    #endif
    if (rebuild.readOnlyPtr(hemi::host)[0] == true)
    {
      // we have run into a problem 
      printf("Detected insert failure for %dth element for table of size %d! Need to generate new functions and try again.\n", i, hf.size);
      i = HT_SIZE+1; continue;
    }
  }
  #ifdef HEMI_CUDA_COMPILER
    cudaDeviceSynchronize();
  #endif

  printf("Rebuild flag: %s\n", (rebuild.readOnlyPtr(hemi::host)[0] == true ? "True" : "False"));
  assert(rebuild.readOnlyPtr(hemi::host)[0] == true);
  return 0;
}
