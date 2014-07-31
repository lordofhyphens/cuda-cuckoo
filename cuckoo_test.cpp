#include <hemi/hemi.h>
#include <hemi/array.h>
#include "cuckoo.h"

#define HT_SIZE 500

HEMI_KERNEL(testCuckooInit)(const unsigned long long* table, unsigned SIZE) 
{
  for (int i = 0; i < SIZE; i++)
  {
    assert(table[i] == HEMI_CONSTANT(EMPTY_SLOT));
  }
  printf("Array is empty.\n");
}

// short test function to 
HEMI_KERNEL(testCuckooInsert)(unsigned long long* array, unsigned key, unsigned item, HashFuncs hf)
{
  bool test;
  unsigned long long testEntry;
  unsigned long long origEntry = make_entry(key, item);
  test = insert_hash(array, key, item, HT_SIZE, hf);
  testEntry = retrieve_hash(array, key, item, HT_SIZE, hf);
  printf("Insert of (%d, %d): %s\n", key, item, (test == true ? "Success" : "Failure"));
  printf("Hash value retrieved: %lx - (%x, %x) -- %lx\n", testEntry, get_key(testEntry), get_value(testEntry), origEntry);
  assert(testEntry == origEntry);
}

int main() 
{
  hemi::Array<unsigned long long> hashTable(HT_SIZE, false);
  // create an obj for the hash functions, pointing at
  // the correct location.
  printf("Initializing data...\n");
  load_hashvals();
  HashFuncs hf = HashFuncs(hash_a.readOnlyPtr(), hash_b.readOnlyPtr()); 


  unsigned gridDim = 1, blockDim = 1;
  int testInsert = 5;
  int key = 23423;
  initCuckooArray(hashTable.writeOnlyHostPtr(), HT_SIZE);
  HEMI_KERNEL_LAUNCH(testCuckooInit, gridDim, blockDim, 0, 0, hashTable.readOnlyPtr(), HT_SIZE);
  HEMI_KERNEL_LAUNCH(testCuckooInsert, gridDim, blockDim, 0, 0, hashTable.ptr(), key, testInsert, hf);
  return 0;
}
