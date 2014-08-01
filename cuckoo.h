#ifndef _CUCKOO_H_
#define _CUCKOO_H_

#include <stdint.h>
#include <hemi/hemi.h>
#include <hemi/array.h>
#include <hemi/atomic.h>

#include <cstdlib>
#include <ctime>

// constants
HEMI_DEFINE_STATIC_CONSTANT(short int MAX_FUNCS, 5);
HEMI_DEFINE_STATIC_CONSTANT(short int MAX_ATTEMPTS,120);
HEMI_DEFINE_STATIC_CONSTANT(long long lg_prime,334214459);
HEMI_DEFINE_CONSTANT(unsigned long long EMPTY_SLOT, 0xffffffff00000000);

// rebuild flag for our array. If this is set true, there was a failure to add. 

HEMI_DEFINE_CONSTANT(unsigned int EMPTY_KEY,0xffffffff);

HEMI_DEV_CALLABLE_INLINE unsigned long long make_entry(unsigned int key, unsigned int value) { return ((((unsigned long long)key) << 32) + (value)); }
#define get_key(entry) ((unsigned)((entry) >> 32))
#define get_value(entry) ((unsigned)((entry) & 0xffffffff))

// hashfunction for table: h_i(k) = (a_i*k+b_i) % p % slots
// p is some large prime number
// slots is the number of spaces in the hashtable.

hemi::Array<unsigned int>hash_a(HEMI_CONSTANT(MAX_FUNCS), true);
hemi::Array<unsigned int>hash_b(HEMI_CONSTANT(MAX_FUNCS), true);
hemi::Array<bool> rebuild(1, true); 

class HashFuncs
{
  const unsigned* hash_a;
  const unsigned* hash_b;
  public: 
  HashFuncs(const unsigned* a, const unsigned* b) : hash_a(a), hash_b(b) { } 
  HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int hashfunc(unsigned key, unsigned slots, unsigned int n) const {
    return (hash_a[n] * key + hash_b[n]) % HEMI_CONSTANT(lg_prime) % slots;
  }

};

class CuckooTable
{
  const unsigned* hash_a;
  const unsigned* hash_b;
  bool* g_rebuild;
  bool* h_rebuild;
  public: 
   unsigned long long* table; 
  CuckooTable(const unsigned* a, const unsigned* b,  unsigned long long* t, bool* grb, bool* hrb) : hash_a(a), hash_b(b), table(t), g_rebuild(grb), h_rebuild(hrb) { } 
  HEMI_DEV_CALLABLE_INLINE_MEMBER unsigned int hashfunc(unsigned key, unsigned slots, unsigned int n) const {
    return (hash_a[n] * key + hash_b[n]) % HEMI_CONSTANT(lg_prime) % slots;
  }
  HEMI_DEV_CALLABLE_INLINE_MEMBER bool needsRebuild() const
  {
    #ifdef HEMI_DEV_CODE
      return *g_rebuild;
    #else
      return *h_rebuild;
    #endif
  }
  HEMI_DEV_CALLABLE_INLINE_MEMBER void setRebuild()
  {
    #ifdef HEMI_DEV_CODE
      *g_rebuild = true;
    #else
      *h_rebuild = true;
    #endif
  }

};
void load_hashvals(unsigned int seed = 0)
{
  rebuild.writeOnlyHostPtr()[0] = false;
  if (seed != 0) 
    srand(seed);
  else
    srand(time(NULL));
	for (int i = 0; i < HEMI_CONSTANT(MAX_FUNCS); i++) {
		hash_a.writeOnlyHostPtr()[i] = rand();
		hash_b.writeOnlyHostPtr()[i] = rand();
	}
}

HEMI_DEV_CALLABLE_INLINE unsigned long long int retrieve_hash(unsigned long long* table,unsigned int k,unsigned int v, unsigned int SIZE, const CuckooTable hash) {
	unsigned int location_0 = hash.hashfunc(k,SIZE,0);
	unsigned int location_1 = hash.hashfunc(k,SIZE,1);
	unsigned int location_2 = hash.hashfunc(k,SIZE,2);
	unsigned int location_3 = hash.hashfunc(k,SIZE,3);
	unsigned int location_4 = hash.hashfunc(k,SIZE,4);

	unsigned long long entry;
	if(get_key(entry = table[location_0]) != k)
		if(get_key(entry = table[location_1]) != k)
			if(get_key(entry = table[location_2]) != k)
				if(get_key(entry = table[location_3]) != k)
					if(get_key(entry = table[location_4]) != k)
						entry = make_entry(HEMI_CONSTANT(EMPTY_KEY), 0);
	if (get_value(entry) != v) 
		entry = make_entry(HEMI_CONSTANT(EMPTY_KEY), 0);
	return entry;
}
void initCuckooArray(unsigned long long* table, unsigned int SIZE)
{
  rebuild.writeOnlyHostPtr()[0] = false;
  for (int i = 0; i < SIZE; i++) 
  {
    table[i] = make_entry(HEMI_CONSTANT(EMPTY_KEY), 0);
  }
}
// returns whether or not we were successful.
HEMI_DEV_CALLABLE_INLINE bool insert_hash(unsigned long long* table, unsigned int k, unsigned int v, unsigned int SIZE, CuckooTable hash) {
	unsigned long long entry = make_entry(k,v); // initial value to place into the table
	if (retrieve_hash(table, k,v, SIZE, hash) == entry) return true; // already in the table
	unsigned int key = k;

	// initial location
	unsigned int location = hash.hashfunc(k, SIZE, 0);
  assert(k == get_key(entry));
  assert(v == get_value(entry));
  printf("Initial: Putting (%x, %x) into position %u.\n", k, v, location);

	for (int its = 0; its < HEMI_CONSTANT(MAX_ATTEMPTS); its++) {
		// insert new item and check for eviction
		// on gpu we use atomicexch, serial cpu just evicts and uses
		// a temp variable. MP cpu needs to have this part in a critical section.
		//
    printf("Trying to put (%x, %x) in location %u\n", get_key(entry), get_value(entry), location);
		entry = hemi::atomicExch(table+location,entry);
    #ifdef HEMI_DEV_CODE
      __syncthreads();
    #endif
    printf("(%x, %x) was in location %u\n", get_key(entry), get_value(entry), location);


		key = get_key(entry);
		if (key == HEMI_CONSTANT(EMPTY_KEY)) return true;
		// if we had an eviction, figure out the next function to use, 
		// round-robin.

    unsigned int location_0 = hash.hashfunc(k,SIZE,0);
    unsigned int location_1 = hash.hashfunc(k,SIZE,1);
    unsigned int location_2 = hash.hashfunc(k,SIZE,2);
    unsigned int location_3 = hash.hashfunc(k,SIZE,3);
    unsigned int location_4 = hash.hashfunc(k,SIZE,4);
		     if (location == location_0) location = location_1;
		else if (location == location_1) location = location_2;
		else if (location == location_2) location = location_3;
		else if (location == location_3) location = location_4;
		else location = location_0;

	}
  hash.setRebuild();
	return false; // too many attempts.
}

// just rebuilds the table itself. If the second int was an index, more problems.
HEMI_DEV_CALLABLE_INLINE void rebuild_table(unsigned long long* table, unsigned int SIZE, const CuckooTable hash) {
	for (unsigned i = 0; i < SIZE; i++) {
		if (table[i] != make_entry(HEMI_CONSTANT(EMPTY_KEY), 0)) // this can be distributed on the GPU, I think.
			insert_hash(table, get_key(table[i]), get_value(table[i]), SIZE, hash);
	}
}


#endif // _CUCKOO_H

