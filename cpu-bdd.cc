/*
 * Prototype to work out organization of a parallel-based BDD with 
 * flat arrays.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h> 
#include "cuckoo.h"

// Node 0 is constant 0.
// Node 1 is constant 1.
// two hashtables, unique and computed.
// key is a signature for I/T/E. 

// signature function from Velev2014, two "magic numbers" are primes larger than SIZE.
inline unsigned int keyfunc(unsigned I, unsigned T, unsigned E) { return (unsigned int)((I*600011+T)*6000023+E); }


void ITE_down(unsigned F, unsigned int G, unsigned int H);
void ITE_up();

// recursion stack to collect arguments for the stack
// Contains the I,T,E arguments needed. 
struct recursion {
	unsigned int* I;
	unsigned int* T;
	unsigned int* E;
	// original place
	unsigned int* key; 
};

// Storage arrays, static here, not planned to keep static
unsigned int g_bdd__variable[SIZE]; // F
unsigned int g_bdd__uid_high[SIZE]; // G?
unsigned int  g_bdd__uid_low[SIZE]; // H


// recursion stack arrays.
unsigned int  stk_i[SIZE];
unsigned int  stk_t[SIZE];
unsigned int  stk_e[SIZE];
unsigned int  stk_0[SIZE];
unsigned int  stk_1[SIZE];
unsigned char valid[SIZE];

// basic hashtables 
unsigned long long unique[int(SIZE*1.5)];

// computed table we can use a different failure mode (so long as it's in the unique table) - just evict at the end of the chain. 
unsigned long long computed[int(SIZE*1.5)];

// count of necessary ITEs to do at level i
unsigned int cont[SIZE];
// Position of first ITE operator at level i
unsigned int pos[SIZE];

int main() {

	// Initialization 
	load_hashvals(5); // fixed hash values for testing

	// zero out both the unique and computed tables.
	for (int i = 0; i < SIZE*1.5; i++) {
		unique[i] = make_entry(EMPTY_KEY,0);
		computed[i] = make_entry(EMPTY_KEY,0);
	}

	// hardcoded function F = ab + ac => F = a(b+c)
	// a is var 2, b is var 3, c is var 4.
	// Constant 1 is 'var 1'
	// Constant 0 is 'var 0'
	unsigned int j = 0;
	unsigned int key_0 = keyfunc(0,0,0);
	insert_hash(unique,key_0, j++);
	g_bdd__variable[0] = 0;
	g_bdd__uid_high[0] = 0;
	g_bdd__uid_low[0] = 0;

	unsigned int key_1 = keyfunc(1,1,1);
	insert_hash(unique,key_1, j);
	g_bdd__variable[j] = 1;
	g_bdd__uid_high[j] = 1;
	g_bdd__uid_low[j] = 1;
	j++;

	return 0;
}


void ITE_down(unsigned int F, unsigned int G, unsigned int H) {
	// check to see if this is a terminal case
	// Terminal cases: 
	// 1, 1, 1; 0;0;0
	// Constant 1/0.
	if (F <= 1; G <= 1 && H <= 1) {
		// this is a constant node.
	}
	// X, 0, 1; X, 1, 0;
	// This is right before a leaf node.

	// Has this been done and cached?
	
	// Queue up an ITE for the next level.
}

void ITE_up() {
	// Scan the list of ITEs in this recursion level, merging those that 
	// have the same signature.
	// Then get T/E, insert/retrieve (v, T, E), and then insert computed table entry ({F,G,H},R).
	// Put R (the unique table entry for this node) into the data list for the next level up.
	
};
