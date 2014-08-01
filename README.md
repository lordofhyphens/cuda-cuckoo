cuda-cuckoo
===========

Cuckoo hashtable library for CUDA C. Based on chapter by D. Alacantra in "GPU Computing Gems - Jade Edition"

Why hemi? 

I think Hemi's strengths are well-explaned by its author, Mark Harris. Since this project is a building block to something greater, I need to be able to do quick sanity tests on the host-side. 

Great, how do I use this thing?

You will need to ensure that hemi is on your include path. Pulling the submodule and pointing to that is the simplest solution. Right now, pulling from harrism/hemi won't work as I've added some functionality that isn't in his repo yet (specifically, atomics).

Well, there's always looking at cuckoo_test.cpp for an example usage. But if tl;dr strikes you, here's the version with extra explanation attached:

The CuckooTable class itself is the center of all of this. You need to allocate a few things before you instantiate it, though, and pass them to the constructor.

1) You need an array of unsigned long long on the device. This is the actual memory location for the hashtable itself. 

2) You need two unsigned int arrays of at least length 5 on the device. Preload them with random integers. These form the basis of your hash functions. They're a bit weak, but it's been shown that it's sufficient in most cases. 

3) Because our table can fail to insert (and thus require a complete rebuild with new hash functions), we need some way of determining this case. As of now, this is being done with host-pinned boolean memory (zero-copy). It should only add a little latency to the kernel execution. Kernels using the cuckoo table will need to check the status before continuing. 

The reason for all this is that, at least with hemi::array, the structs themselves aren't portable across the device and host. Instead hemi::array exposes pointers to allow for migrating data to/from the host.

Gotchas: 

If you're using hemi::array for the values, you have to set up the host memory area (particularly the hash functions and table space) before creating a CuckooTable. This is because hemi::array determines which is the "active" copy (host or device) when one of the ptr() functions are called. To work around this, you would need to force a copy-back and then call devicePtr() to ensure everything is sync'd.
