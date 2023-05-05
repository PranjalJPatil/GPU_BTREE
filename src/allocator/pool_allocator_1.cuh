/*Copyright(c) 2020, The Regents of the University of California, Davis.            */
/*                                                                                  */
/*                                                                                  */
/*Redistribution and use in source and binary forms, with or without modification,  */
/*are permitted provided that the following conditions are met :                    */
/*                                                                                  */
/*1. Redistributions of source code must retain the above copyright notice, this    */
/*list of conditions and the following disclaimer.                                  */
/*2. Redistributions in binary form must reproduce the above copyright notice,      */
/*this list of conditions and the following disclaimer in the documentation         */
/*and / or other materials provided with the distribution.                          */
/*                                                                                  */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.*/
/*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,  */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT */
/*NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR*/
/*PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, */
/*WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) */
/*ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        */
/*POSSIBILITY OF SUCH DAMAGE.                                                       */
/************************************************************************************/
/************************************************************************************/

#pragma once
#include <stdio.h>

#include <cstdint>
#include<poggers/beta/slab_one_size.cuh>
#define NUM_ALLOCS 31250000+15000000
#define ALLOC_SIZE 128
using namespace beta::allocators;

using slab_one_size = one_size_slab_allocator<4>; 

class PoolAllocator_1 {
 public:
  PoolAllocator_1() {}
  ~PoolAllocator_1() {}
  void init() {
    ptr = slab_one_size::generate_on_device(NUM_ALLOCS, ALLOC_SIZE);
    cudaDeviceSynchronize();
  }
  void free() {
    slab_one_size::free_on_device(ptr);
    cudaDeviceSynchronize();
    //CHECK_ERROR(memoryUtil::deviceFree(d_pool));
    //CHECK_ERROR(memoryUtil::deviceFree(d_count));
  }

  double compute_usage() {
    return 0;
  }

  PoolAllocator_1& operator=(const PoolAllocator_1& rhs) {
   // d_pool = rhs.d_pool;
   // d_count = rhs.d_count;
    //return *this;
    ptr = rhs.ptr;
    return * this;
  }
  __device__ void createContext(){
      ptr->create_local_context();
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ AddressT allocate() {
    return ptr->malloc_ofset();
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ uint32_t* getAddressPtr(AddressT& address) {
    return (uint32_t *)ptr->get_ptr_from_offset(address);
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ void freeAddress(AddressT& address) {
    ptr->free(ptr->get_ptr_from_offset(address));
  }

  __host__ __device__ uint32_t getCapacity() { return NUM_ALLOCS; }

  __host__ __device__ uint32_t getOffset() { return 0; }

 private:
    slab_one_size *ptr;

};