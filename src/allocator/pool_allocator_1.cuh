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
#include<poggers/allocators/one_size_allocator.cuh>
#define NUM_ALLOCS 31250000
#define ALLOC_SIZE 128
using namespace poggers::allocators;

class PoolAllocator_1 {
 public:
  PoolAllocator_1() {}
  ~PoolAllocator_1() {}
  void init() {
    ptr = one_size_allocator::generate_on_device(NUM_ALLOCS, ALLOC_SIZE, 420);
    cudaDeviceSynchronize();
  }
  void free() {
    one_size_allocator::free_on_device(ptr);
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

  template<typename AddressT = uint32_t>
  __device__ __forceinline__ AddressT allocate() {
    return ptr->get_offset();
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ uint32_t* getAddressPtr(AddressT& address) {
    return (uint32_t *)ptr->get_mem_from_offset(address);
  }
  template<typename AddressT = uint32_t>
  __device__ __forceinline__ void freeAddress(AddressT& address) {
    ptr->free_offset(address);
  }

  __host__ __device__ uint32_t getCapacity() { return NUM_ALLOCS; }

  __host__ __device__ uint32_t getOffset() { return 0; }

 private:
    one_size_allocator *ptr;

};