/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_CORE_DEVICE_H_
#define SINGA_CORE_DEVICE_H_

#include <type_traits>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <iostream>

#include "singa/singa_config.h"
#include "singa/core/common.h"
#include "singa/core/memory.h"
#include "singa/core/scheduler.h"
#include "singa/proto/core.pb.h"

#ifdef USE_CUDA
#include "singa/utils/cuda_utils.h"
#endif // USE_CUDA

#ifdef USE_OPENCL
#include "singa/utils/opencl_utils.h"
#endif // USE_OPENCL

#ifdef USE_MKLDNN
#include "singa/utils/mkldnn_utils.h"
#endif  // USE_MKLDNN

using std::vector;
using std::string;
using std::function;
using std::shared_ptr;
typedef long long LL;
using namespace std;
namespace singa {
    class Block;
    struct InfoBlock{
        Block* ptr;
        size_t size;
        int operation_type;
        int idx;
        LL t;
        int execCnt;
        bool fake=false;
        InfoBlock(Block* p, size_t s, int op, int i,LL tt,int cnt,bool f=false)
                :ptr(p),size(s),operation_type(op),idx(i),t(tt),execCnt(cnt),fake(f){}
        bool operator == (const InfoBlock& rhs) const {
            return size == rhs.size && operation_type == rhs.operation_type;// && ptr == rhs.ptr;
        }
    };
/// Allocate memory and execute Tensor operations.
/// There are three types of devices distinguished by their programming
/// languages, namely cpp, cuda and opencl.
class Device {
  public:
  // Device() = default;
  virtual ~Device() {}
  /// Constructor with device ID, num of executors (e.g., cuda streams),
  /// max mem size to use (in MB)
  Device(int id, int num_executors);

  virtual void SetRandSeed(unsigned seed) = 0;

  /// Called by Tensor.
  Block* NewBlock(int size);

  /// Called by Tensor.
  void FreeBlock(Block* block);
    void AppendInfo(Block* block,int type);
  /// Return the size (bytes) of memory in use
  /// TODO(wangwei) override this function for all devices.
  virtual size_t GetAllocatedMem() {
    return 0u;
  }

  /// Copy data within or across devices.
  virtual void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                      CopyDirection direction, int dst_offset, int src_offset);

  void CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                           size_t dst_offset = 0);
  /// Submit the operation to the device, which may execute it right now or
  /// delay it depending on the scheduler.
  void Exec(function<void(Context*)>&& fn, const vector<Block*> read_blocks,
                    const vector<Block*> write_blocks,
                    bool use_rand_generator = false);
    virtual void* UpdateGpuPtr(const Block* block_ptr){
        cout << "don't use device updateGpuPtr" << endl;
        return nullptr;
    }
    virtual int Get_debug_status(){
        cout << "should not use device get debug status" << endl;
        return 0;
    };

  // Wait for one event.
  // void WaitFor();
  int const READ=2,WRITE=4,MALLOC=1,FREE=-1;
    int delay=100;
  /// wait for all operations submitted to this device.
  void Sync();

  /// Return the programming language for this device.
  LangType lang() const {
    return lang_;
  }

  virtual std::shared_ptr<Device> host() const { return host_;}

  Context* context(int k) {
    return &ctx_;
  }

  int id() const { return id_; }
    virtual void Append(InfoBlock b)=0;

 private:
  Device() {};

 protected:
  /// Execute one operation on one executor.
  virtual void DoExec(function<void(Context*)>&& fn, int executor) = 0;

  virtual void CopyToFrom(void* dst, const void* src, size_t nBytes,
                          CopyDirection direction, Context* ctx) = 0;

  /// Allocate device memory.
  virtual void* Malloc(int size) = 0;

  /// Free device memory.
  virtual void Free(void* ptr) = 0;

 protected:
  int id_ = 0;
  int num_executors_ = 0;
  unsigned seed_ = 0;
  int execnt=0;
  // Scheduler* scheduler_ = nullptr;
  // VirtualMemory* vm_ = nullptr;
  /// Programming language type, could be kCpp, kCuda, kOpencl
  LangType lang_;
  // SafeQueue<Operation> op_queue_;
  // SafeQueue<Operation> op_log_;
  /// The host device
  std::shared_ptr<Device> host_;
  // TODO(wangwei) define multiple contexts, one per executor
  Context ctx_;
};
    struct BlockMeta{
        /*
         meta of swapping memory blocks
         */
        bool vis = false;
        bool out = false;
        bool synin = false;
        bool synout = false;
        Block* block_ = nullptr;
        void* data_ = nullptr;
        void* cpu_ptr = nullptr;
        int operation_type;
        size_t size = 0;
        cudaEvent_t out_event;
        cudaEvent_t in_event;
        cudaStream_t out_stream;
        cudaStream_t in_stream;
        //BlockMeta(bool v=false,bool o=false,bool si= false,bool so= false):
        //vis(v),out(o),synout(so),synin(si){}
    };

    struct SwapBlock{
        /*
        meta of candidate blocks
        */
        Block* ptr;
        string cat; //sub category of the candidate blocks, read-read, write-read, etc.
        int name;
        size_t size;
        //index of last read/write before swap out, and first read/write after swap in
        int r_idx; //out idx
        int d_idx; //in idx
        //index of last read/write before swap out, and first read/write after swap in
        LL r_time; // out time
        LL d_time; //in time
        int r_idx_ready; //r_idx + buffer
        int operation_type=0;
        //below are index and time for scheduling
        int idx_out_start  = 0;
        int idx_out_end = 0;
        int idx_in_end = 0;
        int idx_in_start = 0;
        LL t_out_start = 0;
        LL t_out_end = 0;
        LL t_in_end  = 0;
        LL t_in_start = 0;

        SwapBlock(Block* p, size_t s, int idx_out_start, int idx_in_end, LL t_out_start, LL t_in_end):
                ptr(p), size(s), r_idx(idx_out_start),d_idx(idx_in_end),r_time(t_out_start), d_time(t_in_end) {}
    };
/// Device able to Swap memory between Nvidia GPU and CPU
    class SwapGPU : public Device {
    public:
        ~SwapGPU();
        /// Construct the device using default mem pool setting.
        SwapGPU(int id = 0);
        /// Construct the device given the physical device ID and memory pool.
        SwapGPU(int id, std::shared_ptr<DeviceMemPool> pool);

        void SetRandSeed(unsigned seed) override;
        size_t GetAllocatedMem() override;
        int Get_debug_status() override{return DEBUG;};


    protected:
        void DoExec(function<void(Context*)>&& fn, int executor) override;

        void CopyToFrom(void* dst, const void* src, size_t nBytes,
                        CopyDirection direction, Context* ctx) override;

        /// Allocate cpu memory.
        void* Malloc(int size) override;

        /// Free cpu memory.
        void Free(void* ptr) override;

        //Append at every index: free, read, mutable
        void Append(InfoBlock b) override;

        //append info after Malloc, as Block* is not available till Malloc() done.
        void AppendAfterMalloc(Block* block,void* data_ptr,size_t size);

        //Detection and Plan
        void DetectionIteration();


        //entire plan, from SelectBlock() to Scheduling(), BuildMetaTables()
        void Plan();

        //block selection algo
        vector<SwapBlock> SelectBlock(vector<SwapBlock>&vec_swap,vector<LL> temp_load,LL mem_limit);

        //schedule algo
        void StickToLimit(vector<SwapBlock>&vec_swap_selct, vector<LL>&vec_load_temp,LL &overhead,LL mem_limit,string mode);

        //make tables table_sched and table_meta
        void BuildMetaTables(vector<SwapBlock>&vec_swap_selct);

        //update table_meta, during Append()
        void UpdateMetaTables(Block* block_ptr);

        //swap/sync during Append()
        void DeploySwap();

        //exec DelpoySwap
        void DeploySwapOut(int relative_counter);
        void DeploySwapIn(int relative_counter);

        //load profile as per synchronous swap.
        vector<LL> GetIdealLoad(vector<LL>vec_load,vector<SwapBlock> vec_swap_selct);

        //in case gpu ptr wrong, updated it after swap_in ad hoc
        void* UpdateGpuPtr(const Block* block_ptr) override;

        //Swap Synchronous, for early iterations
        void SwapOutSynchronous(const Block* block_ptr);
        void SwapInSynchronous(const Block* block_ptr);

        //Swap asynchronous, for middle iteraions
        void SwapOut(const int idx);
        void SwapIn(const int idx);
        void SwapOutSyn(const int idx);
        void SwapInSyn(const int idx);
        int shiftForConflict(int idx,int inc);

        //LL update_accum(int i,LL accum,int neg);
        //LL check_accum(int i,LL accum,int neg);
        LL SwapOutTime(size_t size);
        LL SwapInTime(size_t size);



    private:
        void Setup();
        //map<int,BlockMeta>table_meta;
        static const int MAXN=2000000;
        BlockMeta table_meta[MAXN];
        vector<int> table_sched[6][MAXN];
        //map<int,vector<int>>table_sched[4];

        map<Block*,int>removed;
        map<int,int>findSmoothL;
        map<int,int>findSmoothR;
        map<Block*,int>blockVis;

        bool overheadvis[MAXN];
        //map<int,bool>overheadvis;
        int rfornot=0;
        int recomputetype=1;
        //map<const Block*,BlockMeta>table_block_meta; //for measure speed only.
        //map<const Block*, int>table_not_at_device;  //int refers to its r_idx of the block/meta
        //map<int,std::tuple<int,int,int,int>>table_sched; // changed to with sync_r_idx

        //vec_block
        vector<InfoBlock>vecBlock;
        vector<SwapBlock>removeBlock;
        vector<string>vec_block; //iterations for Detection, i.e. detect iterations.
        vector<string>vec_block_fresh; //iterations that are used for Planning,
        vector<string>vec_block_mf; //iterations used to construct pool
        vector<LL>realload;
        vector<LL>accload;
        vector<LL>accSmoothLoad;
        vector<LL>swapload;
        vector<LL>origin_load; //3 iteration load, for planning.
        vector<InfoBlock>vec_run;
        vector<int>operation_sequence; //sequence of operations of one middle iteration
        vector<size_t>size_sequence; //size of all operations of one middle iteration
        vector<LL>poolvec;

        int async_swap_flag = 0; //0 for sync, 1 for async.
        int past_test_flag = 0; //0 means need to test, 1 means no need test anymore.
        int global_index = 0; //global counter, index, add 1 after each Malloc/Free/read/write.
        int global_index_threshold = -1;
        int iterlen = 0;
        int iter2 = 0; //index of start of 2nd iteration
        int fastinterval = 0; //index of start of 5th iteration
        int fastiter = -1;
        int number_of_swap_blocks=0;
        int remove_limit=22;
        int mode_type=0;
        int ignorefactor=10;
        int openremovedirect=0;
        string outputfile="resnetswap";
        double swap_factor=1.0;
        int syncfactor=0;
        LL deploytime=0;
        int justrun=0;
        int swapMemOp=1;
        int stopswap;
        int recompute=0;
        int faketrain=0;
        int beginremove=200;
        int delayremove=200;

        double swapoutcof=0.0756,swapoutbias=47200;
        double swapincof=0.0823,swapinbias=9700;

        //design specs
        double mem_limit_ratio = 0.70;
        LL smallest_block = (1<<22); //1 MB
        int data_buffer = 4; // used to control readyIdx
        int mutable_data_buffer = 6;
        LL max_load;
        int max_idx;
        LL total_swap_in_time = 0;
        LL total_swap_out_time = 0;
        LL temp_time = 0;
        LL temp_time_baseline; //vec_run[0] time


        map<int,bool>recomputeUsedBlocks;
        vector<int>recomputeFileSelect;
        //map<Block*,bool>useBlockAddress;
        int maxRNum=0;
        int setmaxRNum=0;
        int setLastRNum=0;
        int Refile=0;
        int overlapcr = 0;
        int lastrecompute=1;
        int iterlen_threshold = 1000;
        int DEBUG;
        LL maxnoswapload=0;
        LL maxswapload=0;
        LL maxpoolsize=0;
        LL globalmaxpoolsize=0;

        int recomRELU=1,recomBN=2,recomReLUBN=3;
        int recomCONV=4,recomPOOL=5,recomBNCONV=6;
        int recomALL=7,recomPBNCONV=10;

        friend class Block;

    private:
        shared_ptr<DeviceMemPool> pool_;
    };
/// a singleton CppDevice as the host for all devices.
extern std::shared_ptr<Device> defaultDevice;

/// Represent a CPU device which may have multiple threads/executors.
/// It runs cpp code.
class CppCPU : public Device {
 public:
  ~CppCPU();
  CppCPU();

  std::shared_ptr<Device> host() const override { return defaultDevice;}
  void SetRandSeed(unsigned seed) override;

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;
    void Append(InfoBlock b) override{};
};


// Implement Device using OpenCL libs.
// class OpenclDevice : public Device { };

#ifdef USE_CUDA
// Represent a Nvidia GPU which runs cuda code.
class CudaGPU : public Device {
 public:
  ~CudaGPU();
  /// Construct the device using default mem pool setting.
  CudaGPU(int id = 0);
  /// Construct the device given the physical device ID and memory pool.
  CudaGPU(int id, std::shared_ptr<DeviceMemPool> pool);

  void SetRandSeed(unsigned seed) override;
  size_t GetAllocatedMem() override;

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;
  void Append(InfoBlock b) override{};

 private:
  void Setup();

 private:
	shared_ptr<DeviceMemPool> pool_;
};

/// CudaCPU which uses cudaMallocHost to allocate pinned memory for host.

#endif  // USE_CUDA

#ifdef USE_OPENCL

// Implement Device using OpenCL libs.
class OpenclDevice : public singa::Device {
public:

  // TODO: Constructor arguments to consider:
  // Path to kernel sources?
  // Select only certain device types?
  OpenclDevice(int id = 0, int num_executors = 1);
  ~OpenclDevice();

// Overridden, inherited methods
  void SetRandSeed(unsigned seed) override;

  virtual void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                      CopyDirection direction, int dst_offset = 0,
                      int src_offset = 0) override;

protected:
  /// The OpenCL device that this object represents.
  /// Each OpenclDevice contains exactly one cl::Device for the lifetime of the
  /// object.
  viennacl::ocl::device this_device;

  /// Each OpenclDevice has one OpenCL context. It is created along with the
  /// creation of this object.
  viennacl::ocl::context vcl_ctx;

  /// Searches the given paths for all .cl files and builds
  /// OpenCL programs, then stores them in the Kernels map.
  void BuildPrograms();

// Overridden, inherited methods.

  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx = nullptr) override;

  /// Allocates memory on this OpenCL device
  /// by creating and returning an empty cl::Buffer object.
  /// with the indicated size.
  void* Malloc(int size) override;

  /// Converts the void pointer into a Buffer object, then deletes the object.
  /// This has the effect of freeing up device memory.
  void Free(void* ptr) override;
  void Append(InfoBlock b) override{};

private:

  static const std::string cl_src_path;
};
#endif  // USE_OPENCL

/// This class queries all available calculating devices on a given machine
/// grouped according to manufacturer or device drivers. All methods should be static.
/// If CUDA or OPENCL are not enabled, then the respective related methods should
/// return something that indicates their absence (for example, 0 devices);
/// however they should always be available regardless of compile-time switches.
class Platform {
public:

  /// Return the default host device
  static std::shared_ptr<Device> GetDefaultDevice() {
    return defaultDevice;
  }

#ifdef USE_CUDA
  /// Return the number of total available GPUs
  static int GetNumGPUs();

  /// Return the device IDs of available GPUs.
  /// TODO(wangwei) return the IDs according to free memory in decending order
  static const std::vector<int> GetGPUIDs();

  static const std::pair<size_t, size_t> GetGPUMemSize(const int device);

  /// Return the memory of a GPU <free, total>
  static const std::vector<std::pair<size_t, size_t>> GetGPUMemSize();

  /// Return a string containing all hardware info, e.g., version, memory size.
  static const std::string DeviceQuery(int id, bool verbose = false);

  /// Create a set of CudaGPU Device using 'num_devices' free GPUs.
  static const std::vector<std::shared_ptr<Device>>
  CreateCudaGPUs(const size_t num_devices, size_t init_size = 0);

  /// Create a set of CudaGPU Device using given GPU IDs.
  static const std::vector<std::shared_ptr<Device>>
  CreateCudaGPUsOn(const std::vector<int> &devices, size_t init_size = 0);
  
  /// This function is implementd by Caffe (http://caffe.berkeleyvision.org/).
  /// This function checks the availability of GPU #device_id.
  /// It attempts to create a context on the device by calling cudaFree(0).
  /// cudaSetDevice() alone is not sufficient to check the availability.
  /// It lazily records device_id, however, does not initialize a
  /// context. So it does not know if the host thread has the permission to use
  /// the device or not.
  ///
  /// In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  /// or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  /// even if the device is exclusively occupied by another process or thread.
  /// Cuda operations that initialize the context are needed to check
  /// the permission. cudaFree(0) is one of those with no side effect,
  /// except the context initialization.
  static bool CheckDevice(const int device_id);
#endif // USE_CUDA

#ifdef USE_OPENCL

  const int GetNumOpenclPlatforms();
  
  const int GetNumOpenclDevices();
  
  static const std::shared_ptr<Device> GetDefaultOpenclDevice();

  /// Create a \p num_devices set of valid OpenCL devices, regardless of
  /// platforms.  If there are fewer valid devices than requested, then this
  /// method will return as many as possible. If OpenCL is not in use, this
  /// method will return an empty array.
//  static const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const size_t num_devices);

  /// Create a set of valid OpenCL devices, regardless of platforms, assigning
  /// \p id to each device in sequence.
  /// If there are fewer valid devices than requested, then this method will
  /// return as many as possible.
  /// If OpenCL is not in use, this method will return an empty array.
//  const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const vector<int> &id);
#endif // USE_OPENCL

};


}  // namespace singa

#endif  // SINGA_CORE_DEVICE_H_
