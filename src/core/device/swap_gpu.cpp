//
// Created by csc on 1/19/19.
//

#include "singa/singa_config.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore
#include "singa/core/device.h"
#include "singa/utils/cuda_utils.h"
#include <sstream>
#include <assert.h>

using namespace std;

namespace singa {

const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                   cudaMemcpyDeviceToHost,
                                   cudaMemcpyDeviceToDevice};

struct sort_by_ptr_idx_ascending{
    inline bool operator() (const InfoBlock& struct1, const InfoBlock& struct2){
        return ((struct1.ptr<struct2.ptr)||((struct1.ptr==struct2.ptr)&&(struct1.idx<struct2.idx)));
    }
};
struct sort_by_didx_ascending_swap{
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2){
    return (struct1.d_idx<struct2.d_idx);
  }
};
struct sort_by_idx_descending_swap{
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2){
    return (struct1.d_idx>struct2.d_idx);
  }
};
struct sort_by_idx_ascending_swap{
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2){
    return (struct1.r_idx<struct2.r_idx);
  }
};
int SwapGPU::SwapOutTime(int size){
  //measured in 16 PCIe, pinned memory.
  return (0.0756 * size + 47200) * swap_factor;
}

int SwapGPU::SwapInTime(int size){
  //measured as per ncra ~ ncrd, 16 PCIe, pinned memory.
  return (0.0823 * size + 9700) * swap_factor;
}

void RepeatableTest(vector<InfoBlock>vecBlock, int &iterlen, int &location_of_2nd_iteration, int iterlen_threshold, int global_index ){
  /*
  repeatable test, input vector of int,
  in-place update max_legth (length of iteration)
  and location_of_2nd_iteration (where 2nd iteration starts)
  */
  int idx_range = (int)vecBlock.size();
  int threshold = iterlen_threshold;

  for (int i=0; i<idx_range;i++){
    if (iterlen>threshold)break;
    for (int len=threshold;2*len+i<idx_range;len++){
      if (iterlen>threshold)break;
      if(equal(vecBlock.begin()+i,vecBlock.begin()+i+len,vecBlock.begin()+i+len)) {
        iterlen = len;
        location_of_2nd_iteration = i;
      }
    }
  }
}

void UpdateLoad(vector<double>& vec_load,int start_idx, int end_idx, int plus_minus, int size,int iterlen){
  /*
  update load [start_idx, end_idx) by plus_minus*size
  */
  for (int i = start_idx+iterlen; i<end_idx+iterlen; i++){
    vec_load[i] = vec_load[i] + static_cast<double>(size) * plus_minus;
  }
}

pair<double,int> GetLoadPeak(vector<double>vec_load_test,int iterlen){
  /*
  return value and index of load peak
  */
  double max_load_test = 0;
  int max_idx_test = 0;
  for (int i = iterlen; i < iterlen*2; i++){
    if (max_load_test < vec_load_test[i]){
      max_load_test = vec_load_test[i];
      max_idx_test = i - iterlen;
    }
  }
  return std::make_pair(max_load_test,max_idx_test);
}



int SwapGPU::Detection(vector<InfoBlock>vecBlock,int &iterlen, int &location_of_2nd_iteration){
  /*
  test repeatability, detect iteration, and return global_index_threshold.
  */

  ///vec_str (vecBlock) to vec_opt_info, sort by ptr and idx.
  int idx_range = (int)(vecBlock.size());
  RepeatableTest(vecBlock,iterlen,location_of_2nd_iteration,iterlen_threshold,global_index);
  //Note here location_of_2nd_iteration not exactly start of one iteration,
  //adjust to nearly start of one by restricting "Malloc"
  int shift_counter = 0;
  for (int i=0;i<iterlen;i++){
    if (vecBlock[location_of_2nd_iteration+i].operation_type==1)shift_counter = i;
    else break;
  }
  location_of_2nd_iteration =location_of_2nd_iteration+shift_counter;

  if (iterlen<iterlen_threshold) {return -1;}
  else{
      cout << "iterlen" << iterlen << endl;
      cout << "location_of_2nd_iteration" << location_of_2nd_iteration << endl;
  }

  return global_index+iterlen-(global_index-location_of_2nd_iteration)%iterlen;
}


vector<SwapBlock> SwapGPU::SelectBlock(vector<SwapBlock>vec_swap,vector<double> temp_load,double mem_limit,string mode){

  vector<SwapBlock>vec_swap_selct;
  sort(vec_swap.begin(),vec_swap.end(),sort_by_idx_ascending_swap());
  for (int i=0; i<vec_swap.size() && i < number_of_swap_blocks; i++){
    vec_swap_selct.push_back(vec_swap[i]);
  }
  return vec_swap_selct;
}
int SwapGPU::check_accum(int i,int accum,int pos_neg){
    if(overheadvis[i])return accum;
    else{
        if(vec_run[i].operation_type == 1 || vec_run[i].operation_type == -1){
            return accum+vec_run[i].size*vec_run[i].operation_type*pos_neg;
        }
        else return accum;
    }
}
int SwapGPU::update_accum(int i,int accum,int pos_neg){
    if(overheadvis[i])return accum;
    else{
        overheadvis[i]=true;
        if(vec_run[i].operation_type == 1 || vec_run[i].operation_type == -1){
            //cout << " oper type?"<< vec_run[i].operation_type<< endl;
            //cout << " vec_run size:"<< vec_run[i].size << endl;
            //cout << "vec_run idx:"<<vec_run[i].idx << endl;
            return accum+vec_run[i].size*vec_run[i].operation_type*pos_neg;
        }
        else return accum;
    }
}

void SwapGPU::Scheduling(vector<SwapBlock>&vec_swap_selct, vector<double>&vec_load_temp,double &overhead,double mem_limit,string mode){
  if (mode == "stick-to-limit"){
    overhead=0;
    int accum = 0;
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
    int cnt=0;
    for (int i = 0;i < vec_run.size()&&i<=max_idx&&cnt<vec_swap_selct.size();i++){
        accum=update_accum(i,accum,1);
        //cout << " out accum:"<< accum << endl;
        if(i < vec_swap_selct[cnt].r_idx)continue;
        auto itm = vec_swap_selct[cnt];
        itm.idx_out_start = i;
        itm.t_out_start = vec_run[i].t;
        itm.t_out_end = vec_run[i].t + SwapOutTime(itm.size);
        while(vec_run[i].t < itm.t_out_end){
            if(check_accum(i+1,accum,1) < mem_limit && i+1<=max_idx){
                i++;
                accum=update_accum(i,accum,1);
                //cout << " accum " << accum << endl;
            }
            else{
                cout << "overheadi:" << i << endl;
                overhead += (itm.t_out_end-vec_run[i].t);
                break;
            }
        }
        accum -= itm.size;
        itm.idx_out_end=i;
        vec_swap_selct[cnt]=itm;
        cnt++;
    }
    int originalsize=vec_swap_selct.size();
    for(int i = cnt;i<originalsize;i++){
        cout << "remove ith element:" << i << endl;
        vec_swap_selct.erase(vec_swap_selct.begin() + i);
    }
    cout << "duration overhead swapout:" << overhead << endl;
//////////////////////////////////////////
    int endi=iterlen*2;
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_didx_ascending_swap());
    for (int i = max_idx,cnt=0; i<=endi&&cnt<vec_swap_selct.size(); i++){
      accum=update_accum(i,accum,+1);
      if(accum + vec_swap_selct[cnt].size >= mem_limit){
          cout << "cannot swap in now " << endl;
          continue;
      }
      auto itm = vec_swap_selct[cnt];
      itm.idx_in_start = i;
      if(i > itm.d_idx){
          itm.idx_in_start=itm.idx_in_end=itm.d_idx;
          cout << "report here:" << itm.d_idx  << endl;
          vec_swap_selct[cnt] = itm;
          cnt++;
          continue;
      }
      itm.t_out_end = vec_run[i].t+SwapInTime(itm.size);
      itm.t_out_start = vec_run[i].t;
      accum += itm.size;
      while(vec_run[i].t < itm.t_out_end){
          if(i+1>=itm.d_idx){
              cout << "overheadi:" << i << endl;
              overhead += (itm.t_out_end-vec_run[i].t);
              break;
          }
          else{
              i++;
              //cout << "in accum " << accum << endl;
              accum=update_accum(i,accum,1);
          }
      }
      itm.idx_in_end=i;
      vec_swap_selct[cnt] = itm;
      cnt++;
    }
    cout << "duration overhead:" << overhead << endl;
  }



  if (mode == "no-overhead"){
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int ready_idx = itm.r_idx;

      if (i > 0){
        ready_idx = std::max(ready_idx,vec_swap_selct[i-1].idx_out_end);
      }
      itm.idx_out_start = ready_idx;
      itm.t_out_start = vec_run[ready_idx].t;
      itm.t_out_end = itm.t_out_start + SwapOutTime(itm.size);
      while (itm.t_out_end > vec_run[ready_idx].t){
        ready_idx++;
      }
      itm.idx_out_end = ready_idx;
      vec_swap_selct[i] = itm;
    }
    //update idx_in_start
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_descending_swap());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int need_idx = itm.d_idx;
      //todo:csc tried syn swapin with earier deadline
      if (i > 0){ need_idx = std::min(need_idx,vec_swap_selct[i-1].idx_in_start); }
      itm.idx_in_end = need_idx;
      double prepareTime = vec_run[need_idx].t - SwapInTime(itm.size);
      while (prepareTime < vec_run[need_idx].t){
        need_idx--;
      }
      itm.idx_in_start = need_idx;
      itm.t_in_start = prepareTime;
      vec_swap_selct[i] = itm;
      UpdateLoad(vec_load_temp,itm.idx_out_end,itm.idx_in_start+1,-1,itm.size,iterlen);
    }
  }
}

void SwapGPU::BuildMetaTables(vector<SwapBlock>vec_swap_selct){

  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaError_t result1 = cudaStreamCreate(&stream1);
  cudaError_t result2 = cudaStreamCreate(&stream2);
  cout << result1 << " result1 " << endl;
  cout << result2 << " result2 " << endl;
  sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
  for (int i =0; i<vec_swap_selct.size(); i++){

    auto itm = vec_swap_selct[i];
    cout << "item r_idx:" << itm.idx_out_start << "," << itm.idx_out_end << endl;
    cout << "item d_idx:" << itm.idx_in_start << "," << itm.idx_in_end << endl;
    table_sched[0][itm.idx_out_start].push_back(itm.r_idx);
    table_sched[1][itm.idx_out_end].push_back(itm.r_idx);
    table_sched[2][itm.idx_in_start].push_back(itm.r_idx);
    table_sched[3][itm.idx_in_end].push_back(itm.r_idx);

    ///Make table_meta
    void* temp_ptr = nullptr;
    cudaMallocHost(&temp_ptr,itm.size); //pinned memory.
    BlockMeta meta;
    meta.size = itm.size;
    meta.cpu_ptr = temp_ptr;
    meta.out_stream = stream1;
    meta.in_stream = stream2;
    meta.vis=true;
    table_meta[itm.r_idx] = meta;

  }
}

void SwapGPU::UpdateMetaTables(Block* block_ptr){
  /*
  update table_meta's block_ and data_; update once atfer swap test is passed.
  enable to update negative r_idx.
  it's safe in below procedure, as r_global_index and relative_counter should never be the same.
  */

  if (past_test_flag == 1) {
    //update positive r_idx
    int r_global_index = (global_index-location_of_2nd_iteration)%iterlen;
    if (table_meta.find(r_global_index)!=table_meta.end())
        table_meta[r_global_index].block_ = block_ptr;
    else if(table_meta.find(r_global_index+iterlen)!=table_meta.end())
        table_meta[r_global_index+iterlen].block_ = block_ptr;
    else if(table_meta.find(r_global_index+iterlen*2)!=table_meta.end())
        table_meta[r_global_index+iterlen*2].block_ = block_ptr;
  }

}

void SwapGPU::Plan(){
  /*
  major stream of functions: from make candidate blocks, selection swaps, make tables, etc.
  */

  int idx_range = 0;
  vector<InfoBlock> vec_opt_info = vecBlock;
  //sort(vec_opt_info.begin(),vec_opt_info.end(),sort_by_idx_ascending());

  // scale down idx, to middle iteration.
  temp_time_baseline = vec_opt_info[location_of_5th_iteration].t;
  for (int i=0; i<vec_opt_info.size();i++){
    vec_opt_info[i].idx = vec_opt_info[i].idx - location_of_5th_iteration;
    vec_opt_info[i].t = vec_opt_info[i].t - temp_time_baseline;
  }

  // build opsSqn, and sizeSqn
  vector<InfoBlock>one_itr(&vec_opt_info[location_of_2nd_iteration+4*iterlen],&vec_opt_info[location_of_2nd_iteration+5*iterlen]);
  for (int i =0; i<one_itr.size();i++){
    operation_sequence.push_back(one_itr[i].operation_type);
    size_sequence.push_back(one_itr[i].size);
  }

  //3 iterations of vec_run and vec_load, max_idx and max_load
  vector<InfoBlock>temp_vec_run(&vec_opt_info[location_of_2nd_iteration+3*iterlen],&vec_opt_info[location_of_2nd_iteration+6*iterlen]);
  vec_run = temp_vec_run;

  vector<InfoBlock>temp_vec_run2(&vec_opt_info[location_of_2nd_iteration],&vec_opt_info[location_of_2nd_iteration+3*iterlen]);
  auto vec_run2 = temp_vec_run2;


  vector<double>vec_load(&global_load[location_of_2nd_iteration],&global_load[location_of_2nd_iteration+3*iterlen]);
  origin_load = vec_load;

  auto max_current = GetLoadPeak(vec_load,iterlen);
  max_load = max_current.first;
  max_idx = max_current.second+iterlen;

  cout << "max_idx:"<< max_idx << endl;
  //sort by ptr & idx, sorting the duplicate
  auto vec_run_dup = vec_run;
  sort(vec_run_dup.begin(),vec_run_dup.end(),sort_by_ptr_idx_ascending());

  for (auto i : vec_run_dup)if(i.idx <= max_idx+200 && i.idx >= max_idx-200)removed[i.ptr]=1;



  ///formulate swappable items.
  vector<SwapBlock>vec_swap;
  for (int i =1; i<vec_run_dup.size(); i++){
    //SwapBlock(Block* p, size_t s, int idx_out_start, int idx_in_end, double t_out_start, double t_in_end):
    //ptr(p), size(s), r_idx(idx_out_start),d_idx(idx_in_end),r_time(t_out_start), d_time(t_in_end) {}
    smallest_block=(1 << 22);
    if((vec_run_dup[i].size >= smallest_block) && (vec_run_dup[i-1].idx<max_idx-200)
      && (vec_run_dup[i].idx>max_idx+200) && (removed.find(vec_run_dup[i].ptr) == removed.end())
      && (vec_run_dup[i-1].ptr ==vec_run_dup[i].ptr)&& (vec_run_dup[i-1].operation_type==2))
      //&& ((vec_run_dup[i-1].operation_type==3) or (vec_run_dup[i-1].operation_type==2) or (vec_run_dup[i-1].operation_type==4)))
    {
        //cout << vec_run_dup[i-1].idx-iterlen << " idx " << vec_run_dup[i].idx-iterlen << endl;
        SwapBlock itm(vec_run_dup[i].ptr, vec_run_dup[i].size, vec_run_dup[i-1].idx, vec_run_dup[i].idx, vec_run_dup[i-1].t, vec_run_dup[i].t);
        vec_swap.push_back(itm);
        cout << vec_run_dup[i-1].idx << " idx " << vec_run_dup[i].idx << endl;
    }
  }
  cout << "vec_swap check:" << vec_swap.size() << endl;


  /// majority voting, can specify mode here, can specify load_limit
  auto temp_load = origin_load;

  //mem_limit_majority_voting = 550<<20;
  //576716800
  //mem_limit_majority_voting = 6000000000;
  auto vec_swap_selct = SelectBlock(vec_swap,temp_load,mem_limit_majority_voting,"majority_voting");
  // vec_swap_selct_global = vec_swap_selct;

  auto vec_load_WDOA = origin_load;
  //string mode = "stick-to-limit";
  string mode;
  if(mode_type == 0)mode = "no-overhead";
  else mode = "stick-to-limit";

  double overhead = 0;
  Scheduling(vec_swap_selct, vec_load_WDOA,overhead,mem_limit_majority_voting,mode);

  BuildMetaTables(vec_swap_selct);

}


void SwapGPU::DetectionPlan(){
  /*
    test after every index, at Append. order and index changed.
  */
  ///test iteration
  if (((global_index+1)%(iterlen_threshold) == 0) && (async_swap_flag == 0) && (past_test_flag == 0)){
    global_index_threshold = Detection(vecBlock,iterlen,location_of_2nd_iteration);
    //finished 4 iterations, iter5
    //iterlen_threshold = std::max(iterlen_threshold,global_index/10);
    //iterlen_threshold = std::min(2000,iterlen_threshold);
    if (iterlen > iterlen_threshold) {
      past_test_flag = 1;
      three_more_iteration_global_index_threshold = global_index_threshold + 3*iterlen;
      //iter8, begin index of 8th iteration
      location_of_5th_iteration = location_of_2nd_iteration + 3*iterlen;
   }
 }
 ///switch flag; next idx
 if ((global_index+1) == three_more_iteration_global_index_threshold){
     //cout << "begin to plan" << endl;
    Plan();
    async_swap_flag = 1;
 }
}


void SwapGPU::DeploySwap(){

  if (async_swap_flag == 1){
     int r_global_index = (global_index-location_of_2nd_iteration)%iterlen;
      //if(table_sched[2][r_global_index].size())cout << " swap in "<< r_global_index << endl;
      //if(table_sched[1][r_global_index].size())cout << " swap out sync idx:"<< r_global_index << endl;


    if ((global_index >= three_more_iteration_global_index_threshold && global_index < three_more_iteration_global_index_threshold + iterlen)) {
        if(((table_sched[0][r_global_index].size()) || (table_sched[1][r_global_index].size()) || (table_sched[2][r_global_index].size()) || (table_sched[3][r_global_index].size())))
            DeploySwapOut(r_global_index);
    }
    if ((global_index >= three_more_iteration_global_index_threshold + iterlen)) {
        if(((table_sched[0][r_global_index].size()) || (table_sched[1][r_global_index].size()) || (table_sched[2][r_global_index].size()) || (table_sched[3][r_global_index].size())))
            DeploySwapOut(r_global_index);
        /////////////////////////////////////////
        r_global_index += iterlen;
        if(((table_sched[0][r_global_index].size()) || (table_sched[1][r_global_index].size()) || (table_sched[2][r_global_index].size()) || (table_sched[3][r_global_index].size()))){
            if(r_global_index<max_idx)DeploySwapOut(r_global_index);
            else DeploySwapIn(r_global_index);
        }
        /////////////////////////////////////////
        r_global_index += iterlen;
        if(((table_sched[0][r_global_index].size()) || (table_sched[1][r_global_index].size()) || (table_sched[2][r_global_index].size()) || (table_sched[3][r_global_index].size())))
            DeploySwapIn(r_global_index);
    }
  }
}

void SwapGPU::SwapOut(const int idx){
  cout << "swapout" << endl;
  BlockMeta meta = table_meta[idx];
  if(syncfactor) cudaEventCreate (&meta.out_event);
  if(meta.block_->get_data() == nullptr)cout << "swapout() should not have nullptr" << endl;
  cudaError_t err = cudaMemcpyAsync(meta.cpu_ptr,meta.block_->get_data(),meta.size,cudaMemcpyDeviceToHost,meta.out_stream);
  if(syncfactor) cudaEventRecord(meta.out_event,meta.out_stream);
  table_meta[idx] = meta;
  cout << "swapout begin" << endl;
}

void SwapGPU::SwapIn(const int idx){
  cout << "swapin" << endl;
  BlockMeta meta = table_meta[idx];
  if(syncfactor)cudaEventCreate (&meta.in_event);
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);
  meta.block_->update_data(ptr);
  cudaError_t err = cudaMemcpyAsync(meta.block_->get_data(),meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream);
  if(syncfactor)cudaEventRecord(meta.in_event,meta.in_stream);
  table_meta[idx] = meta;
  cout << "swapin begin" << endl;
}
void SwapGPU::SwapInSyn(const int idx){
    cout << "sync in begin" << endl;
    auto last_meta = table_meta[idx];
    if(syncfactor)cudaEventSynchronize(last_meta.out_event);
    table_meta[idx] = last_meta;
    cout << "sync in succ " << endl;
}
void SwapGPU::SwapOutSyn(const int idx){
    cout << "sync out" << endl;
    auto last_meta = table_meta[idx];
    if(syncfactor)cudaEventSynchronize(last_meta.in_event);
    pool_->Free(last_meta.block_->get_data());
    last_meta.block_->update_data(nullptr);
    table_meta[idx] = last_meta;
    cout << "sync out succ " << endl;
}
void SwapGPU::DeploySwapOut(int r_global_index){
  for(int i = 0;i < table_sched[0][r_global_index].size();i++)
        SwapOut(table_sched[0][r_global_index][i]);
  for(int i = 0;i < table_sched[1][r_global_index].size();i++)
        SwapOutSyn(table_sched[1][r_global_index][i]);
}
void SwapGPU::DeploySwapIn(int r_global_index){
  for(int i = 0;i < table_sched[2][r_global_index].size();i++)
        SwapIn(table_sched[2][r_global_index][i]);
  for(int i = 0;i < table_sched[3][r_global_index].size();i++)
        SwapInSyn(table_sched[3][r_global_index][i]);
}

void SwapGPU::Append(InfoBlock b){
    if (iterlen < iterlen_threshold){
    if (b.operation_type == 1){
      if (global_load.size()>0){
        global_load.push_back(global_load[global_load.size()-1]+b.size);
      } else {
        global_load.push_back(b.size);
      }
    } else if (b.operation_type == -1){
      global_load.push_back(global_load[global_load.size()-1]-b.size);
    } else {
      global_load.push_back(global_load[global_load.size()-1]);
    }
  }

  //append into vec_block
  //vec_block.push_back(block_info);
  b.idx=global_index;
  vecBlock.push_back(b);

  //change swap flag on and off
  if (async_swap_flag == 1){
    int r_global_index = (global_index-location_of_2nd_iteration)%iterlen;
    if (size_sequence.size() > r_global_index && b.size != size_sequence[r_global_index]){
      async_swap_flag = 0;
      cout<<"!!!! async_swap_flag changed back to 0"<<endl;
    }
    else if(size_sequence.size() < r_global_index)
      cout << "size_sequence.size" << size_sequence.size() << endl;
  }
  UpdateMetaTables(b.ptr);
  DeploySwap();
  DetectionPlan();
  global_index++;
}

SwapGPU::~SwapGPU() {
    std::ofstream outfile;
    outfile.open("/mount/incubator-singa/examples/cifar10/vggswap");
    for(int i = 0;i < vecBlock.size();i++)
        outfile << vecBlock[i].operation_type << "," <<vecBlock[i].ptr << "," << vecBlock[i].size<<"," << (long long)vecBlock[i].t<<"\n";
    outfile.close();
    int iter2=location_of_2nd_iteration;
    cout << "iteration time duration"<<endl;
    cout << "vecblock size:" << vecBlock.size() << endl;
    for(int i = 0;i < 20;i++){
        if(iter2+iterlen*(i+1) > vecBlock.size())break;
        cout << (long long)(vecBlock[iter2+iterlen*(i+1)].t-vecBlock[iter2+iterlen*i].t) << endl;
    }
  if (ctx_.cublas_handle) CUBLAS_CHECK(cublasDestroy(ctx_.cublas_handle));
  if (ctx_.curand_generator)
    CURAND_CHECK(curandDestroyGenerator(ctx_.curand_generator));
#ifdef USE_CUDNN
  if (ctx_.cudnn_handle) {
    auto status = cudnnDestroy(ctx_.cudnn_handle);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  }
#endif
}
const int kNumCudaStream = 1;

SwapGPU::SwapGPU(int id) : Device(id, kNumCudaStream) {
  MemPoolConf conf;
  conf.add_device(id);
  pool_ = std::make_shared<CnMemPool>(conf);
  Setup();


}

SwapGPU::SwapGPU(int id, std::shared_ptr<DeviceMemPool> pool)
    : Device(id, kNumCudaStream) {
  CHECK(pool != nullptr);
  pool_ = pool;
  Setup();
  /////////////////////////////

  {
    ifstream infile("/mount/incubator-singa/examples/cifar10/input.txt");
    assert(infile.is_open());
    int d;
    infile >> mem_limit_majority_voting >> mode_type >> number_of_swap_blocks >> d;
    infile >> swap_factor;
    infile >> syncfactor;
    cout << "mem_limit_majority_voting:" << mem_limit_majority_voting << endl;
    cout << "mode_type:" << mode_type << endl;
    cout << "number_of_swap_blocks:" << number_of_swap_blocks << endl;
    cout << "swap_factor:" << swap_factor << endl;
    cout << "syncfactor:" << syncfactor << endl;
    infile.close();
  }


  memset(overheadvis,false,sizeof(overheadvis));
}

void SwapGPU::Setup() {
  lang_ = kCuda;
  ctx_.stream = NULL;  // use the default sync stream
  // TODO(wangwei) create one handle for each steam?
  CUDA_CHECK(cudaSetDevice(id_));
  // use curandCreateGeneratorHost for CudaHost device
  CURAND_CHECK(
      curandCreateGenerator(&ctx_.curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed);
  // TODO(wangwei) if one generator per stream, then need diff offset per gen?
  CURAND_CHECK(curandSetGeneratorOffset(ctx_.curand_generator, 0));
  CUBLAS_CHECK(cublasCreate(&(ctx_.cublas_handle)));

#ifdef USE_CUDNN
  // TODO(wangwei) create one handle for each stream?
  auto status = cudnnCreate(&ctx_.cudnn_handle);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
#endif  // USE_CUDNN
}

void SwapGPU::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void SwapGPU::DoExec(function<void(Context*)>&& fn, int executor) { fn(&ctx_); }

void SwapGPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                         CopyDirection direction, Context* ctx) {
  cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  // TODO(wangwei) use async copy
  // cudaMemcpyAsync(dst, src, nBytes,cudaMemcpyDefault, ctx_.stream);
}

size_t SwapGPU::GetAllocatedMem() {
  if (pool_ != nullptr) {
    auto ret = pool_->GetMemUsage();
    return ret.second - ret.first;
  }
  LOG(ERROR) << "The memory pool is not set";
  return 0u;
}

/// Allocate gpu memory.
void* SwapGPU::Malloc(int size) {
  void* ptr = nullptr;
  if (size > 0) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Malloc((void**)&ptr, size);
    // TODO(wangwei) remove the memset.
    CUDA_CHECK(cudaMemset(ptr, 0, size));
  }
  return ptr;
}

/// Free gpu memory.
void SwapGPU::Free(void* ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
  }
}

}  // namespace singa
#endif  // USE_CUDA