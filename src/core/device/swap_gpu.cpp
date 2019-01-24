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
int SwapOutTime(size_t size){
  int ans = 0;
  //measured in 16 PCIe, pinned memory.
  if (size==0) {ans = 47200;} else {ans = 0.0756 * size + 47200;}
  return ans;
}

int SwapInTime(size_t size){
  int ans = 0;
  //measured as per ncra ~ ncrd, 16 PCIe, pinned memory.
  if (size==0) {ans = 9700;} else {ans = 0.0823 * size + 9700;}
  return ans;
}

void RepeatableTest(vector<InfoBlock>vecBlock, int &iteration_length, int &location_of_2nd_iteration, int iteration_length_threshold, int global_index ){
  /*
  repeatable test, input vector of int,
  in-place update max_legth (length of iteration)
  and location_of_2nd_iteration (where 2nd iteration starts)
  */
  int idx_range = (int)vecBlock.size();
  int threshold = iteration_length_threshold;
  vector<pair<int,int>>iteration_length_location_of_2nd_iteration;

  for (int i=0; i<idx_range;i++){
    if (iteration_length>threshold)break;
    for (int len=threshold; len<(idx_range-i);len++){
      if (iteration_length>threshold)break;
      //cout << "should not be here" << endl;
      if(equal(vecBlock.begin()+i,vecBlock.begin()+i+len,vecBlock.begin()+i+len)) {

        iteration_length = len;
        location_of_2nd_iteration = i;
      }
    }
  }
}

void UpdateLoad(vector<double>& vec_load,int start_idx, int end_idx, int plus_minus, size_t size,int iteration_length){
  /*
  update load [start_idx, end_idx) by plus_minus*size
  */
  for (int i = start_idx+iteration_length; i<end_idx+iteration_length; i++){
    vec_load[i] = vec_load[i] + static_cast<double>(size) * plus_minus;
  }
}

pair<double,int> GetLoadPeak(vector<double>vec_load_test,int iteration_length){
  /*
  return value and index of load peak
  */
  double max_load_test = 0;
  int max_idx_test = 0;
  for (int i = iteration_length; i < iteration_length*2; i++){
    if (max_load_test < vec_load_test[i]){
      max_load_test = vec_load_test[i];
      max_idx_test = i - iteration_length;
    }
  }
  return std::make_pair(max_load_test,max_idx_test);
}


vector<double> SwapGPU::GetIdealLoad(vector<double>vec_load,vector<SwapBlock> vec_swap_selct){
  /*
  get load_ideal, which is equivalent to load by synchronous swapping.
  */
  auto vec_load_return = vec_load;
  for (int i =0; i<vec_swap_selct.size(); i++){
    int auto_buffer = 0;
    auto itm = vec_swap_selct[i];
    if (itm.cat == "A2") auto_buffer = data_buffer;
    if (itm.cat == "A3") auto_buffer = mutable_data_buffer;
    UpdateLoad(vec_load_return, itm.r_idx+auto_buffer, itm.d_idx, -1, itm.size, iteration_length);
  }
  return vec_load_return;
}

pair<int,int> GetOptIdxAboveLoadLimit(vector<double>vec_load, size_t mem_limit, int start_idx, int end_idx,int iteration_length){
  /*
  get operation index (range) that above the load limit.
  input: vec_load, mem_limit, range [start_idx, end_idx)
  return range overlimit [first_over_limit, first_below_limit)
  */
  int first_over_limit = start_idx;
  int first_below_limit = end_idx;

  for (int i = start_idx+iteration_length; i < end_idx+iteration_length; i++){
    if (vec_load[i] > mem_limit){
      first_over_limit = i-iteration_length;
      break;
    }
  }

  for (int i = end_idx+iteration_length; i > first_over_limit+iteration_length; i--){
    if (vec_load[i] > mem_limit){
      first_below_limit = i-1-iteration_length;
      break;
    }
  }

  if (first_over_limit == start_idx) first_over_limit = -1;

  if (first_below_limit == end_idx) first_below_limit = -1;

  return std::make_pair(first_over_limit, first_below_limit);
}

int SwapGPU::Detection(vector<InfoBlock>vecBlock,int &iteration_length, int &location_of_2nd_iteration){
  /*
  test repeatability, detect iteration, and return global_index_threshold.
  */

  ///vec_str (vecBlock) to vec_opt_info, sort by ptr and idx.
  int idx_range = (int)(vecBlock.size());
  ///rep test
  //vector<size_t> vec_rep = DeviceOptSeqRepeatableTestPreProcess(vec_opt_info);
  //csc did not use size_delta
  RepeatableTest(vecBlock,iteration_length,location_of_2nd_iteration,iteration_length_threshold,global_index);

  //Note here location_of_2nd_iteration not exactly start of one iteration,
  //adjust to nearly start of one by restricting "Malloc"
  int shift_counter = 0;
  for (int i=0;i<iteration_length;i++){
    if (vecBlock[location_of_2nd_iteration+i].operation_type==1)shift_counter = i;
    else break;
  }
  location_of_2nd_iteration =location_of_2nd_iteration+shift_counter;

  if (iteration_length<iteration_length_threshold) {return -1;}
  else{
      cout << "iteration_length" << iteration_length << endl;
      cout << "location_of_2nd_iteration" << location_of_2nd_iteration << endl;
  }

  return global_index+iteration_length-(global_index-location_of_2nd_iteration)%iteration_length;
}

struct sort_by_DOA_origin_descending{
  inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2)
  {
    return (struct1.DOA_origin>struct2.DOA_origin);
  }
};

vector<SwapBlock> SwapGPU::SelectBlock(vector<SwapBlock>vec_swap,vector<double> temp_load,double mem_limit,string mode){

  vector<SwapBlock>vec_swap_selct;
  int cnt=0;
  sort(vec_swap.begin(),vec_swap.end(),sort_by_DOA_origin_descending());
  //select block one by one till updated peak load is no larger than limit.
  for (int i=0; i<vec_swap.size(); i++){
    UpdateLoad(temp_load,vec_swap[i].r_idx_ready,vec_swap[i].d_idx,-1,vec_swap[i].size,iteration_length);
    vec_swap_selct.push_back(vec_swap[i]);
    auto temp_over_limit_ = GetOptIdxAboveLoadLimit(temp_load,mem_limit,0,iteration_length,iteration_length);
    auto max_current = GetLoadPeak(temp_load,iteration_length);
    auto newmax_load = max_current.first;
    ///even one swap in swap out, there is a problem
    if(++cnt>number_of_swap_blocks)break;
    if (number_of_swap_blocks == 0 && newmax_load < mem_limit)break;
  }
  //swap all of the elements
  //todo:csc try to swap all of the blocks
  //return vec_swap;
  return vec_swap_selct;
}

void SwapGPU::Scheduling(vector<SwapBlock>&vec_swap_selct, vector<double>&vec_load_temp,double &overhead,double mem_limit,string mode){
  /*
  Swap Scheduling algo
  update idx_out_end, idx_in_start
  compute overhead time
  mode selection: no overhead or stick to limit.
  */

  overhead = 0;

  /// mode that stick to the mem_limit
  if (mode == "stick-to-limit"){
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int ready_idx = itm.r_idx_ready;

      if (i > 0){
        ready_idx = std::max(ready_idx,vec_swap_selct[i-1].idx_out_end);
      }

      itm.idx_out_start = ready_idx;
      itm.t_out_start = vec_run[ready_idx+iteration_length].t;
      itm.t_out_end = itm.t_out_start + SwapOutTime(itm.size);
      total_swap_out_time+=SwapOutTime(itm.size);
      while (itm.t_out_end > vec_run[ready_idx+iteration_length].t){
        //ready means when able to finish swapOut, w/ or w/o overhead.
        ready_idx++;
      }

      //get min compare with max_idx and ready_idx.
      ready_idx = std::min(max_idx,ready_idx);
      UpdateLoad(vec_load_temp,ready_idx+1,itm.d_idx,-1,itm.size,iteration_length);
      auto temp_over_limit_ = GetOptIdxAboveLoadLimit(vec_load_temp,mem_limit,0,iteration_length,iteration_length);
      if ((temp_over_limit_.first != -1) && (temp_over_limit_.first <= ready_idx)) {
        UpdateLoad(vec_load_temp,temp_over_limit_.first-1,ready_idx+1,-1,itm.size,iteration_length);
        ready_idx = temp_over_limit_.first - 1;
        overhead+=(itm.t_out_end-vec_run[ready_idx+iteration_length].t);
      }
      itm.idx_out_end = ready_idx;
      vec_swap_selct[i] = itm;
    }

    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_descending_swap());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int need_idx = itm.d_idx-6;
      //swap in advance, here is six idx
      if (i > 0){ need_idx = std::min(need_idx,vec_swap_selct[i-1].idx_in_start); }
      itm.idx_in_end = need_idx;
      double prepareTime = vec_run[need_idx+iteration_length].t - SwapInTime(itm.size);
      total_swap_in_time+=SwapInTime(itm.size);
      while (prepareTime < vec_run[need_idx+iteration_length].t){
        need_idx--;
      }
      need_idx = std::max(need_idx,max_idx+1);
      itm.idx_in_start = need_idx;
      itm.t_in_start = prepareTime;
      UpdateLoad(vec_load_temp,itm.idx_in_start,itm.d_idx,1,itm.size,iteration_length);
      auto temp_over_limit_3 = GetOptIdxAboveLoadLimit(vec_load_temp,mem_limit,0,iteration_length,iteration_length);

      if ((temp_over_limit_3.second != -1) && (vec_run[temp_over_limit_3.second+iteration_length].t > itm.t_in_start)) {
        overhead+=(vec_run[temp_over_limit_3.second+iteration_length].t - itm.t_in_start);
        UpdateLoad(vec_load_temp,itm.idx_in_start,temp_over_limit_3.second+1,-1,itm.size,iteration_length);
        itm.idx_in_start = temp_over_limit_3.second+1;
        auto temp_over_limit_4 = GetOptIdxAboveLoadLimit(vec_load_temp,mem_limit,0,iteration_length,iteration_length);
      }
      vec_swap_selct[i] = itm;
    }
  }///end of first mode.


  ///mode that incurs zero overhead
  if (mode == "no-overhead"){
    //update idx_out_end
    //sort by r_idx for idx_out_end update
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
    for (int i = 0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int ready_idx = itm.r_idx_ready;

      if (i > 0){
        ready_idx = std::max(ready_idx,vec_swap_selct[i-1].idx_out_end);
      }
      itm.idx_out_start = ready_idx;
      itm.t_out_start = vec_run[ready_idx+iteration_length].t;
      itm.t_out_end = itm.t_out_start + SwapOutTime(itm.size);
      while (itm.t_out_end > vec_run[ready_idx+iteration_length].t){
        ready_idx++;
      }
      itm.idx_out_end = ready_idx;
      vec_swap_selct[i] = itm;
    }
    //update idx_in_start
    sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_descending_swap());
    for (int i =0; i<vec_swap_selct.size(); i++){
      auto itm = vec_swap_selct[i];
      int need_idx = itm.d_idx-6;
      //todo:csc tried syn swapin with earier deadline
      if (i > 0){ need_idx = std::min(need_idx,vec_swap_selct[i-1].idx_in_start); }
      itm.idx_in_end = need_idx;
      double prepareTime = vec_run[need_idx+iteration_length].t - SwapInTime(itm.size);
      while (prepareTime < vec_run[need_idx+iteration_length].t){
        need_idx--;
      }
      itm.idx_in_start = need_idx;
      itm.t_in_start = prepareTime;
      vec_swap_selct[i] = itm;
      UpdateLoad(vec_load_temp,itm.idx_out_end,itm.idx_in_start+1,-1,itm.size,iteration_length);
    }
  }
}

void SwapGPU::BuildMetaTables(vector<SwapBlock>vec_swap_selct){
  /*
  construct tables: table_sched, and table_meta
  */
  cudaStream_t stream1;
  cudaStream_t stream2;
  sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
  //for each swap select, make table_sched and table_meta
  // for (int i = static_cast<int>(vec_swap_selct.size()-1);i>=0; i--){
  for (int i =0; i<vec_swap_selct.size(); i++){
    auto itm = vec_swap_selct[i];
    cout << "item r_idx:" << itm.idx_out_start << "," << itm.idx_out_end << endl;
    cout << "item d_idx:" << itm.idx_in_start << "," << itm.idx_in_end << endl;
    if (table_sched.find(itm.idx_out_start) == table_sched.end()){
      table_sched[itm.idx_out_start] = std::make_tuple(itm.r_idx,0,-1,-1);
    } else {
      std::get<0>(table_sched.find(itm.idx_out_start)->second) = itm.r_idx;
      std::get<1>(table_sched.find(itm.idx_out_start)->second) = 0;
    }
    //idx_in_start swap
    if (table_sched.find(itm.idx_in_start) == table_sched.end()){
      table_sched[itm.idx_in_start] = std::make_tuple(itm.r_idx,1,-1,-1);
    } else {
      std::get<0>(table_sched.find(itm.idx_in_start)->second) = itm.r_idx;
      std::get<1>(table_sched.find(itm.idx_in_start)->second) = 1;
    }
    // idx_out_end sync
    if (table_sched.find(itm.idx_out_end) == table_sched.end()){
      table_sched[itm.idx_out_end] = std::make_tuple(-1,-1,itm.r_idx,0);
    } else {
      std::get<2>(table_sched.find(itm.idx_out_end)->second) = itm.r_idx;
      std::get<3>(table_sched.find(itm.idx_out_end)->second) = 0;
    }
    //i2 sync
    if (table_sched.find(itm.idx_in_end) == table_sched.end()){
      table_sched[itm.idx_in_end] = std::make_tuple(-1,-1,itm.r_idx,1);
    } else {
      std::get<2>(table_sched.find(itm.idx_in_end)->second) = itm.r_idx;
      std::get<3>(table_sched.find(itm.idx_in_end)->second) = 1;
    }

    ///Make table_meta
    void* temp_ptr = nullptr;
    cudaMallocHost(&temp_ptr,itm.size); //pinned memory.
    BlockMeta meta;
    meta.size = itm.size;
    meta.cpu_ptr = temp_ptr;
    meta.out_stream = stream1;
    meta.in_stream = stream2;
    meta.block_=vec_run[itm.r_idx+iteration_length].ptr;
    //todo:csc add block ptr here
    table_meta[itm.r_idx] = meta;

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
    vec_opt_info[i].idx = vec_opt_info[i].idx - location_of_5th_iteration - iteration_length;
    vec_opt_info[i].t = vec_opt_info[i].t - temp_time_baseline;
  }

  // build opsSqn, and sizeSqn
  vector<InfoBlock>one_itr(&vec_opt_info[location_of_2nd_iteration+4*iteration_length],&vec_opt_info[location_of_2nd_iteration+5*iteration_length]);
  for (int i =0; i<one_itr.size();i++){
    operation_sequence.push_back(one_itr[i].operation_type);
    size_sequence.push_back(one_itr[i].size);
  }

  //3 iterations of vec_run and vec_load, max_idx and max_load
  vector<InfoBlock>temp_vec_run(&vec_opt_info[location_of_2nd_iteration+3*iteration_length],&vec_opt_info[location_of_2nd_iteration+6*iteration_length]);
  vec_run = temp_vec_run;

  vector<InfoBlock>temp_vec_run2(&vec_opt_info[location_of_2nd_iteration],&vec_opt_info[location_of_2nd_iteration+3*iteration_length]);
  auto vec_run2 = temp_vec_run2;


  vector<double>vec_load(&global_load[location_of_2nd_iteration],&global_load[location_of_2nd_iteration+3*iteration_length]);
  origin_load = vec_load;

  auto max_current = GetLoadPeak(vec_load,iteration_length);
  max_load = max_current.first;
  max_idx = max_current.second;

  //sort by ptr & idx, sorting the duplicate
  auto vec_run_dup = vec_run;
  sort(vec_run_dup.begin(),vec_run_dup.end(),sort_by_ptr_idx_ascending());

  ///formulate swappable items.
  vector<SwapBlock>vec_swap;
  for (int i =1; i<vec_run_dup.size(); i++){
    //SwapBlock(Block* p, size_t s, int idx_out_start, int idx_in_end, double t_out_start, double t_in_end):
    //ptr(p), size(s), r_idx(idx_out_start),d_idx(idx_in_end),r_time(t_out_start), d_time(t_in_end) {}
    if((vec_run_dup[i].size >= smallest_block) && (vec_run_dup[i-1].idx<max_idx) && (vec_run_dup[i].idx>max_idx)
      && (vec_run_dup[i-1].ptr ==vec_run_dup[i].ptr)
      && ((vec_run_dup[i-1].operation_type==3) or (vec_run_dup[i-1].operation_type==2) or (vec_run_dup[i-1].operation_type==4)))
    {
        //todo :csc debug, remove cross blocks
        if(vec_run_dup[i].idx >=iteration_length || vec_run_dup[i-1].idx < 0)break;
        SwapBlock itm(vec_run_dup[i].ptr, vec_run_dup[i].size, vec_run_dup[i-1].idx, vec_run_dup[i].idx, vec_run_dup[i-1].t, vec_run_dup[i].t);
      itm.DOA_origin = itm.d_time-itm.r_time;
      itm.DOA = itm.d_time-itm.r_time-SwapOutTime(itm.size)-SwapOutTime(itm.size);
      if (itm.DOA>=0){
        itm.AOA = itm.DOA * itm.size;
      } else {
        itm.AOA = itm.DOA * 1/itm.size;
      }
      //cat A
      if (vec_run_dup[i-1].operation_type == 3){ itm.cat = "A1"; itm.r_idx_ready = itm.r_idx + 6;}
      if (vec_run_dup[i-1].operation_type == 2){ itm.cat = "A2"; itm.r_idx_ready = itm.r_idx + 6;}//data_buffer;}
      if (vec_run_dup[i-1].operation_type == 4){ itm.cat = "A3"; itm.r_idx_ready = itm.r_idx + 6;}//mutable_data_buffer;}

      vec_swap.push_back(itm);
    }
  }

  ///////////////////////////////////////////////////
  ////swapable block
  //////////////////////////////////////////////////
  {
    ///load ideal, swap all vec_swap, lest possible memory by one-swap, for data collection only.
//    auto vec_load_ideal = GetIdealLoad(vec_load,vec_swap);
//    fstream file_load_ideal("load_ideal.csv", ios::in|ios::out|ios::app);
//    for (int i=iteration_length; i<iteration_length*2; i++){
//        file_load_ideal<<vec_load_ideal[i]<<endl;
//    }
//    auto max_ideal = GetLoadPeak(vec_load_ideal,iteration_length);
//    size_t max_load_ideal = max_ideal.first;
//    int max_idx_ideal = max_ideal.second;
  }


  /// majority voting, can specify mode here, can specify load_limit
  auto temp_load = origin_load;


  std::ifstream infile("/mount/incubator-singa/examples/cifar10/input.txt");
  std::string line;
  while (std::getline(infile, line))
  {
    std::istringstream iss(line);
    int a, b, c,d;
    if (!(iss >> a >> b >> c >> d)) { break; } // error
    mem_limit_majority_voting = a;
    mode_type = b;
    number_of_swap_blocks=c;
  }



  //mem_limit_majority_voting = 550<<20;
  //mem_limit_majority_voting = 6000000000;
  auto vec_swap_majority_voting = SelectBlock(vec_swap,temp_load,mem_limit_majority_voting,"majority_voting");
  // vec_swap_selct_global = vec_swap_majority_voting;

  auto vec_load_WDOA = origin_load;
  //string mode = "stick-to-limit";
  string mode;
  if(mode_type == 0)
    mode = "no-overhead";
  else
    mode = "stick-to-limit";

  double overhead_WDOA = 0;
  Scheduling(vec_swap_majority_voting, vec_load_WDOA,overhead_WDOA,mem_limit_majority_voting,mode);

  BuildMetaTables(vec_swap_majority_voting);

}


void SwapGPU::DetectionPlan(){
  /*
    test after every index, at Append. order and index changed.
  */
  ///test iteration
  if (((global_index+1)%(iteration_length_threshold) == 0) && (async_swap_flag == 0) && (past_test_flag == 0)){
    global_index_threshold = Detection(vecBlock,iteration_length,location_of_2nd_iteration);
    //finished 4 iterations, iter5
    //iteration_length_threshold = std::max(iteration_length_threshold,global_index/10);
    //iteration_length_threshold = std::min(2000,iteration_length_threshold);
    if (iteration_length > iteration_length_threshold) {
      past_test_flag = 1;
      three_more_iteration_global_index_threshold = global_index_threshold + 3*iteration_length;
      //iter8, begin index of 8th iteration
      location_of_5th_iteration = location_of_2nd_iteration + 3*iteration_length;
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

  //swap and sync as per schedule, at every index, by calling DeploySwapExec()


  int r_global_index = (global_index-location_of_2nd_iteration)%iteration_length;
  int r_global_index_n = r_global_index - iteration_length;

  if (async_swap_flag == 1){
    if ((global_index < three_more_iteration_global_index_threshold + iteration_length) && (!(table_sched.find(r_global_index_n) == table_sched.end()))) {
      DeploySwapExec(r_global_index_n);
    }
    if ((global_index >= three_more_iteration_global_index_threshold + iteration_length) && (!(table_sched.find(r_global_index_n) == table_sched.end()))) {
      DeploySwapExec(r_global_index_n);
    }
    if ((global_index >= three_more_iteration_global_index_threshold + iteration_length) && (!(table_sched.find(r_global_index) == table_sched.end()))) {
      DeploySwapExec(r_global_index);
    }
  }
}


void SwapGPU::DeploySwapExec(int r_global_index){
  //execute DeploySwap
  auto swap_idx = std::get<0>(table_sched.find(r_global_index)->second);
  auto swap_dir = std::get<1>(table_sched.find(r_global_index)->second);
  auto sync_idx = std::get<2>(table_sched.find(r_global_index)->second);
  auto sync_dir = std::get<3>(table_sched.find(r_global_index)->second);
  if (swap_dir == 0){
      cout << "swapout" << endl;
    SwapOut(swap_idx);
    cout << "swapout begin" << endl;
  }
  if (swap_dir == 1){
    cout << "swapin" << endl;
      SwapIn(swap_idx);
      cout << "swapin begin" << endl;
  }
  if (sync_dir == 0){
      cout << "sync out" << endl;
    ///sync swap-out, including sync, update block's data_ to nullptr, free data_, update meta.
    auto last_meta = table_meta.find(sync_idx)->second;
    cudaEventSynchronize(last_meta.in_event);

    table_not_at_device[last_meta.block_] = sync_idx;

    //last_meta.block_->update_data(nullptr);
    //pool_->Free(last_meta.data_);
    pool_->Free(last_meta.block_->get_data());
    last_meta.block_->update_data(nullptr);
    //last_meta.data_ = nullptr;
    //todo:csc debug


    table_meta.find(sync_idx)->second = last_meta;
    cout << "sync out succ " << endl;
  }
  if (sync_dir == 1){
    cout << "sync in" << endl;
      ///sync swap-in, including sync, update block's data_ to new gpu address, update meta.
    auto last_meta = table_meta.find(sync_idx)->second;
    cudaEventSynchronize(last_meta.out_event);
    table_not_at_device.erase(last_meta.block_);
    last_meta.block_->update_data(last_meta.data_);

    table_meta.find(sync_idx)->second = last_meta;
    cout << "sync in succ " << endl;
  }
}

void SwapGPU::Append(InfoBlock b){
    //cout << global_index << ":global index" << endl;
    if (iteration_length < iteration_length_threshold){
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
    int r_global_index = (global_index-location_of_2nd_iteration)%iteration_length;
    if (size_sequence.size() > r_global_index && b.size != size_sequence[r_global_index]){
      async_swap_flag = 0;
      cout<<"!!!! async_swap_flag changed back to 0"<<endl;
    }
    else if(size_sequence.size() < r_global_index)
        cout << "size_sequence.size" << size_sequence.size() << endl;
  }
  DeploySwap();
  DetectionPlan();
  global_index++;
}


void SwapGPU::SwapOut(const int idx){

  //memory copy asynchronously GPU -> CPU, and update meta.
  cudaError_t err;
  BlockMeta meta = table_meta.find(idx)->second;
  cudaEventCreate (&meta.out_event);
  //err = cudaMemcpyAsync(meta.cpu_ptr,meta.data_,meta.size,cudaMemcpyDeviceToHost,meta.out_stream);
  err = cudaMemcpyAsync(meta.cpu_ptr,meta.block_->get_data(),meta.size,cudaMemcpyDeviceToHost,meta.out_stream);
  //todo:csc debug
  cudaEventRecord(meta.out_event,meta.out_stream);
  table_meta.find(idx)->second = meta;
}

void SwapGPU::SwapIn(const int idx){

  //memory copy asynchronously CPU -> GPU, and update meta.
  cudaError_t err;
  BlockMeta meta = table_meta.find(idx)->second;
  cudaEventCreate (&meta.in_event);
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);
  meta.data_ = ptr;
  err = cudaMemcpyAsync(meta.data_,meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream);
  cudaEventRecord(meta.in_event,meta.in_stream);
  table_meta.find(idx)->second = meta;
}


SwapGPU::~SwapGPU() {
    std::ofstream outfile;
    outfile.open("/mount/incubator-singa/examples/cifar10/vggtxt");
    for(int i = 0;i < vecBlock.size();i++)
        outfile << vecBlock[i].operation_type << "," <<vecBlock[i].ptr << "," << vecBlock[i].size<<"," << vecBlock[i].t<<"\n";
    outfile.close();

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