//
// Created by csc on 1/19/19.
//

#include "singa/singa_config.h"
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
#include "singa/model/layer.h"
#include "singa/core/common.h"
#include "singa/utils/cuda_utils.h"
#include <sstream>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <sstream>
using namespace std::chrono;
using namespace std;


//schedule swap and recomputation at the beginning


namespace singa {


    struct sort_by_ptr_idx_ascending{
        //sort by block ptr, in order to get the same block ptr with continuous idx
        inline bool operator() (const InfoBlock& struct1, const InfoBlock& struct2){
            return ((struct1.ptr<struct2.ptr)||((struct1.ptr==struct2.ptr)&&(struct1.idx<struct2.idx)));
        }
    };
    struct sort_by_didx_ascending_swap{
        // sort by index for schedule
        inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2){
            return (struct1.d_idx<struct2.d_idx);
        }
    };
    struct sort_by_didx_descending_swap{
        inline bool operator() (const SwapBlock& struct1, const SwapBlock& struct2){
            return (struct1.d_idx>struct2.d_idx);
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
    LL SwapGPU::SwapOutTime(size_t size){
        return (swapoutcof * size + swapoutbias);
    }

    LL SwapGPU::SwapInTime(size_t size){
        return (swapincof * size + swapinbias) ;
    }


    pair<LL,int> GetLoadPeak(vector<LL>&vec_load_test,int iterlen){
        /*
        return value and index of load peak
        */
        LL max_load_test = 0;
        int max_idx_test = 0;
        for (int i = iterlen; i < iterlen*2; i++){
            if (max_load_test < vec_load_test[i]){
                max_load_test = vec_load_test[i];
                max_idx_test = i - iterlen;
            }
        }
        return std::make_pair(max_load_test,max_idx_test);
    }



    vector<SwapBlock> SwapGPU::SelectBlock(vector<SwapBlock>&vec_swap,vector<LL> temp_load,LL mem_limit){
        // select the swap blocks
        // you should write the index of selected blocks in rfvariables and then read from there.
        vector<SwapBlock>vec_swap_selct;
        vector<int>checkfile;
        int maxn=0;
        if(rfornot){
            for(int i=0; i<(int)vec_swap.size();i++)checkfile.push_back(0);
            ifstream infile("rffile.txt");
            assert(infile.is_open());
            int numofblocks;
            infile >> numofblocks;
            int cnt=0;
            for(int i = 0;i < min(numofblocks,(int)vec_swap.size());i++){
                int cur;
                infile >>cur;
                checkfile[cur]=++cnt;
                maxn = max(maxn,cur);
            }
        }
        else for(int i=0; i<(int)vec_swap.size();i++)checkfile.push_back(1);
        sort(vec_swap.begin(),vec_swap.end(),sort_by_idx_ascending_swap());

        if(rfornot){
            for (int i=0;i<(int)vec_swap.size()&& i < number_of_swap_blocks; i++){
                for (int j=0;j<(int)vec_swap.size()&& j < number_of_swap_blocks;j++){
                    if(checkfile[j]==i+1){
                        vec_swap_selct.push_back(vec_swap[j]);
                    }
                }
            }
        }
        else{
            for (int i=0;i<(int)vec_swap.size()&& i < number_of_swap_blocks; i++){
                if(i > maxn)vec_swap_selct.push_back(vec_swap[i]);
                else if(checkfile[i])vec_swap_selct.push_back(vec_swap[i]);
            }
        }



        return vec_swap_selct;
    }

    void SwapGPU::StickToLimit(vector<SwapBlock>&vec_swap_selct, vector<LL>&vec_load_temp,LL &overhead,LL mem_limit,string mode){
        // set the memory limitation that you should not exceed.
        if (mode == "stick-to-limit"){
            overhead=0;
            sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
            int cnt=0;
            LL extra0=0,extra1=0;
            for (int i = 1;i < (int)vec_run.size()&&i<max_idx;i++){
                if(cnt>=(int)vec_swap_selct.size())continue;
                if(i < vec_swap_selct[cnt].r_idx)continue;
                auto itm = vec_swap_selct[cnt];
                itm.idx_out_start = i;
                itm.t_out_start = vec_run[i].t;
                itm.t_out_end = vec_run[i].t + SwapOutTime(itm.size);
                while(vec_run[i].t < itm.t_out_end){
                    if(accSmoothLoad[i+1]-extra0 < mem_limit && accSmoothLoad[i+2]-extra0 < mem_limit &&
                       accSmoothLoad[i+3]-extra0 < mem_limit && i+1<max_idx){
                        i++;
                    }
                    else{
                        overhead += (itm.t_out_end-vec_run[i].t);
                        break;
                    }
                }
                extra0+=itm.size;
                itm.idx_out_end=i;
                vec_swap_selct[cnt]=itm;
                cnt++;
            }
            int originalsize=vec_swap_selct.size();
            for(int i = cnt;i<originalsize;i++){
                vec_swap_selct.erase(vec_swap_selct.begin() + i);
            }
//////////////////////////////////////////
            int endi=iterlen*2;

            sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_didx_descending_swap());
            {
                int i;
                for (i = iterlen*3-2,cnt=0;cnt<(int)vec_swap_selct.size(); i--){
                    if(i > vec_swap_selct[cnt].d_idx)continue;
                    auto itm = vec_swap_selct[cnt];
                    itm.idx_in_end = i;
                    itm.t_in_end = vec_run[i].t;
                    itm.t_in_start = vec_run[i].t-SwapInTime(itm.size);
                    while(vec_run[i].t > itm.t_in_start){
                        if(accSmoothLoad[i-1]-extra1<mem_limit && accSmoothLoad[i-2]-extra1<mem_limit
                           && accSmoothLoad[i-3]-extra1<mem_limit && i-1 > max_idx){
                            i--;
                        }
                        else{
                            break;
                        }
                    }
                    extra1 += itm.size;
                    itm.idx_in_start=i;
                    vec_swap_selct[cnt] = itm;
                    cnt++;
                }
            }
        }
    }

    void SwapGPU::BuildMetaTables(vector<SwapBlock>&vec_swap_selct){
        //make schedule table, when should we launch tasks.
        cudaStream_t stream1;
        cudaStream_t stream2;
        assert(cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking)==cudaSuccess);
        assert(cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking)==cudaSuccess);
        sort(vec_swap_selct.begin(),vec_swap_selct.end(),sort_by_idx_ascending_swap());
        //make schedules for swap blocks
        for (int i =0; i<(int)vec_swap_selct.size(); i++){

            auto itm = vec_swap_selct[i];
            table_sched[0][itm.r_idx%iterlen].push_back(itm.r_idx%iterlen);
            table_sched[1][itm.idx_out_end%iterlen].push_back(itm.r_idx%iterlen);
            table_sched[2][itm.idx_in_start%iterlen].push_back(itm.r_idx%iterlen);
            table_sched[3][itm.idx_in_end%iterlen].push_back(itm.r_idx%iterlen);



            ///Make table_meta
            void* temp_ptr = nullptr;
            cudaMallocHost(&temp_ptr,itm.size); //pinned memory.
            BlockMeta meta;
            cudaEventCreate (&meta.in_event);
            cudaEventCreate (&meta.out_event);
            meta.size = itm.size;
            meta.cpu_ptr = temp_ptr;
            meta.out_stream = stream1;
            meta.in_stream = stream2;
            meta.vis=true;
            table_meta[itm.r_idx%iterlen] = meta;

        }

        //make schedule for remove directly blocks, some variables do not need to be swap
        for (int i =0; i<(int)removeBlock.size() && i < openremovedirect; i++){
            auto itm = removeBlock[i];
            table_sched[4][(itm.r_idx+6)%iterlen].push_back(itm.r_idx%iterlen);
            table_sched[5][itm.d_idx%iterlen].push_back(itm.r_idx%iterlen);
            BlockMeta meta;
            meta.vis=true;
            meta.size = itm.size;
            meta.operation_type=itm.operation_type;
            table_meta[itm.r_idx%iterlen] = meta;

        }
    }


    void SwapGPU::Plan(){
        /*
        major stream of functions: from make candidate blocks, selection swaps, make tables, etc.
        */

        vector<InfoBlock> vec_opt_info = vecBlock;
        //sort(vec_opt_info.begin(),vec_opt_info.end(),sort_by_idx_ascending());

        // scale down idx, to middle iteration.
        temp_time_baseline = vec_opt_info[fastinterval].t;
        for (int i=0; i<(int)vec_opt_info.size();i++){
            vec_opt_info[i].idx = vec_opt_info[i].idx - fastinterval;
            vec_opt_info[i].t = vec_opt_info[i].t - temp_time_baseline;
        }

        // build opsSqn, and sizeSqn
        vector<InfoBlock>one_itr(&vec_opt_info[fastinterval],&vec_opt_info[fastinterval+iterlen]);
        for (int i =0; i<(int)one_itr.size();i++){
            operation_sequence.push_back(one_itr[i].operation_type);
            size_sequence.push_back(one_itr[i].size);
        }

        //3 iterations of vec_run and vec_load, max_idx and max_load
        vector<InfoBlock>temp_vec_run(&vec_opt_info[fastinterval],&vec_opt_info[fastinterval+iterlen*3]);
        vec_run = temp_vec_run;

        vector<LL>vec_load(&accload[fastinterval],&accload[fastinterval+3*iterlen]);
        origin_load = vec_load;

        auto max_current = GetLoadPeak(vec_load,iterlen);
        max_load = max_current.first;
        max_idx = max_current.second+iterlen;
        auto vec_run_dup = vec_run;
        sort(vec_run_dup.begin(),vec_run_dup.end(),sort_by_ptr_idx_ascending());

        findSmoothL.clear();findSmoothR.clear();
        //select removalbe blocks
        //if the second usage of the blocks are free or write, I will remove the blocks directly.
        // when it will be used again, i will malloc space for it.
        vector<SwapBlock>vec_swap;
        for (int i =1; i<(int)vec_run_dup.size(); i++){
            if((vec_run_dup[i].size > (1<<remove_limit))&&(vec_run_dup[i-1].idx+delay<vec_run_dup[i].idx)&&(vec_run_dup[i-1].ptr ==vec_run_dup[i].ptr)
               && ((vec_run_dup[i-1].operation_type==READ) || (vec_run_dup[i-1].operation_type==WRITE) || (vec_run_dup[i-1].operation_type==MALLOC))
               && ((vec_run_dup[i].operation_type==FREE) || (vec_run_dup[i].operation_type==WRITE))
               && recomputeUsedBlocks.find(vec_run_dup[i-1].idx%iterlen) == recomputeUsedBlocks.end()){
                SwapBlock itm(vec_run_dup[i].ptr, vec_run_dup[i].size, vec_run_dup[i-1].idx, vec_run_dup[i].idx, vec_run_dup[i-1].t, vec_run_dup[i].t);
                itm.operation_type=vec_run_dup[i].operation_type;
                removeBlock.push_back(itm);
                findSmoothL[vec_run_dup[i-1].idx+6] = (int)removeBlock.size()-1;
                findSmoothR[vec_run_dup[i].idx] = (int)removeBlock.size()-1;
            }
        }
        int beginv = accload[fastinterval-1];
        accSmoothLoad.clear();
        for(int i = 0;i < (int)vec_run.size();i++){
            LL extra=0;
            if(openremovedirect && findSmoothL.find(i) != findSmoothL.end())extra-=(LL)removeBlock[findSmoothL[i]].size;
            if(openremovedirect && findSmoothR.find(i) != findSmoothR.end())extra+=(LL)removeBlock[findSmoothR[i]].size;
            if(vec_run[i].operation_type == MALLOC)extra += vec_run[i].size;
            else if(vec_run[i].operation_type == FREE)extra -= vec_run[i].size;
            if(i==0)accSmoothLoad.push_back(beginv+extra);
            else accSmoothLoad.push_back(accSmoothLoad[accSmoothLoad.size()-1]+extra);
        }


        max_current = GetLoadPeak(accSmoothLoad,iterlen);
        max_load = max_current.first;
        max_idx = max_current.second+iterlen;

        int midremove=iterlen/ignorefactor;
        for (auto i : vec_run_dup)if(i.idx <= max_idx+midremove && i.idx >= max_idx-midremove)removed[i.ptr]=1;
        //find condidate varibles which can be swapped
        for (int i =1; i<(int)vec_run_dup.size(); i++){
            if((vec_run_dup[i].size >= smallest_block)&&(vec_run_dup[i-1].idx+delay<vec_run_dup[i].idx)
               && (vec_run_dup[i-1].idx<max_idx-midremove)
               && (vec_run_dup[i].idx>max_idx+midremove) && (removed.find(vec_run_dup[i].ptr) == removed.end())
               && (vec_run_dup[i-1].ptr ==vec_run_dup[i].ptr) &&
               ((vec_run_dup[i-1].operation_type==READ) || (vec_run_dup[i-1].operation_type==WRITE) || (vec_run_dup[i-1].operation_type==MALLOC))
               &&(vec_run_dup[i].operation_type==READ)
               && recomputeUsedBlocks.find(vec_run_dup[i-1].idx%iterlen) == recomputeUsedBlocks.end()
                    ){
                SwapBlock itm(vec_run_dup[i].ptr, vec_run_dup[i].size, vec_run_dup[i-1].idx, vec_run_dup[i].idx, vec_run_dup[i-1].t, vec_run_dup[i].t);
                itm.operation_type=vec_run_dup[i].operation_type;
                vec_swap.push_back(itm);
            }
        }

        auto temp_load = origin_load;
        LL mem_limit = mem_limit_ratio*maxnoswapload;
        auto vec_swap_selct = SelectBlock(vec_swap,temp_load,mem_limit);

        auto vec_load_WDOA = origin_load;
        string mode="stick-to-limit";


        LL overhead = 0;
        StickToLimit(vec_swap_selct, vec_load_WDOA,overhead,mem_limit,mode);

        BuildMetaTables(vec_swap_selct);

    }


    void SwapGPU::DetectionIteration(){
        /*
          test after every index, at Append. order and index changed.
        */
        if (((global_index+1)%(iterlen_threshold) == 0) && (async_swap_flag == 0) && (past_test_flag == 0)){
            for (int i=iterlen_threshold; i<(int)vecBlock.size();i++){
                if (iterlen>iterlen_threshold)break;
                for (int len=iterlen_threshold;2*len+i<(int)vecBlock.size();len++){
                    if (iterlen>iterlen_threshold)break;
                    if(equal(vecBlock.begin()+i,vecBlock.begin()+i+len,vecBlock.begin()+i+len)) {
                        iterlen = len;
                        iter2 = i;
                    }
                }
            }

            if (iterlen<iterlen_threshold) {return;}
            else{

                global_index_threshold= global_index+iterlen-(global_index-iter2)%iterlen;
                past_test_flag = 1;
                fastiter = global_index_threshold + 6*iterlen;
                stopswap=fastiter + iterlen*5;
                //iter8, begin index of 8th iteration
                fastinterval = fastiter-4*iterlen;
            }
        }
        ///switch flag; next idx
        if ((global_index+1) == fastiter && async_swap_flag == 0){
            Plan();
            async_swap_flag = 1;
        }
    }

/////////////////////////////////////swapruntime


    const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                       cudaMemcpyDeviceToHost,
                                       cudaMemcpyDeviceToDevice};

    void SwapGPU::UpdateMetaTables(Block* block_ptr){
        //every iteration, new blocks replace old blocks
        if (async_swap_flag == 1) {
            int r_global_index = (global_index-iter2)%iterlen;
            if (table_meta[r_global_index].vis)
                table_meta[r_global_index].block_ = block_ptr;
        }
    }


    void SwapGPU::DeploySwap(){
        //deploy swap, which gpu operation index should we swap in/out
        if (async_swap_flag == 1) {
            int r_global_index = (global_index - iter2) % iterlen;
            if (global_index > fastiter + iterlen) {
                if ((table_sched[0][r_global_index].size()) ||
                    (table_sched[1][r_global_index].size()) ||
                    (table_sched[2][r_global_index].size()) ||
                    (table_sched[3][r_global_index].size()) ||
                    (table_sched[4][r_global_index].size()) ||
                    (table_sched[5][r_global_index].size()))
                {
                    DeploySwapOut(r_global_index);
                    DeploySwapIn(r_global_index);
                }
            }
        }
    }

    void SwapGPU::SwapOut(const int idx){
        //swap out a block
        BlockMeta meta = table_meta[idx];
        if(meta.out==false && meta.synout==false)meta.out=meta.synout=true;
        else return;
        LL now0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cudaError_t err = cudaMemcpyAsync(meta.cpu_ptr,meta.block_->get_data(),meta.size,cudaMemcpyDeviceToHost,meta.out_stream);
        assert(err==cudaSuccess);
        if(syncfactor) cudaEventRecord(meta.out_event,meta.out_stream);
        table_meta[idx] = meta;
        LL now1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        deploytime+=(now1-now0);
    }
    void SwapGPU::SwapOutSyn(const int idx){
        //time to sync for swap, free memory of swapped variable
        auto meta = table_meta[idx];
        if(meta.out==true && meta.synout==true)meta.synout=false;
        else return;
        LL now0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if(syncfactor)cudaEventSynchronize(meta.out_event);
        if(swapMemOp){
            pool_->Free(meta.block_->get_data());
            meta.block_->update_data(nullptr);
        }
        swapload[swapload.size()-1]-=(LL)meta.size;
        table_meta[idx] = meta;
        LL now1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        deploytime+=(now1-now0);
    }
    void SwapGPU::SwapIn(const int idx){
        //swap in a variable
        BlockMeta meta = table_meta[idx];
        if(meta.out==true && meta.synout==false && meta.synin == false)meta.synin=true;
        else return;
        LL now0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        swapload[swapload.size()-1]+=(LL)meta.size;
        assert(cudaMemcpyAsync(meta.block_->get_data(),meta.cpu_ptr,meta.size,cudaMemcpyHostToDevice,meta.in_stream)==cudaSuccess);
        if(syncfactor)cudaEventRecord(meta.in_event,meta.in_stream);
        table_meta[idx] = meta;
        LL now1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        deploytime+=(now1-now0);
    }
    void SwapGPU::SwapInSyn(const int idx){
        //swap in sync, if meet swap in sync index, pause gpu execution
        auto meta = table_meta[idx];
        if(meta.out==true && meta.synout==false && meta.synin == true)meta.synin=meta.out=false;
        else return;
        LL now0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        if(syncfactor)cudaEventSynchronize(meta.in_event);
        table_meta[idx] = meta;
        LL now1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        deploytime+=(now1-now0);
    }
    void SwapGPU::DeploySwapOut(int r_global_index){
        //check whether the current index, should we swap
        for(int i = 0;i < (int)table_sched[0][r_global_index].size();i++)
            SwapOut(table_sched[0][r_global_index][i]);
        for(int i = 0;i < (int)table_sched[1][r_global_index].size();i++)
            SwapOutSyn(table_sched[1][r_global_index][i]);
        for(int i = 0;i < (int)table_sched[4][r_global_index].size();i++){
            int idx = table_sched[4][r_global_index][i];
            auto meta = table_meta[idx];
            if(!meta.out)meta.out=true;
            else continue;
            auto ptr = meta.block_->get_data();
            pool_->Free(ptr);
            meta.block_->update_data(nullptr);
            table_meta[idx]=meta;
            swapload[swapload.size()-1]-=(LL)meta.size;
        }
    }
    void SwapGPU::DeploySwapIn(int r_global_index){
        //check whether the current index, should we swap in variables or remalloc directly
        for(int i = 0;i < (int)table_sched[2][r_global_index].size();i++)
            SwapIn(table_sched[2][r_global_index][i]);

        for(int i = 0;i < (int)table_sched[3][r_global_index].size();i++)
            SwapInSyn(table_sched[3][r_global_index][i]);
        for(int i = 0;i < (int)table_sched[5][r_global_index].size();i++){
            int idx = table_sched[5][r_global_index][i];
            BlockMeta meta = table_meta[idx];
            if(meta.out)meta.out=false;
            else continue;
            table_meta[idx] = meta;
            swapload[swapload.size()-1]+=(LL)meta.size;
        }


    }

    void SwapGPU::Append(InfoBlock b){
        //get current gpu operation block information
        if (b.operation_type == 1){
            if (accload.size()>0){
                accload.push_back(accload.back()+b.size);
                swapload.push_back(swapload.back()+b.size);
                realload.push_back(realload.back()+b.size);
            }
            else{
                accload.push_back(b.size);
                swapload.push_back(b.size);
                realload.push_back(b.size);
            }
        }else if (b.operation_type == -1){
            accload.push_back(accload.back()-b.size);
            swapload.push_back(swapload.back()-b.size);
            realload.push_back(realload.back()-b.size);
        }else{
            accload.push_back(accload.back());
            swapload.push_back(swapload.back());
            realload.push_back(realload.back());
        }

        {
            //get memory usage information
            maxnoswapload=max(maxnoswapload,realload.back());
            globalmaxpoolsize=max(globalmaxpoolsize,(LL)pool_->GetMemUsage().second);
            poolvec.push_back((LL)pool_->GetMemUsage().second);
            if(async_swap_flag == 1 && global_index>fastiter+2*iterlen && global_index<stopswap){
                maxswapload=max(maxswapload,swapload.back());
                maxpoolsize=max(maxpoolsize,(LL)pool_->GetMemUsage().second);
            }
        }
        b.idx=global_index;
        vecBlock.push_back(b);
        if(justrun)return;
        if (async_swap_flag == 1){
            int r_global_index = (global_index-iter2)%iterlen;
            if ((int)size_sequence.size() > r_global_index && b.size != size_sequence[r_global_index]){
                async_swap_flag = 0;
                cout<<"!!!!!!!!!!!!! async_swap_flag changed back to 0"<<endl;
            }
            else if((int)size_sequence.size() < r_global_index)
                cout << "size_sequence.size" << size_sequence.size() << endl;
        }
        //update block information
        UpdateMetaTables(b.ptr);
        //check whether swap in/out or not
        DeploySwap();
        //make swap schedule
        DetectionIteration();
        global_index++;
        if(async_swap_flag == 1 && global_index>fastiter+2*iterlen && (global_index-iter2)%iterlen==0 && global_index < stopswap){
            deploytime=0;
        }


        //this code is fake training, just want to get the training time
        // not save intermediate results, if them are used again, just malloc a new one.
        if(faketrain && (past_test_flag==0 ||(global_index < fastiter + faketrain*iterlen))){
            int delay=delayremove;
            int gindex = b.idx;

            if(b.operation_type==-1)blockVis.erase(b.ptr);
            else if(b.size < smallest_block || b.fake){}
            else blockVis[b.ptr]=gindex;

            if(gindex-delay>beginremove &&
               !vecBlock[gindex-delay].fake &&
               vecBlock[gindex-delay].ptr!=nullptr &&
               blockVis.find(vecBlock[gindex-delay].ptr)!=blockVis.end() &&
               blockVis[vecBlock[gindex-delay].ptr]==gindex-delay){
                Block* gptr = vecBlock[gindex-delay].ptr;
                if(gptr->get_data() != nullptr){
                    pool_->Free(gptr->get_data());
                    gptr->update_data(nullptr);
                }
            }
        }
    }

    void* SwapGPU::UpdateGpuPtr(const Block* block_ptr){
        //update gpu operation ptr
        //before peak read, after peak write
        void* ptr = nullptr;
        pool_->Malloc((void**)&ptr, block_ptr->size());
        return ptr;
    }


    SwapGPU::~SwapGPU() {
        cout << "deploytime:"<<deploytime<<endl;
        std::ofstream outfile;
        outfile.open(outputfile);
        for(int i = 0;i < (int)vecBlock.size();i++)
            outfile << vecBlock[i].operation_type << "," <<vecBlock[i].ptr << "," << vecBlock[i].size<<"," << (LL)vecBlock[i].t << "," << vecBlock[i].execCnt<< "\n";
        outfile.close();
        cout << "iterlen:"<< iterlen << endl;
        cout << "iter2:"<< iter2 << endl;
        cout << "iteration time duration"<<endl;
        cout << "vecblock size:" << vecBlock.size() << endl;
        for(int i = 0;i < 20;i++){
            if(iter2+iterlen*(i+1) > (int)vecBlock.size())break;
            if((vecBlock[iter2+iterlen*(i+1)].t-vecBlock[iter2+iterlen*i].t) == 0)break;
            cout << (LL)(vecBlock[iter2+iterlen*(i+1)].t-vecBlock[iter2+iterlen*i].t) << endl;
        }
        double t0=0,t1=1;
        if(iter2+iterlen*7 < (int)vecBlock.size())
            t0 = (LL)vecBlock[iter2+iterlen*7].t-(LL)vecBlock[iter2+iterlen*6].t;
        if(iter2+iterlen*13 < (int)vecBlock.size())
            t1 = (LL)vecBlock[iter2+iterlen*13].t-(LL)vecBlock[iter2+iterlen*12].t;
        cout << "t0,t1:" << t0 << " " << t1 << endl;
        cout << "time ratio:" << t0/t1 << endl;

        cout << "maxnoswapload:"<< maxnoswapload << endl;
        cout << "maxpoolsize:"<< maxpoolsize << endl;
        cout << "maxswapload:"<< maxswapload << endl;
        cout << "globalmaxpoolsize:"<< globalmaxpoolsize << endl;
        cout << "ratio:"<< (1.0*(double)maxswapload/(double)maxnoswapload) << endl;
        cout << "maxRNum:"<< maxRNum << endl;

        if (ctx_.cublas_handle) CUBLAS_CHECK(cublasDestroy(ctx_.cublas_handle));
        if (ctx_.curand_generator)
            CURAND_CHECK(curandDestroyGenerator(ctx_.curand_generator));
//#ifdef USE_CUDNN
        if (ctx_.cudnn_handle) {
            auto status = cudnnDestroy(ctx_.cudnn_handle);
            CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
        }
//#endif
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

        {
            int s[] = {1,(int)(1e9)};
            for (int i = 0;i < 2;i++){
                void *hostArray=(void*)0;
                cudaMallocHost(&hostArray,s[i]);
                void *deviceArray=(void*)0;
                cudaMalloc((void**)&deviceArray,s[i]);
                long long a0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                cudaMemcpy(deviceArray,hostArray,s[i],cudaMemcpyHostToDevice);
                long long b0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                cudaMemcpy(hostArray,deviceArray,s[i],cudaMemcpyDeviceToHost);
                long long c0 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                cudaFree(deviceArray);
                cudaFreeHost(hostArray);
                if(i == 0){
                    swapoutbias = b0-a0;
                    swapinbias = c0-b0;
                }
                if(i==1){
                    swapoutcof = (b0-a0-swapoutbias)*1.0/s[i];
                    swapincof = (c0-b0-swapinbias)*1.0/s[i];
                }

            }
        }

        CHECK(pool != nullptr);
        pool_ = pool;
        Setup();
        /////////////////////////////
        //read data from file for specific operation
        {
            ifstream infile("networkParameters");
            assert(infile.is_open());
            string s;
            while(getline(infile,s)){
                if(s[0]=='#' || s[0]=='{')continue;
                if(s[0]=='}')break;
                size_t pos = s.find(":");
                std::istringstream iss(s.substr (pos+1));
                if(s.compare(1,pos-2,"mem_limit_ratio")==0)iss>>mem_limit_ratio;
                else if(s.compare(1,pos-2,"number_of_swap_blocks")==0)iss>>number_of_swap_blocks;
                else if(s.compare(1,pos-2,"swap_factor")==0)iss>>swap_factor;
                else if(s.compare(1,pos-2,"syncfactor")==0)iss>>syncfactor;
                else if(s.compare(1,pos-2,"rfornot")==0)iss>>rfornot;
                else if(s.compare(1,pos-2,"swapMemOp")==0)iss>>swapMemOp;
                else if(s.compare(1,pos-2,"iterlen_threshold")==0)iss>>iterlen_threshold;
                else if(s.compare(1,pos-2,"DEBUG")==0)iss>>DEBUG;
                else if(s.compare(1,pos-2,"ignorefactor")==0)iss>>ignorefactor;
                else if(s.compare(1,pos-2,"justrun")==0)iss>>justrun;
                else if(s.compare(1,pos-2,"openremovedirect")==0)iss>>openremovedirect;
                else if(s.compare(1,pos-2,"remove_limit")==0)iss>>remove_limit;
                else if(s.compare(1,pos-2,"recompute")==0)iss>>recompute;
                else if(s.compare(1,pos-2,"recomputetype")==0)iss>>recomputetype;
                else if(s.compare(1,pos-2,"setmaxRNum")==0)iss>>setmaxRNum;
                else if(s.compare(1,pos-2,"setLastRNum")==0)iss>>setLastRNum;
                else if(s.compare(1,pos-2,"Refile")==0)iss>>Refile;
                else if(s.compare(1,pos-2,"overlapcr")==0)iss>>overlapcr;
                else if(s.compare(1,pos-2,"faketrain")==0)iss>>faketrain;
                else if(s.compare(1,pos-2,"beginremove")==0)iss>>beginremove;
                else if(s.compare(1,pos-2,"delayremove")==0)iss>>delayremove;
            }
            infile.close();
        }


        memset(table_meta,0,sizeof(table_meta));
        memset(overheadvis,false,sizeof(overheadvis));
        for(int i =0;i < MAXN;i++)for(int j =0;j < 6;j++)
                table_sched[j][i].clear();

    }

    void SwapGPU::Setup() {
        lang_ = kCuda;
        //ctx_.stream = NULL;  // use the default sync stream
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

//#ifdef USE_CUDNN
        // TODO(wangwei) create one handle for each stream?
        auto status = cudnnCreate(&ctx_.cudnn_handle);
        CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
        assert(cudaStreamCreate(&ctx_.stream)==cudaSuccess);
        assert(cudnnSetStream(ctx_.cudnn_handle,ctx_.stream)==CUDNN_STATUS_SUCCESS);

        cudaEventCreate (&ctx_.event);
        cudaEventCreate (&ctx_.event2);
        assert(cudnnCreate(&ctx_.cudnn_handle2)==CUDNN_STATUS_SUCCESS);
        assert(cudaStreamCreateWithFlags(&ctx_.stream2,cudaStreamNonBlocking)==cudaSuccess);
        assert(cudnnSetStream(ctx_.cudnn_handle2,ctx_.stream2)==CUDNN_STATUS_SUCCESS);


//#endif  // USE_CUDNN
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