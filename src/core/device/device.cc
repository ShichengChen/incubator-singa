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

#include "singa/core/device.h"

namespace singa {
Device::Device(int id, int num_executors)
    : id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
}

void Device::Exec(function<void(Context*)>&& fn, const vector<Block*> read_blocks,
                  const vector<Block*> write_blocks, bool use_rand_generator){
    // TODO(wangwei) execute operations scheduled by the scheduler.
    for(auto it = read_blocks.begin(); it != read_blocks.end() && (*it) != 0; it++){
        //read blocks
        long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        Append(InfoBlock(*it,(int)(*it)->size(),2,-1,now,execnt,0));
    }
    DoExec(std::move(fn), 0);
    for(auto it = write_blocks.begin(); it != write_blocks.end() && (*it) != 0; it++){
        //write blocks
        long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        Append(InfoBlock(*it,(int)(*it)->size(),4,-1,now,execnt,0));
    }

    execnt++;
}

// TODO(wangwei) get Block from the memory manager
    Block* Device::NewBlock(int size) {
        CHECK_GE(size, 0) << "size is negative, could be caused by the type cast "
                          << "from size_t to int. In that case, the size is too large.";
        if (size > 0) {
            void* ptr = Malloc(size);
            auto newblock = new Block(ptr, (size_t)size,0,this);
            long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            Append(InfoBlock(newblock,size,1,-1,now,-1));
            return newblock;
        } else {
            return nullptr;
        }
    }
    void Device::AppendInfo(Block* block,int type){
        LL now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        Append(InfoBlock(block,(int)block->size(),type,-1,now,-1));
    }

// TODO(wangwei) return Block to the memory manager
    void Device::FreeBlock(Block* block) {
        if (block != nullptr) {
            long long now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if(block->data_ptr()== nullptr){
                Append(InfoBlock(block,(int)block->size(),-1,-1,now,-1));
                delete block;
            }
            else{
                auto cptr = block->mutable_data();
                Append(InfoBlock(block,(int)block->size(),-1,-1,now,-1));
                Free(cptr);
                delete block;
            }

        }
    }
void Device::CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                            CopyDirection direct, int dst_offset,
                            int src_offset) {
  this->Exec(
      [this, dst, src, nBytes, direct, dst_offset, src_offset](Context* ctx) {
        this->CopyToFrom(
            reinterpret_cast<char*>(dst->mutable_data()) + dst_offset,
            reinterpret_cast<const char*>(src->data()) + src_offset, nBytes,
            direct, ctx);
      },
      {src}, {dst});
}

void Device::CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                                 size_t dst_offset) {
  auto direct = lang_ == kCpp ? kHostToHost : kHostToDevice;
  void* dstptr = reinterpret_cast<char*>(dst->mutable_data()) + dst_offset;
  Exec([this, dstptr, src, nBytes,
        direct](Context* ctx) { CopyToFrom(dstptr, src, nBytes, direct, ctx); },
       {}, {dst});
}
void Device::Sync() {}
}  // namespace singa
