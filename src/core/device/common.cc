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
//#ifndef DISABLE_WARNINGS

#include "singa/core/common.h"
#include "singa/core/device.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;


namespace singa {


    void* Block::get_data(){
        if(data_== nullptr){

            data_ = ptr_device_->UpdateGpuPtr(this);
        }
        return static_cast<char*>(data_);
    }
    void* Block::mutable_data() {
        initialized_ = true;
        if(data_== nullptr){
            data_ = ptr_device_->UpdateGpuPtr(this);
        }
        return static_cast<char*>(data_) + offset_;
    }
    const void* Block::data() const {
        CHECK(initialized_) << "Must initialize data before reading it";
        if(data_== nullptr){
            void *ptr = const_cast<void*>(ptr_device_->UpdateGpuPtr(this));
            data_=ptr;
        }
        return static_cast<char*>(data_) + offset_;
    }

}  // namespace singa

//#endif