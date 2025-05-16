// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team
#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"
#include "model/model_topology.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>
#include <fstream>
#include <iomanip>

#include <fstream>
#include <iomanip>

void ExpertDispatcher::GetCurrentState(int num_experts, int num_layers, const std::string& filepath) {
  std::ofstream out_file(filepath, std::ios::app | std::ios::out);
  // std::ofstream out_file(filepath);
  if (!out_file.is_open()) {
      std::cerr << "Failed to open state.txt for writing." << std::endl;
      return;
  }
  totle_decode += 1;
  // ---------------BEGIN-PRIORITY-INFO---------------
  // // 写入表头（仅在文件为空时写入）
  // out_file.seekp(0, std::ios::end);
  // if (out_file.tellp() == 0) {
  //     out_file << std::left
  //              << std::setw(10) << "Expert"
  //              << std::setw(10) << "Layer"
  //              << std::setw(15) << "Device"
  //              << std::setw(20) << "Incache Visit"
  //              << std::setw(20) << "base_priority"
  //              << std::setw(20) << "cache_priority"
  //              << std::endl;
  //     out_file << std::string(75, '-') << std::endl;
  // }
  // for (int i = 1; i < num_experts; ++i) {
  //   for (int j = 1; j < num_layers; ++j) {
  //       auto node = experts_[i][j]->node;
  //       if (node == nullptr) {
  //           out_file << std::left 
  //                    << std::setw(10) << i
  //                    << std::setw(10) << j
  //                    << std::setw(15) << "N/A"
  //                    << std::setw(20) << "nullptr"
  //                    << std::setw(20) << "nullptr"
  //                    << std::setw(20) << "nullptr"
  //                    << std::endl;
  //           continue;
  //       }
  //       bool is_cuda = node->device.is_cuda();
  //       out_file << std::left
  //                << std::setw(10) << i
  //                << std::setw(10) << j
  //                << std::setw(15) << (is_cuda ? "YES" : "")
  //                << std::setw(20) << node->incache_visit_count
  //                << std::setw(20) << node->base_priority
  //                << std::setw(20) << node->cache_priority
  //                << std::endl;
  //   }
  // }
  // ---------------END-PRIORITY-INFO---------------
    double hit_rate = static_cast<double>(totle_hit) / totle_dispatch * 100;
    out_file << "\n[Cache Statistics-" << totle_decode << "]\n"
            << " Dispatches ALL: "  << totle_dispatch 
            << " Cache Hits: " << totle_hit 
            << " Hit Rate: " << std::fixed << std::setprecision(2) 
            << hit_rate << "%\n";
}
ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype,
                                   int expert_type, int num_threads)
    : pending_(0),
      num_enqueued_(0),
      start_(false),
      expert_type_(expert_type),
      input_mutex_(kNumDevices),
      input_cv_(kNumDevices),
      exec_mutex_(kNumDevices),
      exec_cv_(kNumDevices),
      input_queue_(kNumDevices),
      exec_queue_(kNumDevices),
      gpu_overload_(kNumDevices, false) {
  main_thread_stop_flag_.store(false);

  for (int i = 0; i < kNumDevices; ++i) {
    cudaSetDevice(i);
    cudaStream_t fetch_stream;
    cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
    fetch_streams_.emplace_back(fetch_stream);

    cudaStream_t out_stream;
    cudaStreamCreateWithFlags(&out_stream, cudaStreamNonBlocking);
    out_streams_.emplace_back(out_stream);

    auto thread_func = std::bind(&ExpertDispatcher::GPUFetchFunc, this, i);
    threads_.emplace_back(new base::Thread(thread_func));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());

    auto cache_limit =
        kTopologyHandle->GetSparseCacheLimit(torch::Device(torch::kCUDA, i));
    cache_sizes_.push_back(cache_limit);
  }

  for (int i = 0; i < kNumDevices * num_threads; ++i) {
    cudaSetDevice(i % kNumDevices);
    cudaStream_t exec_stream;
    cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
    exec_streams_.emplace_back(exec_stream);
    // cudaDeviceSynchronize();

    auto thread_func =
        std::bind(&ExpertDispatcher::GPUExecFunc, this, i % kNumDevices);
    threads_.emplace_back(new base::Thread(thread_func));
    threads_.back()->start();
    // SetThreadAffinity(threads_.back()->tid());
  }

  at::InferenceMode infer_guard(0);

  for (int i = 0; i < num_experts; ++i) {
    experts_.emplace_back();
    for (int j = 0; j < num_layers; ++j) {
      experts_[i].emplace_back();
      experts_[i][j] = std::make_shared<ExpertNode>();
      experts_[i][j]->expert_type = expert_type;
      int expert_type = expert_type_;
      switch (expert_type) {
        case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
          experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
          break;
        case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
          experts_[i][j]->module =
              new SwitchTransformersDenseGatedActDense(dtype);
          break;
        case NLLB_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
          break;
        case FSGPT_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
          break;
        case MIXTRAL_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
          break;
        case DEEPSEEK_MOE_DENSE_ACT_DENSE:
          experts_[i][j]->module = new DeepSeekMoEDenseActDense(dtype);
          break;
        default:
          DLOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                     expert_type);
      }
      experts_[i][j]->module->eval();
      experts_[i][j]->layer_idx = j;
      experts_[i][j]->expert_idx = i;
    }
  }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id,
                                     bool remote) {
  ExpertDispatcher::CallArgs args;
  args.layer_idx = layer_idx;
  args.expert_idx = expert_idx;
  args.gpu_id = gpu_id;
  args.remote = remote;
  Enqueue(args);
}

// void ExpertDispatcher::Enqueue(CallArgs& args) {
//   // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
//   int layer_idx = args.layer_idx;
//   int expert_idx = args.expert_idx;
//   auto expert_node = experts_[expert_idx][layer_idx];

//   if (!expert_node->node->mutex.try_lock()) {
//     // NOTE: try lock must success, if there is no prefetching
//     DLOG_FATAL("ExpertDispatcher::Enqueue: mutex try_lock failed (expert_idx ",
//                expert_idx, " layer_idx ", layer_idx, "node ",
//                expert_node->node->str(), ")");
//   }
//   expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

//   if (expert_node->node->device.is_cuda()) {
//     args.gpu_id = expert_node->node->device.index();
//   }

//   {
//     std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
//     input_queue_[args.gpu_id].push_back(std::move(args));
//   }
//   input_cv_[args.gpu_id].notify_all();
//   // input_queue_.push_back(std::move(args));
//   num_enqueued_.fetch_add(1);

//   // auto& a = input_queue_.back();
//   // if (expert_node->node->device.is_cuda()) {
//   //   a.gpu_id = expert_node->node->device.index();
//   // }
//   // DLOG_TRACE("ExpertDispatcher::Enqueue: num_enqueued_ ",
//   // num_enqueued_.load(),
//   //            "input_queue_ ", input_queue_.size(), "gpu_id ", a.gpu_id,
//   //            "layer_idx ", a.layer_idx, "expert_idx ", a.expert_idx, "remote
//   //            ", a.remote);
//   // lock.unlock();
//   // cvs_[MUTEX_TYPE::INPUT_MUTEX].notify_all();
// }
// ----------------------------------START调试锁-----------------------------------
#include <iomanip>  // for std::put_time
#include <chrono>   // for time formatting
void ExpertDispatcher::Enqueue(CallArgs& args) {
  // 辅助函数：带颜色和时间的日志输出
  auto print_trace = [](const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "\033[1;36m[TRACE]\033[0m " 
              << "\033[90m" << std::put_time(std::localtime(&now_time), "%T") << "\033[0m "
              << message << std::endl;
  };

  auto print_error = [](const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::cout << "\033[1;31m[ERROR]\033[0m " 
              << "\033[90m" << std::put_time(std::localtime(&now_time), "%T") << "\033[0m "
              << message << "\033[0m" << std::endl;
  };

  int layer_idx = args.layer_idx;
  int expert_idx = args.expert_idx;
  auto expert_node = experts_[expert_idx][layer_idx];

  // 1. 尝试加锁（带调试信息）
  std::ostringstream oss;
  // oss << "Thread \033[1;33m" << std::this_thread::get_id() << "\033[0m "
  //     << "attempting to lock node \033[1;35mLayer[" << layer_idx << "]-Expert[" << expert_idx << "]\033[0m";
  // print_trace(oss.str());
  // oss.str("");

  if (!expert_node->node->mutex.try_lock()) {
    oss << "Thread \033[1;33m" << std::this_thread::get_id() << "\033[0m "
        << "failed to lock node (\033[1;35mlayer=" << layer_idx << ", expert=" << expert_idx << "\033[0m). "
        << "Another thread may be holding the lock.";
    print_error(oss.str());
    return;
  }

  // // 2. 加锁成功（带调试信息）
  // oss << "Lock Acquired on \033[1;35mLayer[" << layer_idx << "]-Expert[" << expert_idx << "]\033[0m "
  //     << "by Thread \033[1;33m" << std::this_thread::get_id() << "\033[0m";
  // print_trace(oss.str());
  // oss.str("");

  // 3. 更新访问时间并处理队列
  expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

  if (expert_node->node->device.is_cuda()) {
    args.gpu_id = expert_node->node->device.index();
  }

  {
    std::unique_lock<std::mutex> lock(input_mutex_[args.gpu_id]);
    input_queue_[args.gpu_id].push_back(std::move(args));
  }
  input_cv_[args.gpu_id].notify_all();
  num_enqueued_.fetch_add(1);
}

// ----------------------------------END调试锁-----------------------------------

void ExpertDispatcher::RegisterExpert(
    int layer_idx, int expert_idx,
    const std::vector<std::uint32_t>& tensor_ids) {
  NodePtr cached_node = nullptr;
  for (auto tensor_id : tensor_ids) {
    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    if (cached_node == nullptr) {
      cached_node = node;
      experts_[expert_idx][layer_idx]->node = node;
    } else if (cached_node != node) {
      DLOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
    }
  }
}

void ExpertDispatcher::ClearExpertCacheCounts() {
  for (auto& expert : experts_) {
    for (auto& expert_node : expert) {
      if (expert_node->node == nullptr) {
        continue;
      }
      expert_node->node->incache_visit_count = 0;
      totle_dispatch = 0;
      totle_hit = 0;
    }
  }
}
void ExpertDispatcher::ClearExpertCachePriority() {
  for (auto& expert : experts_) {
    for (auto& expert_node : expert) {
      if (expert_node->node == nullptr) {
        continue;
      }
      expert_node->node->cache_priority = 0;
    }
  }
}
void ExpertDispatcher::SetExpertCachePriority(const std::string& priority_file) {
  auto num_layers = experts_[0].size();
  auto num_experts = experts_.size();
  // 1. 如果提供了优先级文件，优先从文件加载
  if (!priority_file.empty()) {
    std::ifstream fin(priority_file);
    if (!fin.is_open()) {
      LOG(ERROR) << "Failed to open priority file: " << priority_file;
    } else {
      std::string line;
      // 跳过表头
      std::getline(fin, line);
      std::getline(fin, line); 
      // 按行读取文件
      while (std::getline(fin, line)) {
        std::istringstream iss(line);
        int expert, layer, incache_visit;
        std::string device;
        float base_pri, cache_pri;
        if (iss >> expert >> layer >> device >> incache_visit >> base_pri >> cache_pri) {
          // 检查expert和layer是否在有效范围内
          if (expert >= 0 && expert < num_experts && 
              layer >= 0 && layer < num_layers) {
            auto node = experts_[expert][layer]->node;
            if (node != nullptr) {
              node->cache_priority = cache_pri;
            }
          }
        }
      }
      fin.close();
      return; // 文件加载模式直接返回，不执行后续动态计算
    }
  }
  int min_visit_count = INT_MAX;
  float min_priority = std::numeric_limits<float>::max();;
  u_int64_t min_time_stamp = UINT64_MAX;
  
  for (int i = 0; i < num_experts; ++i) {
    for (int j = 0; j < num_layers; ++j) {
      auto node = experts_[i][j]->node;
      if (node == nullptr){
         continue;
      }
      int incache_visit_count = node->incache_visit_count;
      u_int64_t time_stamp = node->last_access_time;
      bool is_cuda = node->device.is_cuda();

      if (incache_visit_count > max_visit_count) max_visit_count = incache_visit_count;
      if (time_stamp < min_time_stamp) min_time_stamp = time_stamp;
      if (incache_visit_count < min_visit_count) min_visit_count = incache_visit_count;
      u_int64_t current_time = MCIROSECONDS_SINCE_EPOCH;
      double time_decay = 1.0 - 0.5 * ((double)(current_time - time_stamp) / (current_time - min_time_stamp + 1));
      double normalized_visit_freq = (double)incache_visit_count / (max_visit_count + 1);
      double base_pri = normalized_visit_freq * time_decay;
      double access_factor = (incache_visit_count == 0) ? 0.0 : 1.0 / (1 + incache_visit_count);
      node->cache_priority = node->cache_priority * (1 - 0.05 * access_factor) + base_pri * (0.05 * access_factor);
    }
  }
}

// void ExpertDispatcher::GPUThreadFunc(int gpu_id) {
//   while (!main_thread_stop_flag_.load()) {
//   }
// }

void ExpertDispatcher::GPUFetchFunc(int gpu_id) {
  while (!main_thread_stop_flag_.load()) {
    // std::unique_lock<std::mutex> lock(mutexes_[MUTEX_TYPE::INPUT_MUTEX]);
    // if (cache_ == nullptr) {
    //   auto cache_limit =
    //   kDeviceMemoryPool->GetSparseCacheLimit(torch::Device(torch::kCUDA,
    //   gpu_id));
    //   // get any one expert size
    //   auto num_layers = experts_[0].size();
    //   auto num_experts = experts_.size();
    //   auto expert_node = experts_[num_layers-1][num_experts-1];

    //   int cache_capacity = cache_limit / expert_node->node->byte_size;
    //   cache_capacity_ = cache_capacity;
    // }
    std::unique_lock<std::mutex> lock(input_mutex_[gpu_id]);
    input_cv_[gpu_id].wait(lock, [&] { return !input_queue_[gpu_id].empty(); });

    CallArgs args = std::move(input_queue_[gpu_id].front());
    input_queue_[gpu_id].pop_front();

    lock.unlock();

    auto device = CUDA_DEVICE(gpu_id);
    auto original_device = (args.remote) ? CPU_DEVICE : hidden_states_.device();
    int layer_idx = args.layer_idx;
    int expert_idx = args.expert_idx;

    auto expert_node = experts_[expert_idx][layer_idx];
    bool cache_hit = expert_node->node->device.is_cuda();
    totle_hit += cache_hit ? 1 : 0;
    totle_dispatch += 1;

    // std::ofstream out_file("iter_old.txt", std::ios::app | std::ios::out);
    // if (!out_file.is_open()) {
    //   std::cerr << "Failed to open state.txt for writing." << std::endl;
    //   return;
    // }
    // double hit_rate = static_cast<double>(totle_hit) / totle_dispatch * 100;
    // out_file  << "GPUFetch"
    //           << " layer[" << layer_idx << "]expert[" << expert_idx
    //           << "] ALL: "<< totle_dispatch 
    //           << " Cache Hits: " << totle_hit 
    //           << " Hit Rate: " << std::fixed << std::setprecision(2) 
    //           << hit_rate << "%\n";

    //------------------------------------------改进代码------------------------------------------
    if (!expert_node->node->device.is_cuda() &&cache_sizes_[gpu_id] < expert_node->node->byte_size) {
      // find the expert in gpu and min incache_visit_count
      NodePtr evict_node = nullptr;
      auto num_layers = experts_[0].size();
      auto num_experts = experts_.size();
      int min_visit_count = INT_MAX;
      float min_priority = std::numeric_limits<float>::max();
      for (int i = 0; i < num_experts; ++i) {
        for (int j = 0; j < num_layers; ++j) {
          auto node = experts_[i][j]->node;
          if (node == nullptr) {
            continue;
          }
          int incache_visit_count = node->incache_visit_count;
          bool is_cuda = node->device.is_cuda();
          // 由于读取时间消耗大，单token生成只内的优先级只考虑LFU策略
          if(incache_visit_count > max_visit_count) { max_visit_count = incache_visit_count;}
          if (incache_visit_count < min_visit_count) { min_visit_count = incache_visit_count;}
          node->base_priority = (double)incache_visit_count / (max_visit_count + 1);
          //驱逐策略采用优先级算法
          if(totle_decode % 32 > 20){
            if (is_cuda && node->base_priority < min_priority && node->mutex.try_lock()) {
              evict_node = node;
              min_priority = node->base_priority;
              evict_expert = experts_[i][j];
              node->mutex.unlock();
            }
          }
          else{
            if (is_cuda && node->cache_priority < min_priority && node->mutex.try_lock()) {
              evict_node = node;
              min_priority = node->cache_priority;
              evict_expert = experts_[i][j];
              node->mutex.unlock();
            }
          }
        }
      }
      assert(evict_node != nullptr);
      evict_node->SetDevice(evict_node->default_host);
      cache_sizes_[gpu_id] += evict_node->byte_size;
    }


    //------------------------------------------原始代码------------------------------------------
    // if (!expert_node->node->device.is_cuda() &&cache_sizes_[gpu_id] < expert_node->node->byte_size) {
    //   // find the expert in gpu and min incache_visit_count
    //   NodePtr evict_node = nullptr;
    //   auto num_layers = experts_[0].size();
    //   auto num_experts = experts_.size();
    //   int min_visit_count = INT_MAX;
    //   for (int i = 0; i < num_experts; ++i) {
    //     for (int j = 0; j < num_layers; ++j) {
    //       auto node = experts_[i][j]->node;
    //       if (node == nullptr) {
    //         // std::cerr << "ExpertDispatcher::GPUFetchFunc: node is nullptr"
    //         //           << " layer_idx " << j << " expert_idx " << i <<
    //         //           std::endl;
    //         continue;
    //       }
    //       if (node->device.is_cuda() &&
    //           node->incache_visit_count < min_visit_count &&
    //           node->mutex.try_lock()) {
    //         evict_node = node;
    //         min_visit_count = node->incache_visit_count;
    //         node->mutex.unlock();
    //         // std::cerr << "ExpertDispatcher::GPUFetchFunc: evict node "
    //         //           << evict_node->device.str() << " incache_visit_count "
    //         //           << min_visit_count << std::endl;
    //       }
    //     }
    //   }
    //   assert(evict_node != nullptr);
    //   evict_node->SetDevice(evict_node->default_host);
    //   cache_sizes_[gpu_id] += evict_node->byte_size;
    // }
    //------------------------------------------结束------------------------------------------
    bool success = true;

    expert_node->node->SetDevice(CUDA_DEVICE(gpu_id), true, fetch_streams_[gpu_id]);
    expert_node->node->incache_visit_count += 1;
    expert_node->SetTensorsFromBlob(device);
    cache_sizes_[gpu_id] -= expert_node->node->byte_size;
    // std::cerr << "ExpertDispatcher::GPUFetchFunc: move to device gpu_id "
    //           << gpu_id << " layer_idx " << layer_idx << " expert_idx "
    //           << expert_idx << " node "
    //           << expert_node->node->device.str() << std::endl;

    int expert_type = expert_type_;
    torch::Tensor input;
    auto token_indices =
        router_mask_.index({"...", expert_idx}).to(torch::kBool);
    switch (expert_type) {
      case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
      case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
      case NLLB_MOE_DENSE_ACT_DENSE:
      case FSGPT_MOE_DENSE_ACT_DENSE:
      case MIXTRAL_MOE_DENSE_ACT_DENSE:
      case DEEPSEEK_MOE_DENSE_ACT_DENSE:
        input =
            hidden_states_.index({token_indices}).to(expert_node->node->device);
        break;
      default:
        DLOG_FATAL("ExpertDispatcher::expert_type: unknown expert type ",
                   expert_type);
    }

    DLOG_TRACE("ExpertDispatcher::GPUFetchFunc gpu_id ", gpu_id, "layer_idx ",
               layer_idx, "expert_idx ", expert_idx, "input ",
               input.device().str(), "node ", expert_node->node->device.str());
    {
      ExecArgs exec_args;
      exec_args.hidden_states = std::move(input);
      exec_args.expert_node = expert_node;
      exec_args.out_gpu_id = original_device.index();
      exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
      exec_args.evict = !success;
      exec_args.hit = cache_hit;
      std::lock_guard<std::mutex> lock(exec_mutex_[gpu_id]);
      exec_queue_[gpu_id].emplace_back(std::move(exec_args));
    }
    exec_cv_[gpu_id].notify_all();
  }
}

void ExpertDispatcher::GPUExecFunc(int gpu_id) {
  cudaSetDevice(gpu_id);
  while (!main_thread_stop_flag_.load()) {
    std::unique_lock<std::mutex> lock(exec_mutex_[gpu_id]);
    exec_cv_[gpu_id].wait(lock, [&] { return !exec_queue_[gpu_id].empty(); });

    ExecArgs args = std::move(exec_queue_[gpu_id].front());
    exec_queue_[gpu_id].pop_front();

    lock.unlock();

    if (args.expert_node == nullptr) {
      continue;
    }

    torch::Tensor output;

    // at::InferenceMode infer_guard(true);

    // random int [0,8)
    int rnd = std::rand() % 8;
    c10::cuda::CUDAStream stream =
        c10::cuda::getStreamFromExternal(exec_streams_[gpu_id + rnd], gpu_id);

    {
      auto start = TIME_NOW;
      // c10::cuda::CUDAStreamGuard guard(stream);

      auto* expert_module = args.expert_node->module;
      int expert_type = expert_type_;
      cudaStreamSynchronize(stream);  // make sure the input is ready

      try {
        switch (expert_type) {
          case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
            output = reinterpret_cast<SwitchTransformersDenseActDense*>(
                         expert_module)
                         ->forward(args.hidden_states);
            break;
          case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
            output = reinterpret_cast<SwitchTransformersDenseGatedActDense*>(
                         expert_module)
                         ->forward(args.hidden_states);
            break;
          case NLLB_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<NllbMoeDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          case FSGPT_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<FSGPTMoEDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          case MIXTRAL_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<MixtralMoEDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          case DEEPSEEK_MOE_DENSE_ACT_DENSE:
            output = reinterpret_cast<DeepSeekMoEDenseActDense*>(expert_module)
                         ->forward(args.hidden_states);
            break;
          default:
            DLOG_FATAL("ExpertDispatcher::GPUExecFunc: unknown expert type",
                       expert_type);
        }

      } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "DenseActDense tensor_ids: [";
        for (auto& id : args.expert_node->node->tensor_ids) {
          ss << id << " ";
        }
        ss << "]";
        DLOG_FATAL("ExpertDispatcher::GPUExecFunc", ss.str(), "expert_type",
                   expert_type, e.what());
      }

      stream.synchronize();
      auto end = TIME_NOW;
      // DLOG_INFO("ExpertDispatcher::GPUExecFunc: forward time ",
      //                  std::chrono::duration_cast<MCIROSECONDS>(end -
      //                  start).count(), "us");
    }

    (void)std::async(std::launch::async, &ExpertDispatcher::OutputFunc, this,
                     std::move(args), std::move(output), gpu_id);
  }
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output,
                                  int gpu_id) {
  // c10::cuda::CUDAStream stream =
  // c10::cuda::getStreamFromExternal(out_streams_[gpu_id], gpu_id);
  // c10::cuda::CUDAStreamGuard guard(stream);

  auto output_device =
      (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
  torch::Tensor output_tensor = output.to(output_device).to(args.out_dtype);

  if (args.evict) {
    args.expert_node->node->SetDevice(args.expert_node->node->default_host,
                                      true, nullptr);
    {
      std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
      gpu_overload_[gpu_id] = false;
    }
  }

  args.expert_node->node->mutex.unlock();

  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue_.emplace_back(std::move(output_tensor),
                               args.expert_node->layer_idx,
                               args.expert_node->expert_idx, args.hit);
    DLOG_TRACE("ExpertDispatcher::OutputFunc: output_queue_",
               output_queue_.size(), "output",
               std::get<0>(output_queue_.back()).device().str(), "evict",
               args.evict, "(", args.expert_node->layer_idx,
               args.expert_node->expert_idx, gpu_id, args.hit, ")");
  }
  // stream.synchronize();
  pending_.fetch_sub(1);
  if (pending_.load() == 0) {
    pending_cv_.notify_all();
  }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait() {
  int wait_count = 0;

  std::unique_lock<std::mutex> lock(pending_mutex_);
  pending_cv_.wait(lock, [&] { return pending_.load() == 0; });

  num_enqueued_.store(0);
  std::vector<CallResult> output_queue;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_queue.swap(output_queue_);
  }

  return output_queue;
}
