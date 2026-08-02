#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#include <linux/mempolicy.h>

namespace {

constexpr size_t kGiB = 1ULL << 30;
constexpr size_t kMiB = 1ULL << 20;
constexpr size_t kAllocationBytes = 3ULL * kGiB + 384ULL * kMiB;  // 3.375 GiB
constexpr size_t kPlaneBytes = 6ULL * kMiB + 768ULL * 1024;       // 6.75 MiB
constexpr size_t kPlaneCount = 512;
constexpr size_t kWords = kAllocationBytes / sizeof(uint64_t);
constexpr size_t kPlaneWords = kPlaneBytes / sizeof(uint64_t);
constexpr uint64_t kMul = 0x9e3779b185ebca87ULL;
constexpr uint64_t kAdd = 0xd1b54a32d192ed03ULL;
constexpr int kThreads = 256;
constexpr int kFullBlocks = 2048;
constexpr int kPlaneBlocks = 256;
constexpr uint64_t kNoError = std::numeric_limits<uint64_t>::max();

struct BlockResult {
  uint64_t sum;
  uint64_t errors;
  uint64_t first_error;
};

struct Stats {
  double mean = 0;
  double stddev = 0;
  double min = 0;
  double median = 0;
  double p95 = 0;
  double max = 0;
};

__host__ __device__ inline uint64_t expected_word(uint64_t i) {
  return i * kMul + kAdd;
}

uint64_t expected_sum(uint64_t start, uint64_t count) {
  unsigned __int128 n = count;
  unsigned __int128 first = start;
  unsigned __int128 triangular = n * (2 * first + n - 1) / 2;
  return static_cast<uint64_t>(triangular * kMul + n * kAdd);
}

template <bool Validate>
__global__ void read_kernel(const volatile uint64_t* data, uint64_t start,
                            uint64_t count, BlockResult* output,
                            uint64_t output_offset) {
  __shared__ uint64_t sums[kThreads];
  __shared__ uint64_t errors[kThreads];
  __shared__ uint64_t first_errors[kThreads];
  uint64_t local_sum = 0;
  uint64_t local_errors = 0;
  uint64_t local_first = kNoError;
  uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t rel = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       rel < count; rel += stride) {
    uint64_t index = start + rel;
    uint64_t value = data[index];
    local_sum += value;
    if constexpr (Validate) {
      if (value != expected_word(index)) {
        ++local_errors;
        local_first = min(local_first, index);
      }
    }
  }
  sums[threadIdx.x] = local_sum;
  errors[threadIdx.x] = local_errors;
  first_errors[threadIdx.x] = local_first;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sums[threadIdx.x] += sums[threadIdx.x + offset];
      errors[threadIdx.x] += errors[threadIdx.x + offset];
      first_errors[threadIdx.x] = min(first_errors[threadIdx.x],
                                      first_errors[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[output_offset + blockIdx.x] =
        BlockResult{sums[0], errors[0], first_errors[0]};
  }
}

__global__ void boundary_read_kernel(const volatile uint64_t* data,
                                     const uint64_t* indices,
                                     uint64_t* values, int count) {
  int i = threadIdx.x;
  if (i < count) values[i] = data[indices[i]];
}

__global__ void boundary_write_kernel(volatile uint64_t* data,
                                      const uint64_t* indices,
                                      const uint64_t* values, int count) {
  int i = threadIdx.x;
  if (i < count) data[indices[i]] = values[i];
}

void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA_ERROR " << what << ": "
              << cudaGetErrorName(status) << " - "
              << cudaGetErrorString(status) << std::endl;
    std::exit(2);
  }
}

Stats stats(std::vector<double> values) {
  Stats s;
  if (values.empty()) return s;
  s.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double sq = 0;
  for (double v : values) sq += (v - s.mean) * (v - s.mean);
  s.stddev = std::sqrt(sq / values.size());
  std::sort(values.begin(), values.end());
  s.min = values.front();
  s.max = values.back();
  s.median = values[values.size() / 2];
  s.p95 = values[static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1];
  return s;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20)
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec;
        else
          out << c;
    }
  }
  return out.str();
}

std::string proc_value(const std::string& path, const std::string& key) {
  std::istringstream in(read_file(path));
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind(key, 0) == 0) return line;
  }
  return key + " unavailable";
}

std::string numa_map_for(const void* ptr) {
  uintptr_t target = reinterpret_cast<uintptr_t>(ptr);
  std::istringstream in(read_file("/proc/self/numa_maps"));
  std::string line, best;
  uintptr_t best_start = 0;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string address;
    ls >> address;
    if (address.empty()) continue;
    uintptr_t start = 0;
    try { start = std::stoull(address, nullptr, 16); } catch (...) { continue; }
    if (start <= target && start >= best_start) {
      best_start = start;
      best = line;
    }
  }
  return best;
}

std::vector<uint64_t> query_page_nodes(void* ptr, size_t bytes, int max_node,
                                       int* syscall_errno) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t pages = (bytes + page - 1) / page;
  std::vector<uint64_t> counts(max_node + 2, 0);
  const size_t chunk = 65536;
  std::vector<void*> addresses(chunk);
  std::vector<int> status(chunk);
  *syscall_errno = 0;
  for (size_t base = 0; base < pages; base += chunk) {
    size_t n = std::min(chunk, pages - base);
    for (size_t i = 0; i < n; ++i)
      addresses[i] = static_cast<char*>(ptr) + (base + i) * page;
    long rc = syscall(SYS_move_pages, 0, n, addresses.data(), nullptr,
                      status.data(), 0);
    if (rc < 0) {
      *syscall_errno = errno;
      counts.clear();
      return counts;
    }
    for (size_t i = 0; i < n; ++i) {
      if (status[i] >= 0 && status[i] <= max_node)
        ++counts[status[i]];
      else
        ++counts.back();
    }
  }
  return counts;
}

void set_locality(int node, int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::cerr << "LOCALITY_ERROR sched_setaffinity: " << strerror(errno) << std::endl;
    std::exit(3);
  }
  unsigned long mask = 1UL << node;
  long rc = syscall(SYS_set_mempolicy, MPOL_BIND, &mask,
                    sizeof(mask) * 8);
  if (rc != 0) {
    std::cerr << "LOCALITY_ERROR set_mempolicy: " << strerror(errno) << std::endl;
    std::exit(3);
  }
}

uint64_t aggregate(const BlockResult* results, size_t count,
                   uint64_t* errors = nullptr, uint64_t* first = nullptr) {
  uint64_t sum = 0, errs = 0, fst = kNoError;
  for (size_t i = 0; i < count; ++i) {
    sum += results[i].sum;
    errs += results[i].errors;
    fst = std::min(fst, results[i].first_error);
  }
  if (errors) *errors = errs;
  if (first) *first = fst;
  return sum;
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0;
  cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return ms;
}

void phase(const std::string& name) {
  std::cout << "PHASE " << name << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " OUTPUT_DIR NUMA_NODE CPU" << std::endl;
    return 1;
  }
  const std::string out_dir = argv[1];
  const int numa_node = std::stoi(argv[2]);
  const int cpu = std::stoi(argv[3]);
  const auto process_start = std::chrono::steady_clock::now();
  struct rusage usage_start{}, usage_end{};
  getrusage(RUSAGE_SELF, &usage_start);

  std::ofstream events(out_dir + "/events.log");
  std::ofstream seq_csv(out_dir + "/sequential_samples.csv");
  std::ofstream random_csv(out_dir + "/random_plane_samples.csv");
  std::ofstream step_csv(out_dir + "/decode_step_samples.csv");
  seq_csv << "iteration,latency_ms,bandwidth_GiB_s,checksum,checksum_ok\n";
  random_csv << "round,order,plane,latency_ms,bandwidth_GiB_s,checksum,checksum_ok\n";
  step_csv << "step,latency_ms,bandwidth_GiB_s,steps_per_s,checksum,checksum_ok,planes\n";

  set_locality(numa_node, cpu);
  phase("cuda_context");
  cuda_check(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync),
             "cudaSetDeviceFlags");
  cuda_check(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  if (!prop.canMapHostMemory) {
    std::cerr << "CUDA_ERROR device cannot map host memory" << std::endl;
    return 2;
  }
  size_t free_before = 0, total_vram = 0;
  cuda_check(cudaMemGetInfo(&free_before, &total_vram), "cudaMemGetInfo before");
  std::string mlocked_before = proc_value("/proc/meminfo", "Mlocked:");
  std::string unevictable_before = proc_value("/proc/meminfo", "Unevictable:");
  std::string vmlck_before = proc_value("/proc/self/status", "VmLck:");

  phase("allocate_and_first_touch");
  uint64_t* host = nullptr;
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host), kAllocationBytes,
                           cudaHostAllocMapped | cudaHostAllocPortable),
             "cudaHostAllocMapped 3.375GiB");
  uint64_t* device = nullptr;
  cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0),
             "cudaHostGetDevicePointer");
  for (uint64_t i = 0; i < kWords; ++i) host[i] = expected_word(i);
  __sync_synchronize();

  int placement_errno = 0;
  auto node_counts = query_page_nodes(host, kAllocationBytes, 8, &placement_errno);
  std::string numa_maps_line = numa_map_for(host);
  size_t free_after_main = 0, ignored = 0;
  cuda_check(cudaMemGetInfo(&free_after_main, &ignored), "cudaMemGetInfo after main");
  std::string mlocked_during = proc_value("/proc/meminfo", "Mlocked:");
  std::string unevictable_during = proc_value("/proc/meminfo", "Unevictable:");
  std::string vmlck_during = proc_value("/proc/self/status", "VmLck:");

  const size_t result_count = 12 * kPlaneBlocks;
  BlockResult* results = nullptr;
  BlockResult* device_results = nullptr;
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&results),
                           result_count * sizeof(BlockResult), cudaHostAllocMapped),
             "cudaHostAlloc results");
  cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_results),
                                      results, 0), "cudaHostGetDevicePointer results");
  uint64_t *indices = nullptr, *device_indices = nullptr;
  uint64_t *values = nullptr, *device_values = nullptr;
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&indices), 16 * sizeof(uint64_t),
                           cudaHostAllocMapped), "cudaHostAlloc indices");
  cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&values), 16 * sizeof(uint64_t),
                           cudaHostAllocMapped), "cudaHostAlloc values");
  cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_indices),
                                      indices, 0), "device pointer indices");
  cuda_check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_values),
                                      values, 0), "device pointer values");

  cudaEvent_t start_event, stop_event;
  cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
  cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");

  phase("full_correctness_read");
  read_kernel<true><<<kFullBlocks, kThreads>>>(device, 0, kWords,
                                               device_results, 0);
  cuda_check(cudaGetLastError(), "full validation launch");
  cuda_check(cudaDeviceSynchronize(), "full validation synchronize");
  uint64_t full_errors = 0, first_error = kNoError;
  uint64_t full_sum = aggregate(results, kFullBlocks, &full_errors, &first_error);
  uint64_t full_expected_sum = expected_sum(0, kWords);

  const std::vector<uint64_t> boundary_list = {
      0, 1, kPlaneWords - 1, kPlaneWords, kWords / 2 - 1,
      kWords / 2, kWords - 2, kWords - 1};
  for (size_t i = 0; i < boundary_list.size(); ++i) indices[i] = boundary_list[i];
  boundary_read_kernel<<<1, 32>>>(device, device_indices, device_values,
                                  boundary_list.size());
  cuda_check(cudaGetLastError(), "boundary read launch");
  cuda_check(cudaDeviceSynchronize(), "boundary read sync");
  bool boundaries_ok = true;
  for (size_t i = 0; i < boundary_list.size(); ++i)
    boundaries_ok &= values[i] == expected_word(boundary_list[i]);

  phase("gpu_write_cpu_visibility");
  for (size_t i = 0; i < boundary_list.size(); ++i)
    values[i] = 0xa5a5000000000000ULL ^ boundary_list[i];
  boundary_write_kernel<<<1, 32>>>(device, device_indices, device_values,
                                   boundary_list.size());
  cuda_check(cudaGetLastError(), "boundary write launch");
  cuda_check(cudaDeviceSynchronize(), "boundary write sync");
  bool writes_visible = true;
  for (size_t i = 0; i < boundary_list.size(); ++i)
    writes_visible &= host[boundary_list[i]] == values[i];
  for (uint64_t idx : boundary_list) host[idx] = expected_word(idx);
  __sync_synchronize();

  phase("sequential_benchmark");
  std::vector<double> seq_ms;
  bool seq_checksums_ok = true;
  for (int iteration = 0; iteration < 6; ++iteration) {
    cuda_check(cudaEventRecord(start_event), "record sequential start");
    read_kernel<false><<<kFullBlocks, kThreads>>>(device, 0, kWords,
                                                  device_results, 0);
    cuda_check(cudaEventRecord(stop_event), "record sequential stop");
    cuda_check(cudaEventSynchronize(stop_event), "sync sequential event");
    double ms = elapsed_ms(start_event, stop_event);
    uint64_t checksum = aggregate(results, kFullBlocks);
    bool ok = checksum == full_expected_sum;
    seq_checksums_ok &= ok;
    seq_csv << iteration << ',' << std::fixed << std::setprecision(6) << ms << ','
            << (3.375 / (ms / 1000.0)) << ',' << checksum << ',' << ok << '\n';
    if (iteration > 0) seq_ms.push_back(ms);  // first pass is an unscored warmup
  }

  phase("random_plane_benchmark");
  std::mt19937 rng(0x5eed1234);
  std::vector<int> order(kPlaneCount);
  std::iota(order.begin(), order.end(), 0);
  std::vector<double> random_ms;
  bool random_checksums_ok = true;
  for (int round = 0; round < 3; ++round) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t ordinal = 0; ordinal < order.size(); ++ordinal) {
      int plane = order[ordinal];
      uint64_t start_word = static_cast<uint64_t>(plane) * kPlaneWords;
      cuda_check(cudaEventRecord(start_event), "record plane start");
      read_kernel<false><<<kPlaneBlocks, kThreads>>>(device, start_word, kPlaneWords,
                                                     device_results, 0);
      cuda_check(cudaEventRecord(stop_event), "record plane stop");
      cuda_check(cudaEventSynchronize(stop_event), "sync plane event");
      double ms = elapsed_ms(start_event, stop_event);
      uint64_t checksum = aggregate(results, kPlaneBlocks);
      bool ok = checksum == expected_sum(start_word, kPlaneWords);
      random_checksums_ok &= ok;
      random_ms.push_back(ms);
      random_csv << round << ',' << ordinal << ',' << plane << ',' << std::fixed
                 << std::setprecision(6) << ms << ','
                 << ((6.75 / 1024.0) / (ms / 1000.0)) << ',' << checksum << ','
                 << ok << '\n';
    }
  }

  phase("two_layer_decode_benchmark");
  std::vector<double> step_ms;
  bool step_checksums_ok = true;
  std::vector<int> layer0(256), layer1(256);
  std::iota(layer0.begin(), layer0.end(), 0);
  std::iota(layer1.begin(), layer1.end(), 256);
  for (int step = 0; step < 300; ++step) {
    std::shuffle(layer0.begin(), layer0.end(), rng);
    std::shuffle(layer1.begin(), layer1.end(), rng);
    std::vector<int> selected;
    selected.insert(selected.end(), layer0.begin(), layer0.begin() + 6);
    selected.insert(selected.end(), layer1.begin(), layer1.begin() + 6);
    cuda_check(cudaEventRecord(start_event), "record step start");
    for (size_t j = 0; j < selected.size(); ++j) {
      uint64_t start_word = static_cast<uint64_t>(selected[j]) * kPlaneWords;
      read_kernel<false><<<kPlaneBlocks, kThreads>>>(
          device, start_word, kPlaneWords, device_results, j * kPlaneBlocks);
    }
    cuda_check(cudaEventRecord(stop_event), "record step stop");
    cuda_check(cudaEventSynchronize(stop_event), "sync step event");
    double ms = elapsed_ms(start_event, stop_event);
    uint64_t checksum = aggregate(results, selected.size() * kPlaneBlocks);
    uint64_t expected = 0;
    for (int plane : selected)
      expected += expected_sum(static_cast<uint64_t>(plane) * kPlaneWords,
                               kPlaneWords);
    bool ok = checksum == expected;
    step_checksums_ok &= ok;
    step_ms.push_back(ms);
    step_csv << step << ',' << std::fixed << std::setprecision(6) << ms << ','
             << ((81.0 / 1024.0) / (ms / 1000.0)) << ',' << (1000.0 / ms)
             << ',' << checksum << ',' << ok << ',';
    for (size_t j = 0; j < selected.size(); ++j) {
      if (j) step_csv << ':';
      step_csv << selected[j];
    }
    step_csv << '\n';
  }

  phase("cleanup");
  Stats seq_stats = stats(seq_ms);
  Stats random_stats = stats(random_ms);
  Stats step_stats = stats(step_ms);
  size_t free_before_cleanup = 0;
  cuda_check(cudaMemGetInfo(&free_before_cleanup, &ignored),
             "cudaMemGetInfo before cleanup");
  cuda_check(cudaEventDestroy(start_event), "destroy start event");
  cuda_check(cudaEventDestroy(stop_event), "destroy stop event");
  cuda_check(cudaFreeHost(values), "cudaFreeHost values");
  cuda_check(cudaFreeHost(indices), "cudaFreeHost indices");
  cuda_check(cudaFreeHost(results), "cudaFreeHost results");
  cuda_check(cudaFreeHost(host), "cudaFreeHost main");
  size_t free_after_cleanup = 0;
  cuda_check(cudaMemGetInfo(&free_after_cleanup, &ignored),
             "cudaMemGetInfo after cleanup");
  std::string mlocked_after = proc_value("/proc/meminfo", "Mlocked:");
  std::string unevictable_after = proc_value("/proc/meminfo", "Unevictable:");
  std::string vmlck_after = proc_value("/proc/self/status", "VmLck:");
  cuda_check(cudaDeviceReset(), "cudaDeviceReset");

  getrusage(RUSAGE_SELF, &usage_end);
  const auto process_end = std::chrono::steady_clock::now();
  double wall_s = std::chrono::duration<double>(process_end - process_start).count();
  auto timeval_s = [](const timeval& t) { return t.tv_sec + t.tv_usec / 1e6; };
  double cpu_s = timeval_s(usage_end.ru_utime) + timeval_s(usage_end.ru_stime) -
                 timeval_s(usage_start.ru_utime) - timeval_s(usage_start.ru_stime);

  std::ofstream json(out_dir + "/results.json");
  json << std::fixed << std::setprecision(6);
  json << "{\n"
       << "  \"device\": \"" << json_escape(prop.name) << "\",\n"
       << "  \"device_pci\": \"" << std::hex << std::setw(4) << std::setfill('0')
       << prop.pciDomainID << ':' << std::setw(2) << prop.pciBusID << ':'
       << std::setw(2) << prop.pciDeviceID << std::dec << "\",\n"
       << "  \"compute_capability\": \"" << prop.major << '.' << prop.minor << "\",\n"
       << "  \"can_map_host_memory\": " << (prop.canMapHostMemory ? "true" : "false") << ",\n"
       << "  \"unified_addressing\": " << (prop.unifiedAddressing ? "true" : "false") << ",\n"
       << "  \"numa_node\": " << numa_node << ",\n"
       << "  \"bound_cpu\": " << cpu << ",\n"
       << "  \"allocation_bytes\": " << kAllocationBytes << ",\n"
       << "  \"allocation_GiB\": 3.375000,\n"
       << "  \"plane_bytes\": " << kPlaneBytes << ",\n"
       << "  \"plane_count\": " << kPlaneCount << ",\n"
       << "  \"host_pointer\": \"" << static_cast<void*>(host) << "\",\n"
       << "  \"device_pointer_same_under_UVA\": " << (host == device ? "true" : "false") << ",\n"
       << "  \"vram_total_bytes\": " << total_vram << ",\n"
       << "  \"vram_free_before_host_alloc\": " << free_before << ",\n"
       << "  \"vram_free_after_host_alloc\": " << free_after_main << ",\n"
       << "  \"vram_delta_for_3_375GiB_host_alloc_bytes\": "
       << static_cast<int64_t>(free_before) - static_cast<int64_t>(free_after_main) << ",\n"
       << "  \"vram_free_before_cleanup\": " << free_before_cleanup << ",\n"
       << "  \"vram_free_after_cleanup\": " << free_after_cleanup << ",\n"
       << "  \"mlocked_before\": \"" << json_escape(mlocked_before) << "\",\n"
       << "  \"mlocked_during\": \"" << json_escape(mlocked_during) << "\",\n"
       << "  \"mlocked_after\": \"" << json_escape(mlocked_after) << "\",\n"
       << "  \"unevictable_before\": \"" << json_escape(unevictable_before) << "\",\n"
       << "  \"unevictable_during\": \"" << json_escape(unevictable_during) << "\",\n"
       << "  \"unevictable_after\": \"" << json_escape(unevictable_after) << "\",\n"
       << "  \"process_vmlck_before\": \"" << json_escape(vmlck_before) << "\",\n"
       << "  \"process_vmlck_during\": \"" << json_escape(vmlck_during) << "\",\n"
       << "  \"process_vmlck_after\": \"" << json_escape(vmlck_after) << "\",\n"
       << "  \"move_pages_errno\": " << placement_errno << ",\n"
       << "  \"page_counts_by_node\": [";
  for (size_t i = 0; i < node_counts.size(); ++i) {
    if (i) json << ',';
    json << node_counts[i];
  }
  json << "],\n"
       << "  \"numa_maps_line\": \"";
  json << json_escape(numa_maps_line);
  json << "\",\n"
       << "  \"full_read_checksum\": " << full_sum << ",\n"
       << "  \"full_read_expected_checksum\": " << full_expected_sum << ",\n"
       << "  \"full_read_error_count\": " << full_errors << ",\n"
       << "  \"first_error_word\": " << first_error << ",\n"
       << "  \"boundaries_ok\": " << (boundaries_ok ? "true" : "false") << ",\n"
       << "  \"gpu_writes_cpu_visible\": " << (writes_visible ? "true" : "false") << ",\n"
       << "  \"sequential_checksums_ok\": " << (seq_checksums_ok ? "true" : "false") << ",\n"
       << "  \"random_plane_checksums_ok\": " << (random_checksums_ok ? "true" : "false") << ",\n"
       << "  \"decode_step_checksums_ok\": " << (step_checksums_ok ? "true" : "false") << ",\n"
       << "  \"sequential_latency_ms\": {\"mean\": " << seq_stats.mean
       << ", \"stddev\": " << seq_stats.stddev << ", \"min\": " << seq_stats.min
       << ", \"median\": " << seq_stats.median << ", \"p95\": " << seq_stats.p95
       << ", \"max\": " << seq_stats.max << "},\n"
       << "  \"sequential_bandwidth_GiB_s_mean\": " << 3.375 / (seq_stats.mean / 1000.0) << ",\n"
       << "  \"random_plane_latency_ms\": {\"mean\": " << random_stats.mean
       << ", \"stddev\": " << random_stats.stddev << ", \"min\": " << random_stats.min
       << ", \"median\": " << random_stats.median << ", \"p95\": " << random_stats.p95
       << ", \"max\": " << random_stats.max << "},\n"
       << "  \"random_plane_bandwidth_GiB_s_mean\": " << (6.75 / 1024.0) / (random_stats.mean / 1000.0) << ",\n"
       << "  \"decode_step_latency_ms\": {\"mean\": " << step_stats.mean
       << ", \"stddev\": " << step_stats.stddev << ", \"min\": " << step_stats.min
       << ", \"median\": " << step_stats.median << ", \"p95\": " << step_stats.p95
       << ", \"max\": " << step_stats.max << "},\n"
       << "  \"decode_step_bandwidth_GiB_s_mean\": " << (81.0 / 1024.0) / (step_stats.mean / 1000.0) << ",\n"
       << "  \"decode_steps_per_second_mean\": " << 1000.0 / step_stats.mean << ",\n"
       << "  \"target_60_steps_per_second_met\": " << ((1000.0 / step_stats.mean) >= 60.0 ? "true" : "false") << ",\n"
       << "  \"process_wall_seconds\": " << wall_s << ",\n"
       << "  \"process_cpu_seconds\": " << cpu_s << ",\n"
       << "  \"process_average_cpu_percent\": " << cpu_s / wall_s * 100.0 << "\n"
       << "}\n";
  json.close();
  seq_csv.close(); random_csv.close(); step_csv.close(); events.close();

  bool all_ok = full_errors == 0 && full_sum == full_expected_sum && boundaries_ok &&
                writes_visible && seq_checksums_ok && random_checksums_ok &&
                step_checksums_ok && !node_counts.empty() &&
                node_counts.size() > static_cast<size_t>(numa_node) &&
                node_counts[numa_node] == kAllocationBytes / sysconf(_SC_PAGESIZE);
  std::cout << "RESULT " << (all_ok ? "PASS" : "FAIL")
            << " full_errors=" << full_errors
            << " seq_GiB_s=" << 3.375 / (seq_stats.mean / 1000.0)
            << " step_ms=" << step_stats.mean
            << " step_GiB_s=" << (81.0 / 1024.0) / (step_stats.mean / 1000.0)
            << " steps_s=" << 1000.0 / step_stats.mean << std::endl;
  return all_ok ? 0 : 4;
}
