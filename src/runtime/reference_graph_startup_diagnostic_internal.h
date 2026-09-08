#pragma once

// Isolated startup-attribution build only. These observations never select a
// route or admission result and must not be called during stream capture.
#include <cuda_runtime_api.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

namespace q3x::runtime::graph_startup_diagnostic_detail {

inline long long read_kib_field(const char* const text,
                                const char* const field) noexcept {
  const std::size_t length = std::strlen(field);
  for (const char* line = text; line != nullptr && *line != '\0';) {
    if (std::strncmp(line, field, length) == 0 && line[length] == ':') {
      long long value = -1;
      char unit[3]{};
      if (std::sscanf(line + length + 1U, "%lld %2s", &value, unit) == 2 &&
          value >= 0 && std::strcmp(unit, "kB") == 0) {
        return value;
      }
      return -1;
    }
    line = std::strchr(line, '\n');
    if (line != nullptr) {
      ++line;
    }
  }
  return -1;
}

template <std::size_t Capacity>
inline bool read_proc(const char* const path,
                      char (&output)[Capacity]) noexcept {
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  std::size_t used = 0U;
  bool complete = false;
  while (used + 1U < Capacity) {
    const ssize_t count = ::read(fd, output + used, Capacity - used - 1U);
    if (count == 0) {
      complete = true;
      break;
    }
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    used += static_cast<std::size_t>(count);
  }
  output[used] = '\0';
  (void)::close(fd);
  return complete;
}

inline void emit(const char* const stage, const unsigned int position,
                 const cudaError_t cuda_status, const std::size_t free_bytes,
                 const std::size_t total_bytes) noexcept {
  const int saved_errno = errno;
  char status[8192]{};
  char memory[8192]{};
  const bool status_complete = read_proc("/proc/self/status", status);
  const bool memory_complete = read_proc("/proc/meminfo", memory);
  timespec now{};
  const bool time_valid = ::clock_gettime(CLOCK_MONOTONIC, &now) == 0;
  char line[1024]{};
  const int length = std::snprintf(
      line, sizeof(line),
      "graph-startup-diagnostic stage=%s position=%u pid=%ld "
      "monotonic_seconds=%lld monotonic_nanoseconds=%ld time_valid=%d "
      "cuda_status=%d cuda_free_bytes=%zu cuda_total_bytes=%zu "
      "status_complete=%d memory_complete=%d VmRSS_kib=%lld RssAnon_kib=%lld "
      "RssFile_kib=%lld MemFree_kib=%lld MemAvailable_kib=%lld Cached_kib=%lld "
      "AnonPages_kib=%lld Slab_kib=%lld SUnreclaim_kib=%lld "
      "NvMapMemUsed_kib=%lld\n",
      stage, position, static_cast<long>(::getpid()),
      static_cast<long long>(now.tv_sec), now.tv_nsec,
      static_cast<int>(time_valid), static_cast<int>(cuda_status), free_bytes,
      total_bytes, static_cast<int>(status_complete),
      static_cast<int>(memory_complete), read_kib_field(status, "VmRSS"),
      read_kib_field(status, "RssAnon"), read_kib_field(status, "RssFile"),
      read_kib_field(memory, "MemFree"),
      read_kib_field(memory, "MemAvailable"), read_kib_field(memory, "Cached"),
      read_kib_field(memory, "AnonPages"), read_kib_field(memory, "Slab"),
      read_kib_field(memory, "SUnreclaim"),
      read_kib_field(memory, "NvMapMemUsed"));
  if (length > 0 && static_cast<std::size_t>(length) < sizeof(line)) {
    ssize_t written = -1;
    do {
      written = ::write(STDERR_FILENO, line, static_cast<std::size_t>(length));
    } while (written < 0 && errno == EINTR);
  }
  errno = saved_errno;
}

inline void sample(const char* const stage,
                   const unsigned int position) noexcept {
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  emit(stage, position, status, free_bytes, total_bytes);
}

}  // namespace q3x::runtime::graph_startup_diagnostic_detail
