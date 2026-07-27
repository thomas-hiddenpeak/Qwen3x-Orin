#include "q3x/runtime/resident_weights.h"

#include "q3x/core/sha256.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <linux/if_alg.h>
#include <sys/socket.h>
#endif

namespace q3x::runtime {
namespace {

namespace st = io::safetensors;
namespace mw = model::weights;

constexpr std::uint64_t kMaximumChunkBytes =
    256ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumArenaBytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumTensors = 1'000'000U;
constexpr std::size_t kMaximumShards = 1'024U;
constexpr std::size_t kMaximumParallelShards = 16U;
constexpr std::uint64_t kMaximumPinnedStagingBytes =
    2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kMaximumMemcpyOperations = 10'000'000ULL;
constexpr std::size_t kStagingSlots = 2U;

ResidentLoadDiagnostic make_diagnostic(ResidentLoadErrorCode code,
                                       std::string message,
                                       std::string context = {},
                                       std::string shard = {},
                                       std::uint64_t offset = 0,
                                       std::string expected = {},
                                       std::string actual = {},
                                       int system_error = 0,
                                       int cuda_error = 0) {
    ResidentLoadDiagnostic diagnostic;
    diagnostic.code = code;
    diagnostic.message = std::move(message);
    diagnostic.context = std::move(context);
    diagnostic.shard = std::move(shard);
    diagnostic.offset = offset;
    diagnostic.system_error = system_error;
    diagnostic.cuda_error = cuda_error;
    diagnostic.expected = std::move(expected);
    diagnostic.actual = std::move(actual);
    return diagnostic;
}

PlanResult plan_failure(ResidentLoadDiagnostic diagnostic) {
    PlanResult result;
    result.diagnostic = std::move(diagnostic);
    return result;
}

ResidentLoadResult load_failure(ResidentLoadDiagnostic diagnostic) {
    ResidentLoadResult result;
    result.diagnostic = std::move(diagnostic);
    return result;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t& output) noexcept {
    if (left != 0U &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool checked_align_up(std::uint64_t value,
                      std::uint64_t alignment,
                      std::uint64_t& output) noexcept {
    const std::uint64_t mask = alignment - 1U;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        return false;
    }
    output = (value + mask) & ~mask;
    return true;
}

bool is_lower_hex_digest(std::string_view digest) noexcept {
    if (digest.size() != 64U) {
        return false;
    }
    return std::all_of(digest.begin(), digest.end(), [](const char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

bool tensor_bytes_match(const mw::TensorLocator& locator) noexcept {
    std::uint64_t elements = 1U;
    for (const std::uint64_t dimension : locator.shape) {
        if (!checked_multiply(elements, dimension, elements)) {
            return false;
        }
    }
    std::uint64_t bits = 0;
    return checked_multiply(
               elements, static_cast<std::uint64_t>(st::bit_width(locator.dtype)), bits) &&
           (bits % 8U) == 0U && bits / 8U == locator.byte_size;
}

class UniqueFd {
  public:
    UniqueFd() noexcept = default;
    explicit UniqueFd(int descriptor) noexcept : descriptor_(descriptor) {}
    ~UniqueFd() {
        if (descriptor_ >= 0) {
            (void)::close(descriptor_);
        }
    }

    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    UniqueFd(UniqueFd&& other) noexcept : descriptor_(other.release()) {}
    UniqueFd& operator=(UniqueFd&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return descriptor_; }
    [[nodiscard]] explicit operator bool() const noexcept {
        return descriptor_ >= 0;
    }

    void reset(int descriptor = -1) noexcept {
        if (descriptor_ >= 0) {
            (void)::close(descriptor_);
        }
        descriptor_ = descriptor;
    }

    [[nodiscard]] int release() noexcept {
        const int descriptor = descriptor_;
        descriptor_ = -1;
        return descriptor;
    }

  private:
    int descriptor_ = -1;
};

bool is_valid_sha256_backend(const ResidentSha256Backend backend) noexcept {
    switch (backend) {
        case ResidentSha256Backend::kAuto:
        case ResidentSha256Backend::kPortable:
        case ResidentSha256Backend::kLinuxAfAlg:
            return true;
    }
    return false;
}

bool is_valid_resident_payload(const ResidentPayload payload) noexcept {
    switch (payload) {
        case ResidentPayload::kText:
        case ResidentPayload::kMtp:
            return true;
    }
    return false;
}

struct AfAlgOperationResult {
    std::vector<UniqueFd> operations;
    int error = 0;
    std::string context;

    [[nodiscard]] bool ok() const noexcept {
        return error == 0 && !operations.empty();
    }
};

AfAlgOperationResult create_af_alg_sha256_operations(
    const std::size_t operation_count) {
    AfAlgOperationResult result;
#if defined(__linux__)
    UniqueFd control(
        ::socket(AF_ALG, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
    if (!control) {
        result.error = errno;
        result.context = "socket(AF_ALG)";
        return result;
    }

    struct sockaddr_alg address {};
    address.salg_family = AF_ALG;
    constexpr char kHashType[] = "hash";
    constexpr char kSha256Name[] = "sha256";
    static_assert(sizeof(kHashType) <= sizeof(address.salg_type));
    static_assert(sizeof(kSha256Name) <= sizeof(address.salg_name));
    std::memcpy(address.salg_type, kHashType, sizeof(kHashType));
    std::memcpy(address.salg_name, kSha256Name, sizeof(kSha256Name));
    if (::bind(control.get(),
               reinterpret_cast<const struct sockaddr*>(&address),
               sizeof(address)) != 0) {
        result.error = errno;
        result.context = "bind(AF_ALG sha256)";
        return result;
    }

    result.operations.reserve(operation_count);
    for (std::size_t index = 0U; index < operation_count; ++index) {
        int descriptor = -1;
        do {
            descriptor = ::accept4(control.get(), nullptr, nullptr, SOCK_CLOEXEC);
        } while (descriptor < 0 && errno == EINTR);
        if (descriptor < 0) {
            result.error = errno;
            result.context = "accept4(AF_ALG sha256)";
            result.operations.clear();
            return result;
        }
        result.operations.emplace_back(descriptor);
    }
    return result;
#else
    (void)operation_count;
    result.error = ENOTSUP;
    result.context = "AF_ALG is unavailable on this platform";
    return result;
#endif
}

struct HashBackendPreparation {
    ResidentSha256Backend backend = ResidentSha256Backend::kPortable;
    std::vector<UniqueFd> af_alg_operations;
    int error = 0;
    std::string context;

    [[nodiscard]] bool ok() const noexcept { return error == 0; }
};

HashBackendPreparation prepare_hash_backend(
    const ResidentSha256Backend requested,
    const std::size_t shard_count) {
    HashBackendPreparation result;
    if (requested == ResidentSha256Backend::kPortable) {
        return result;
    }

    AfAlgOperationResult accelerated =
        create_af_alg_sha256_operations(shard_count);
    if (accelerated.ok()) {
        result.backend = ResidentSha256Backend::kLinuxAfAlg;
        result.af_alg_operations = std::move(accelerated.operations);
        return result;
    }
    if (requested == ResidentSha256Backend::kAuto) {
        return result;
    }
    result.error = accelerated.error == 0 ? EIO : accelerated.error;
    result.context = std::move(accelerated.context);
    return result;
}

struct HashOperationResult {
    std::optional<core::Sha256Digest> digest;
    int error = 0;
    std::string context;
};

class ShardSha256 {
  public:
    explicit ShardSha256(const ResidentSha256Backend backend,
                         UniqueFd af_alg_operation = {}) noexcept
        : backend_(backend), af_alg_operation_(std::move(af_alg_operation)) {}

    ShardSha256(const ShardSha256&) = delete;
    ShardSha256& operator=(const ShardSha256&) = delete;

    [[nodiscard]] bool update(const void* const data,
                              const std::size_t size,
                              int& error) noexcept {
        if (finalized_ || (data == nullptr && size != 0U) ||
            !checked_add(bytes_hashed_,
                         static_cast<std::uint64_t>(size),
                         bytes_hashed_)) {
            error = EOVERFLOW;
            return false;
        }
        if (backend_ == ResidentSha256Backend::kPortable) {
            if (!portable_.update(data, size)) {
                error = EOVERFLOW;
                return false;
            }
            return true;
        }
#if defined(__linux__)
        if (backend_ != ResidentSha256Backend::kLinuxAfAlg ||
            !af_alg_operation_) {
            error = ENOTSUP;
            return false;
        }
        const auto* input = static_cast<const std::uint8_t*>(data);
        std::size_t sent = 0U;
        while (sent < size) {
            const ssize_t count = ::send(af_alg_operation_.get(),
                                         input + sent,
                                         size - sent,
                                         MSG_MORE | MSG_NOSIGNAL);
            if (count > 0) {
                sent += static_cast<std::size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            error = count == 0 ? EIO : errno;
            return false;
        }
        return true;
#else
        (void)data;
        (void)size;
        error = ENOTSUP;
        return false;
#endif
    }

    [[nodiscard]] HashOperationResult finalize() noexcept {
        HashOperationResult result;
        if (finalized_) {
            result.error = EINVAL;
            result.context = "SHA-256 operation was already finalized";
            return result;
        }
        finalized_ = true;
        if (backend_ == ResidentSha256Backend::kPortable) {
            result.digest.emplace(portable_.finalize());
            return result;
        }
#if defined(__linux__)
        if (backend_ != ResidentSha256Backend::kLinuxAfAlg ||
            !af_alg_operation_) {
            result.error = ENOTSUP;
            result.context = "AF_ALG SHA-256 operation is unavailable";
            return result;
        }

        // Every non-empty update uses MSG_MORE. A zero-length send without
        // MSG_MORE explicitly terminates the hash message before reading the
        // standard 32-byte digest.
        constexpr std::uint8_t kEmpty = 0U;
        ssize_t terminated = -1;
        do {
            terminated = ::send(af_alg_operation_.get(),
                                &kEmpty,
                                0U,
                                MSG_NOSIGNAL);
        } while (terminated < 0 && errno == EINTR);
        if (terminated != 0) {
            result.error = terminated < 0 ? errno : EPROTO;
            result.context = "send(AF_ALG SHA-256 final)";
            return result;
        }

        core::Sha256Digest digest;
        ssize_t received = -1;
        do {
            received = ::recv(af_alg_operation_.get(),
                              digest.bytes.data(),
                              digest.bytes.size(),
                              MSG_WAITALL);
        } while (received < 0 && errno == EINTR);
        if (received != static_cast<ssize_t>(digest.bytes.size())) {
            result.error = received < 0 ? errno : EPROTO;
            result.context = "recv(AF_ALG SHA-256 digest)";
            return result;
        }
        result.digest.emplace(digest);
        return result;
#else
        result.error = ENOTSUP;
        result.context = "AF_ALG is unavailable on this platform";
        return result;
#endif
    }

  private:
    ResidentSha256Backend backend_ = ResidentSha256Backend::kPortable;
    core::Sha256 portable_;
    UniqueFd af_alg_operation_;
    std::uint64_t bytes_hashed_ = 0U;
    bool finalized_ = false;
};

struct OpenResult {
    UniqueFd descriptor;
    int error = 0;
    std::string component;
};

OpenResult open_relative_nofollow(const int root_descriptor,
                                  const std::string& relative_path) {
    OpenResult result;
    UniqueFd current_directory;
    int parent = root_descriptor;
    std::size_t begin = 0U;
    while (begin < relative_path.size()) {
        const std::size_t slash = relative_path.find('/', begin);
        const bool final_component = slash == std::string::npos;
        const std::size_t end = final_component ? relative_path.size() : slash;
        const std::string component = relative_path.substr(begin, end - begin);
        const int flags = final_component
                              ? O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK
                              : O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_DIRECTORY;
        const int descriptor = ::openat(parent, component.c_str(), flags);
        if (descriptor < 0) {
            result.error = errno;
            result.component = component;
            return result;
        }
        if (final_component) {
            result.descriptor.reset(descriptor);
            return result;
        }
        current_directory.reset(descriptor);
        parent = current_directory.get();
        begin = slash + 1U;
    }
    result.error = EINVAL;
    return result;
}

struct OpenShard {
    const PlannedShard* plan = nullptr;
    UniqueFd descriptor;
};

struct CudaArena {
    void* data = nullptr;

    CudaArena() = default;
    CudaArena(const CudaArena&) = delete;
    CudaArena& operator=(const CudaArena&) = delete;

    ~CudaArena() {
        if (data != nullptr) {
            (void)cudaFree(data);
            data = nullptr;
        }
        (void)cudaGetLastError();
    }

    [[nodiscard]] void* release() noexcept {
        return std::exchange(data, nullptr);
    }
};

struct CudaWorkerResources {
    int device = -1;
    cudaStream_t stream = nullptr;
    std::array<cudaEvent_t, kStagingSlots> events{};
    std::array<void*, kStagingSlots> staging{};

    CudaWorkerResources() = default;
    CudaWorkerResources(const CudaWorkerResources&) = delete;
    CudaWorkerResources& operator=(const CudaWorkerResources&) = delete;

    ~CudaWorkerResources() {
        if (device >= 0) {
            (void)cudaSetDevice(device);
        }
        if (stream != nullptr) {
            (void)cudaStreamSynchronize(stream);
        }
        for (cudaEvent_t& event : events) {
            if (event != nullptr) {
                (void)cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (void*& buffer : staging) {
            if (buffer != nullptr) {
                (void)cudaFreeHost(buffer);
                buffer = nullptr;
            }
        }
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
            stream = nullptr;
        }
        (void)cudaGetLastError();
    }
};

class JoiningThreads {
  public:
    JoiningThreads() = default;
    JoiningThreads(const JoiningThreads&) = delete;
    JoiningThreads& operator=(const JoiningThreads&) = delete;

    ~JoiningThreads() { join_all(); }

    void reserve(const std::size_t count) { threads_.reserve(count); }

    template <typename Function>
    void emplace_back(Function&& function) {
        threads_.emplace_back(std::forward<Function>(function));
    }

    void join_all() {
        for (std::thread& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

  private:
    std::vector<std::thread> threads_;
};

ResidentLoadDiagnostic cuda_diagnostic(cudaError_t status,
                                       std::string context,
                                       std::string shard = {},
                                       std::uint64_t offset = 0U) {
    const char* const name = cudaGetErrorName(status);
    const char* const description = cudaGetErrorString(status);
    std::string actual = name == nullptr ? "unknown CUDA error" : name;
    if (description != nullptr) {
        actual += ": ";
        actual += description;
    }
    return make_diagnostic(ResidentLoadErrorCode::kCudaFailure,
                           "CUDA operation failed",
                           std::move(context),
                           std::move(shard),
                           offset,
                           {},
                           std::move(actual),
                           0,
                           static_cast<int>(status));
}

bool add_stat(std::uint64_t& target, std::uint64_t value) noexcept {
    return checked_add(target, value, target);
}

struct ReadChunkResult {
    std::size_t bytes = 0U;
    int error = 0;
    bool eof = false;
};

ReadChunkResult read_exact_chunk(int descriptor,
                                 void* destination,
                                 std::size_t requested) noexcept {
    auto* output = static_cast<std::uint8_t*>(destination);
    ReadChunkResult result;
    while (result.bytes < requested) {
        const ssize_t count =
            ::read(descriptor, output + result.bytes, requested - result.bytes);
        if (count > 0) {
            result.bytes += static_cast<std::size_t>(count);
            continue;
        }
        if (count == 0) {
            result.eof = true;
            return result;
        }
        if (errno == EINTR) {
            continue;
        }
        result.error = errno;
        return result;
    }
    return result;
}

struct ShardLoadOutcome {
    ShardLoadStats stats;
    ResidentLoadDiagnostic diagnostic;
    bool attempted = false;

    [[nodiscard]] bool ok() const noexcept {
        return attempted &&
               diagnostic.code == ResidentLoadErrorCode::kNone;
    }
};

ShardLoadOutcome load_one_shard(
    const OpenShard& opened_shard,
    const ResidentLoadPlan& plan,
    const ResidentLoadOptions& options,
    const ResidentSha256Backend hash_backend,
    UniqueFd af_alg_operation,
    std::uint8_t* const arena_bytes,
    CudaWorkerResources& resources) {
    const PlannedShard& shard = *opened_shard.plan;
    ShardLoadOutcome outcome;
    outcome.attempted = true;
    outcome.stats.filename = shard.identity.filename;
    ShardSha256 hasher(hash_backend, std::move(af_alg_operation));
    std::array<bool, kStagingSlots> slot_pending{};
    std::uint64_t file_offset = 0U;
    std::uint64_t chunk_index = 0U;
    std::size_t segment_cursor = 0U;

    while (file_offset < shard.identity.file_size) {
        const std::size_t slot =
            static_cast<std::size_t>(chunk_index % kStagingSlots);
        if (slot_pending[slot]) {
            const cudaError_t cuda_status =
                cudaEventSynchronize(resources.events[slot]);
            if (cuda_status != cudaSuccess) {
                outcome.diagnostic = cuda_diagnostic(
                    cuda_status,
                    "cudaEventSynchronize(staging reuse)",
                    shard.identity.filename,
                    file_offset);
                return outcome;
            }
            slot_pending[slot] = false;
        }

        const std::uint64_t remaining =
            shard.identity.file_size - file_offset;
        const std::size_t chunk_size = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, options.chunk_bytes));
        const ReadChunkResult read_result = read_exact_chunk(
            opened_shard.descriptor.get(), resources.staging[slot], chunk_size);
        if (read_result.error != 0) {
            outcome.diagnostic = make_diagnostic(
                ResidentLoadErrorCode::kIoFailure,
                "sequential shard read failed",
                "read",
                shard.identity.filename,
                file_offset + read_result.bytes,
                std::to_string(chunk_size),
                std::strerror(read_result.error),
                read_result.error);
            return outcome;
        }
        if (read_result.eof || read_result.bytes != chunk_size) {
            outcome.diagnostic = make_diagnostic(
                ResidentLoadErrorCode::kShardSizeMismatch,
                "shard was truncated during sequential load",
                "read",
                shard.identity.filename,
                file_offset + read_result.bytes,
                std::to_string(chunk_size),
                std::to_string(read_result.bytes));
            return outcome;
        }
        int hash_error = 0;
        if (!hasher.update(resources.staging[slot], chunk_size, hash_error)) {
            outcome.diagnostic = make_diagnostic(
                ResidentLoadErrorCode::kSha256Failure,
                "SHA-256 streaming update failed",
                std::string(to_string(hash_backend)) + ".update",
                shard.identity.filename,
                file_offset,
                {},
                std::strerror(hash_error),
                hash_error);
            return outcome;
        }

        std::uint64_t chunk_end = 0U;
        if (!checked_add(file_offset,
                         static_cast<std::uint64_t>(chunk_size),
                         chunk_end) ||
            !add_stat(outcome.stats.bytes_read, chunk_size) ||
            !add_stat(outcome.stats.chunks, 1U)) {
            outcome.diagnostic = make_diagnostic(
                ResidentLoadErrorCode::kArithmeticOverflow,
                "loader statistics overflow uint64",
                "statistics",
                shard.identity.filename,
                file_offset);
            return outcome;
        }

        while (segment_cursor < shard.tensor_indices.size() &&
               plan.tensors[shard.tensor_indices[segment_cursor]].source_end <=
                   file_offset) {
            ++segment_cursor;
        }
        for (std::size_t segment = segment_cursor;
             segment < shard.tensor_indices.size(); ++segment) {
            const PlannedTensor& tensor =
                plan.tensors[shard.tensor_indices[segment]];
            if (tensor.source_begin >= chunk_end) {
                break;
            }
            const std::uint64_t copy_begin =
                std::max(tensor.source_begin, file_offset);
            const std::uint64_t copy_end =
                std::min(tensor.source_end, chunk_end);
            if (copy_end <= copy_begin) {
                continue;
            }
            const std::uint64_t copy_bytes = copy_end - copy_begin;
            std::uint64_t destination_offset = 0U;
            if (!checked_add(tensor.arena_offset,
                             copy_begin - tensor.source_begin,
                             destination_offset) ||
                destination_offset > plan.arena_bytes ||
                copy_bytes > plan.arena_bytes - destination_offset ||
                outcome.stats.memcpy_operations >=
                    options.max_memcpy_operations) {
                outcome.diagnostic = make_diagnostic(
                    outcome.stats.memcpy_operations >=
                            options.max_memcpy_operations
                        ? ResidentLoadErrorCode::kInvalidOption
                        : ResidentLoadErrorCode::kArithmeticOverflow,
                    "scatter copy exceeds arena or operation limit",
                    tensor.name,
                    shard.identity.filename,
                    copy_begin);
                return outcome;
            }
            const std::size_t source_offset = static_cast<std::size_t>(
                copy_begin - file_offset);
            const cudaError_t cuda_status = cudaMemcpyAsync(
                arena_bytes + static_cast<std::size_t>(destination_offset),
                static_cast<std::uint8_t*>(resources.staging[slot]) +
                    source_offset,
                static_cast<std::size_t>(copy_bytes),
                cudaMemcpyHostToDevice,
                resources.stream);
            if (cuda_status != cudaSuccess) {
                outcome.diagnostic = cuda_diagnostic(
                    cuda_status,
                    "cudaMemcpyAsync(scatter)",
                    shard.identity.filename,
                    copy_begin);
                return outcome;
            }
            if (!add_stat(outcome.stats.bytes_copied, copy_bytes) ||
                !add_stat(outcome.stats.memcpy_operations, 1U)) {
                outcome.diagnostic = make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "scatter-copy statistics overflow uint64",
                    tensor.name,
                    shard.identity.filename,
                    copy_begin);
                return outcome;
            }
        }

        const cudaError_t cuda_status =
            cudaEventRecord(resources.events[slot], resources.stream);
        if (cuda_status != cudaSuccess) {
            outcome.diagnostic = cuda_diagnostic(
                cuda_status,
                "cudaEventRecord(staging lifetime)",
                shard.identity.filename,
                file_offset);
            return outcome;
        }
        slot_pending[slot] = true;
        file_offset = chunk_end;
        ++chunk_index;
    }

    std::uint8_t extra_byte = 0U;
    const ReadChunkResult extra = read_exact_chunk(
        opened_shard.descriptor.get(), &extra_byte, 1U);
    if (extra.error != 0) {
        outcome.diagnostic = make_diagnostic(
            ResidentLoadErrorCode::kIoFailure,
            "failed while checking shard EOF",
            "read",
            shard.identity.filename,
            shard.identity.file_size,
            {},
            std::strerror(extra.error),
            extra.error);
        return outcome;
    }
    if (extra.bytes != 0U || !extra.eof) {
        outcome.diagnostic = make_diagnostic(
            ResidentLoadErrorCode::kShardSizeMismatch,
            "shard grew during sequential load",
            "read",
            shard.identity.filename,
            shard.identity.file_size);
        return outcome;
    }
    struct stat final_status {};
    if (::fstat(opened_shard.descriptor.get(), &final_status) != 0) {
        const int error = errno;
        outcome.diagnostic = make_diagnostic(
            ResidentLoadErrorCode::kIoFailure,
            "final fstat failed for opened shard",
            "fstat",
            shard.identity.filename,
            shard.identity.file_size,
            {},
            std::strerror(error),
            error);
        return outcome;
    }
    if (!S_ISREG(final_status.st_mode) || final_status.st_size < 0 ||
        static_cast<std::uint64_t>(final_status.st_size) !=
            shard.identity.file_size) {
        outcome.diagnostic = make_diagnostic(
            ResidentLoadErrorCode::kShardSizeMismatch,
            "shard identity changed during sequential load",
            "fstat",
            shard.identity.filename,
            shard.identity.file_size);
        return outcome;
    }

    HashOperationResult hash_result = hasher.finalize();
    if (!hash_result.digest) {
        outcome.diagnostic = make_diagnostic(
            ResidentLoadErrorCode::kSha256Failure,
            "SHA-256 finalization failed",
            hash_result.context.empty()
                ? std::string(to_string(hash_backend)) + ".finalize"
                : hash_result.context,
            shard.identity.filename,
            shard.identity.file_size,
            {},
            std::strerror(hash_result.error),
            hash_result.error);
        return outcome;
    }
    outcome.stats.sha256 = hash_result.digest->hex();
    if (outcome.stats.sha256 != shard.identity.sha256) {
        outcome.diagnostic = make_diagnostic(
            ResidentLoadErrorCode::kSha256Mismatch,
            "full-file SHA-256 differs from authenticated identity",
            "sha256",
            shard.identity.filename,
            0U,
            shard.identity.sha256,
            outcome.stats.sha256);
        return outcome;
    }

    const cudaError_t cuda_status = cudaStreamSynchronize(resources.stream);
    if (cuda_status != cudaSuccess) {
        outcome.diagnostic = cuda_diagnostic(
            cuda_status,
            "cudaStreamSynchronize(shard)",
            shard.identity.filename,
            shard.identity.file_size);
    }
    return outcome;
}

ResidentLoadDiagnostic worker_exception_diagnostic(
    const std::exception_ptr& exception,
    std::string shard,
    const std::uint64_t index) {
    try {
        std::rethrow_exception(exception);
    } catch (const std::bad_alloc&) {
        return make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "allocation failed in resident loader worker",
            "worker",
            std::move(shard),
            index);
    } catch (const std::length_error&) {
        return make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "container size exceeded in resident loader worker",
            "worker",
            std::move(shard),
            index);
    } catch (const std::exception& error) {
        return make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "unexpected exception in resident loader worker",
            "worker",
            std::move(shard),
            index,
            {},
            error.what());
    } catch (...) {
        return make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "unknown exception in resident loader worker",
            "worker",
            std::move(shard),
            index);
    }
}

ResidentLoadDiagnostic from_manifest_failure(
    const mw::ManifestResult& manifest_result) {
    if (manifest_result.diagnostics.empty()) {
        return make_diagnostic(ResidentLoadErrorCode::kInvalidManifest,
                               "pinned weight manifest validation failed",
                               "manifest");
    }
    const mw::ManifestDiagnostic& source = manifest_result.diagnostics.front();
    return make_diagnostic(ResidentLoadErrorCode::kInvalidManifest,
                           source.message,
                           source.context,
                           {},
                           0U,
                           source.expected,
                           source.actual);
}

}  // namespace

const std::vector<ShardIdentity>& pinned_qwen36_27b_shards() noexcept {
    static const std::vector<ShardIdentity> identities = {
        {"model-00001-of-00003.safetensors",
         9'965'652'512ULL,
         "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d"},
        {"model-00002-of-00003.safetensors",
         9'985'757'032ULL,
         "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d"},
        {"model-00003-of-00003.safetensors",
         1'970'287'640ULL,
         "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845"},
    };
    return identities;
}

PlanResult build_resident_load_plan(
    const mw::WeightManifest& manifest,
    const std::vector<ShardIdentity>& identities,
    const ResidentPayload payload) {
    try {
        if (!is_valid_resident_payload(payload)) {
            return plan_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidOption,
                "resident payload selection is invalid",
                "payload"));
        }
        if (identities.empty()) {
            return plan_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidShardIdentity,
                "at least one shard identity is required",
                "identities"));
        }

        std::map<std::string, ShardIdentity, std::less<>> identity_map;
        std::uint64_t source_bytes = 0U;
        for (const ShardIdentity& identity : identities) {
            if (!st::is_safe_relative_shard_path(identity.filename)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kUnsafeShardPath,
                    "shard identity has an unsafe relative path",
                    identity.filename,
                    identity.filename));
            }
            if (identity.file_size == 0U ||
                !is_lower_hex_digest(identity.sha256)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kInvalidShardIdentity,
                    "shard identity requires a non-zero size and lowercase SHA-256",
                    identity.filename,
                    identity.filename));
            }
            if (!identity_map.emplace(identity.filename, identity).second) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kDuplicateShard,
                    "duplicate shard identity",
                    identity.filename,
                    identity.filename));
            }
            if (!checked_add(source_bytes, identity.file_size, source_bytes)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "aggregate shard size overflows uint64",
                    identity.filename,
                    identity.filename));
            }
        }

        struct SourceRange {
            std::string name;
            std::uint64_t begin = 0U;
            std::uint64_t end = 0U;
        };
        std::map<std::string, std::vector<SourceRange>, std::less<>> ranges;
        std::set<std::string, std::less<>> manifest_shards;
        struct Candidate {
            const std::string* name = nullptr;
            const mw::TensorLocator* locator = nullptr;
        };
        std::vector<Candidate> selected_tensors;
        selected_tensors.reserve(manifest.tensors.size());
        const mw::TensorCategory selected_category =
            payload == ResidentPayload::kText
                ? mw::TensorCategory::kText
                : mw::TensorCategory::kMtp;

        for (const auto& item : manifest.tensors) {
            const std::string& name = item.first;
            const mw::TensorLocator& locator = item.second;
            if (!st::is_safe_relative_shard_path(locator.shard)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kUnsafeShardPath,
                    "manifest tensor has an unsafe shard path",
                    name,
                    locator.shard));
            }
            const auto identity = identity_map.find(locator.shard);
            if (identity == identity_map.end()) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kMissingShardIdentity,
                    "manifest tensor has no authenticated shard identity",
                    name,
                    locator.shard));
            }
            manifest_shards.emplace(locator.shard);

            const mw::TensorCategory classified = mw::classify_tensor(name);
            if (locator.file_end <= locator.file_begin) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "tensor source end does not follow source begin (possible offset overflow)",
                    name,
                    locator.shard,
                    locator.file_begin,
                    "file_end > file_begin",
                    std::to_string(locator.file_end)));
            }
            if (classified == mw::TensorCategory::kUnknown ||
                locator.category != classified || locator.byte_size == 0U ||
                locator.file_end - locator.file_begin != locator.byte_size ||
                locator.file_end > identity->second.file_size ||
                !tensor_bytes_match(locator)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kInvalidManifest,
                    "tensor locator is inconsistent with its ABI or shard range",
                    name,
                    locator.shard,
                    locator.file_begin));
            }

            ranges[locator.shard].push_back(
                SourceRange{name, locator.file_begin, locator.file_end});
            if (classified == selected_category) {
                selected_tensors.push_back(Candidate{&name, &locator});
            }
        }

        for (const auto& identity : identity_map) {
            if (manifest_shards.find(identity.first) == manifest_shards.end()) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kUnexpectedShardIdentity,
                    "authenticated shard has no tensor in the manifest",
                    identity.first,
                    identity.first));
            }
        }
        if (manifest.summary.shard_count != 0U &&
            manifest.summary.shard_count != identity_map.size()) {
            return plan_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidManifest,
                "manifest shard count disagrees with authenticated identities",
                "summary.shard_count",
                {},
                0U,
                std::to_string(identity_map.size()),
                std::to_string(manifest.summary.shard_count)));
        }

        for (auto& shard_ranges : ranges) {
            std::sort(shard_ranges.second.begin(),
                      shard_ranges.second.end(),
                      [](const SourceRange& left, const SourceRange& right) {
                          if (left.begin != right.begin) {
                              return left.begin < right.begin;
                          }
                          if (left.end != right.end) {
                              return left.end < right.end;
                          }
                          return left.name < right.name;
                      });
            for (std::size_t index = 1U; index < shard_ranges.second.size();
                 ++index) {
                if (shard_ranges.second[index].begin <
                    shard_ranges.second[index - 1U].end) {
                    return plan_failure(make_diagnostic(
                        ResidentLoadErrorCode::kInvalidManifest,
                        "tensor source ranges overlap",
                        shard_ranges.second[index].name,
                        shard_ranges.first,
                        shard_ranges.second[index].begin,
                        shard_ranges.second[index - 1U].name));
                }
            }
        }

        std::sort(selected_tensors.begin(),
                  selected_tensors.end(),
                  [](const Candidate& left, const Candidate& right) {
                      if (left.locator->shard != right.locator->shard) {
                          return left.locator->shard < right.locator->shard;
                      }
                      if (left.locator->file_begin != right.locator->file_begin) {
                          return left.locator->file_begin <
                                 right.locator->file_begin;
                      }
                      return *left.name < *right.name;
                  });

        ResidentLoadPlan plan;
        plan.source_bytes = source_bytes;
        plan.tensors.reserve(selected_tensors.size());
        plan.shards.reserve(identity_map.size());
        std::map<std::string, std::size_t, std::less<>> shard_indices;
        for (const auto& identity : identity_map) {
            shard_indices.emplace(identity.first, plan.shards.size());
            PlannedShard shard;
            shard.identity = identity.second;
            plan.shards.emplace_back(std::move(shard));
        }

        for (const Candidate& candidate : selected_tensors) {
            std::uint64_t allocation_bytes = 0U;
            if (!checked_align_up(candidate.locator->byte_size,
                                  kResidentTensorAlignment,
                                  allocation_bytes) ||
                !checked_add(plan.arena_bytes,
                             allocation_bytes,
                             plan.arena_bytes) ||
                !checked_add(plan.copied_bytes,
                             candidate.locator->byte_size,
                             plan.copied_bytes)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "resident arena or copied-byte total overflows uint64",
                    *candidate.name,
                    candidate.locator->shard,
                    candidate.locator->file_begin));
            }
            PlannedTensor tensor;
            tensor.name = *candidate.name;
            tensor.shard = candidate.locator->shard;
            tensor.source_begin = candidate.locator->file_begin;
            tensor.source_end = candidate.locator->file_end;
            tensor.arena_offset = plan.arena_bytes - allocation_bytes;
            tensor.byte_size = candidate.locator->byte_size;
            tensor.dtype = candidate.locator->dtype;
            tensor.shape = candidate.locator->shape;
            const std::size_t tensor_index = plan.tensors.size();
            plan.tensors.emplace_back(std::move(tensor));

            PlannedShard& shard =
                plan.shards[shard_indices.at(candidate.locator->shard)];
            shard.tensor_indices.push_back(tensor_index);
            if (!checked_add(shard.copied_bytes,
                             candidate.locator->byte_size,
                             shard.copied_bytes)) {
                return plan_failure(make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "per-shard copied-byte total overflows uint64",
                    *candidate.name,
                    candidate.locator->shard));
            }
        }

        const std::size_t summary_tensor_count =
            payload == ResidentPayload::kText
                ? manifest.summary.text_tensor_count
                : manifest.summary.mtp_tensor_count;
        const std::uint64_t summary_payload_bytes =
            payload == ResidentPayload::kText
                ? manifest.summary.raw_text_bytes
                : manifest.summary.mtp_bytes;
        const std::uint64_t summary_arena_bytes =
            payload == ResidentPayload::kText
                ? manifest.summary.estimated_text_arena_bytes
                : 0U;
        if (plan.tensors.empty() ||
            (summary_tensor_count != 0U &&
             summary_tensor_count != plan.tensors.size()) ||
            (summary_payload_bytes != 0U &&
             summary_payload_bytes != plan.copied_bytes) ||
            (manifest.summary.arena_alignment != 0U &&
             manifest.summary.arena_alignment != kResidentTensorAlignment) ||
            (summary_arena_bytes != 0U &&
             summary_arena_bytes != plan.arena_bytes)) {
            return plan_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidManifest,
                "manifest summary disagrees with the deterministic resident plan",
                "manifest.summary"));
        }

        PlanResult result;
        result.value.emplace(std::move(plan));
        return result;
    } catch (const std::bad_alloc&) {
        return plan_failure(make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "allocation failed while building resident load plan",
            "plan"));
    } catch (const std::length_error&) {
        return plan_failure(make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "container size exceeded while building resident load plan",
            "plan"));
    }
}

ResidentWeights::~ResidentWeights() { release(); }

ResidentWeights::ResidentWeights(ResidentWeights&& other) noexcept
    : arena_(std::exchange(other.arena_, nullptr)),
      arena_bytes_(std::exchange(other.arena_bytes_, 0U)),
      tensors_(std::move(other.tensors_)),
      stats_(std::move(other.stats_)) {}

ResidentWeights& ResidentWeights::operator=(ResidentWeights&& other) noexcept {
    if (this != &other) {
        release();
        arena_ = std::exchange(other.arena_, nullptr);
        arena_bytes_ = std::exchange(other.arena_bytes_, 0U);
        tensors_ = std::move(other.tensors_);
        stats_ = std::move(other.stats_);
    }
    return *this;
}

void ResidentWeights::release() noexcept {
    if (arena_ != nullptr) {
        (void)cudaFree(arena_);
        arena_ = nullptr;
        (void)cudaGetLastError();
    }
    arena_bytes_ = 0U;
    tensors_.clear();
    stats_ = {};
}

const DeviceTensorView* ResidentWeights::find(
    const std::string_view name) const noexcept {
    const auto iterator = tensors_.find(name);
    return iterator == tensors_.end() ? nullptr : &iterator->second;
}

ResidentLoadResult load_resident_weights(
    const std::filesystem::path& directory,
    const mw::WeightManifest& manifest,
    const std::vector<ShardIdentity>& identities,
    const ResidentLoadOptions& options) {
    if (options.chunk_bytes == 0U ||
        options.chunk_bytes > kMaximumChunkBytes ||
        options.chunk_bytes >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        options.max_arena_bytes == 0U ||
        options.max_arena_bytes > kMaximumArenaBytes ||
        options.max_tensors == 0U || options.max_tensors > kMaximumTensors ||
        options.max_shards == 0U || options.max_shards > kMaximumShards ||
        options.max_memcpy_operations == 0U ||
        options.max_memcpy_operations > kMaximumMemcpyOperations ||
        !is_valid_sha256_backend(options.sha256_backend) ||
        !is_valid_resident_payload(options.payload) ||
        options.max_parallel_shards == 0U ||
        options.max_parallel_shards > kMaximumParallelShards) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kInvalidOption,
            "resident loader options exceed defensive limits",
            "options"));
    }

    try {
        PlanResult plan_result = build_resident_load_plan(
            manifest, identities, options.payload);
        if (!plan_result) {
            return load_failure(std::move(plan_result.diagnostic));
        }
        ResidentLoadPlan& plan = *plan_result.value;
        if (plan.arena_bytes > options.max_arena_bytes ||
            plan.arena_bytes >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            plan.tensors.size() > options.max_tensors ||
            plan.shards.size() > options.max_shards) {
            return load_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidOption,
                "resident plan exceeds configured resource limits",
                "plan"));
        }

        const std::size_t worker_count = std::min(
            options.max_parallel_shards, plan.shards.size());
        std::uint64_t staging_slots = 0U;
        std::uint64_t pinned_staging_bytes = 0U;
        if (!checked_multiply(static_cast<std::uint64_t>(worker_count),
                              static_cast<std::uint64_t>(kStagingSlots),
                              staging_slots) ||
            !checked_multiply(staging_slots,
                              options.chunk_bytes,
                              pinned_staging_bytes) ||
            pinned_staging_bytes > kMaximumPinnedStagingBytes) {
            return load_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidOption,
                "parallel staging allocation exceeds defensive limit",
                "options.max_parallel_shards",
                {},
                0U,
                std::to_string(kMaximumPinnedStagingBytes),
                std::to_string(pinned_staging_bytes)));
        }

        std::uint64_t planned_memcpy_operations = 0U;
        for (const PlannedTensor& tensor : plan.tensors) {
            const std::uint64_t first_chunk =
                tensor.source_begin / options.chunk_bytes;
            const std::uint64_t last_chunk =
                (tensor.source_end - 1U) / options.chunk_bytes;
            const std::uint64_t tensor_operations =
                last_chunk - first_chunk + 1U;
            if (!checked_add(planned_memcpy_operations,
                             tensor_operations,
                             planned_memcpy_operations)) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "planned scatter-copy count overflows uint64",
                    tensor.name,
                    tensor.shard,
                    tensor.source_begin));
            }
            if (planned_memcpy_operations >
                options.max_memcpy_operations) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kInvalidOption,
                    "resident plan exceeds scatter-copy operation limit",
                    tensor.name,
                    tensor.shard,
                    tensor.source_begin,
                    std::to_string(options.max_memcpy_operations),
                    std::to_string(planned_memcpy_operations)));
            }
        }

        HashBackendPreparation hash_backend =
            prepare_hash_backend(options.sha256_backend, plan.shards.size());
        if (!hash_backend.ok()) {
            return load_failure(make_diagnostic(
                ResidentLoadErrorCode::kSha256BackendUnavailable,
                "requested SHA-256 backend could not be initialized",
                hash_backend.context,
                {},
                0U,
                std::string(to_string(options.sha256_backend)),
                std::strerror(hash_backend.error),
                hash_backend.error));
        }

        const std::string directory_text = directory.string();
        UniqueFd root_descriptor(
            ::open(directory_text.c_str(),
                   O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
        if (!root_descriptor) {
            const int error = errno;
            return load_failure(make_diagnostic(
                ResidentLoadErrorCode::kOpenDirectoryFailed,
                "could not securely open checkpoint directory",
                directory_text,
                {},
                0U,
                {},
                std::strerror(error),
                error));
        }

        std::vector<OpenShard> open_shards;
        open_shards.reserve(plan.shards.size());
        for (const PlannedShard& shard : plan.shards) {
            OpenResult opened = open_relative_nofollow(
                root_descriptor.get(), shard.identity.filename);
            if (!opened.descriptor) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kOpenShardFailed,
                    "could not securely open shard relative to checkpoint root",
                    opened.component,
                    shard.identity.filename,
                    0U,
                    {},
                    std::strerror(opened.error),
                    opened.error));
            }
            struct stat status {};
            if (::fstat(opened.descriptor.get(), &status) != 0) {
                const int error = errno;
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kIoFailure,
                    "fstat failed for opened shard",
                    "fstat",
                    shard.identity.filename,
                    0U,
                    {},
                    std::strerror(error),
                    error));
            }
            if (!S_ISREG(status.st_mode)) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kShardNotRegular,
                    "opened shard is not a regular file",
                    "fstat",
                    shard.identity.filename));
            }
            if (status.st_size < 0 ||
                static_cast<std::uint64_t>(status.st_size) !=
                    shard.identity.file_size) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kShardSizeMismatch,
                    "opened shard size differs from authenticated identity",
                    "fstat",
                    shard.identity.filename,
                    0U,
                    std::to_string(shard.identity.file_size),
                    status.st_size < 0
                        ? "negative"
                        : std::to_string(
                              static_cast<std::uint64_t>(status.st_size))));
            }
            OpenShard opened_shard;
            opened_shard.plan = &shard;
            opened_shard.descriptor = std::move(opened.descriptor);
            open_shards.emplace_back(std::move(opened_shard));
        }

        ResidentWeights output;
        output.arena_bytes_ = plan.arena_bytes;
        output.stats_.sha256_backend = hash_backend.backend;
        output.stats_.shard_workers = worker_count;
        output.stats_.pinned_staging_bytes = pinned_staging_bytes;
        output.stats_.shards.reserve(plan.shards.size());
        for (const PlannedTensor& tensor : plan.tensors) {
            DeviceTensorView view;
            view.arena_offset = tensor.arena_offset;
            view.byte_size = tensor.byte_size;
            view.dtype = tensor.dtype;
            view.shape = tensor.shape;
            if (!output.tensors_.emplace(tensor.name, std::move(view)).second) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kInvalidManifest,
                    "duplicate text tensor in resident plan",
                    tensor.name,
                    tensor.shard));
            }
        }

        // Isolate any prior caller error before interpreting CUDA return codes.
        (void)cudaGetLastError();
        int cuda_device = -1;
        cudaError_t cuda_status = cudaGetDevice(&cuda_device);
        if (cuda_status != cudaSuccess) {
            return load_failure(
                cuda_diagnostic(cuda_status, "cudaGetDevice"));
        }

        // Declaration order is intentional: worker streams and staging buffers
        // are synchronized and released before the arena on every exit path.
        CudaArena arena;
        std::vector<std::unique_ptr<CudaWorkerResources>> worker_resources;
        worker_resources.reserve(worker_count);
        for (std::size_t worker_index = 0U;
             worker_index < worker_count; ++worker_index) {
            auto resources = std::make_unique<CudaWorkerResources>();
            resources->device = cuda_device;
            cuda_status = cudaStreamCreateWithFlags(
                &resources->stream, cudaStreamNonBlocking);
            if (cuda_status != cudaSuccess) {
                return load_failure(cuda_diagnostic(
                    cuda_status,
                    "cudaStreamCreateWithFlags(worker " +
                        std::to_string(worker_index) + ")"));
            }
            for (std::size_t slot = 0U; slot < kStagingSlots; ++slot) {
                cuda_status = cudaEventCreateWithFlags(
                    &resources->events[slot], cudaEventDisableTiming);
                if (cuda_status != cudaSuccess) {
                    return load_failure(cuda_diagnostic(
                        cuda_status,
                        "cudaEventCreateWithFlags(worker " +
                            std::to_string(worker_index) + ")"));
                }
                cuda_status = cudaHostAlloc(
                    &resources->staging[slot],
                    static_cast<std::size_t>(options.chunk_bytes),
                    cudaHostAllocDefault);
                if (cuda_status != cudaSuccess) {
                    return load_failure(cuda_diagnostic(
                        cuda_status,
                        "cudaHostAlloc(worker " +
                            std::to_string(worker_index) + ")"));
                }
            }
            worker_resources.emplace_back(std::move(resources));
        }

        std::size_t free_bytes = 0U;
        std::size_t total_bytes = 0U;
        cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (cuda_status != cudaSuccess) {
            return load_failure(
                cuda_diagnostic(cuda_status, "cudaMemGetInfo"));
        }
        output.stats_.device_free_before =
            static_cast<std::uint64_t>(free_bytes);
        output.stats_.device_total = static_cast<std::uint64_t>(total_bytes);
        const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
        if (plan.arena_bytes > free_u64 ||
            options.min_free_bytes_after_load > free_u64 - plan.arena_bytes) {
            return load_failure(make_diagnostic(
                ResidentLoadErrorCode::kInsufficientDeviceMemory,
                "cudaMemGetInfo cannot satisfy arena plus configured safety margin",
                "cudaMemGetInfo",
                {},
                0U,
                std::to_string(plan.arena_bytes) + "+" +
                    std::to_string(options.min_free_bytes_after_load),
                std::to_string(free_u64)));
        }

        cuda_status = cudaMalloc(
            &arena.data, static_cast<std::size_t>(plan.arena_bytes));
        if (cuda_status != cudaSuccess) {
            return load_failure(
                cuda_diagnostic(cuda_status, "cudaMalloc(resident arena)"));
        }
        auto* const arena_bytes = static_cast<std::uint8_t*>(arena.data);
        for (auto& tensor : output.tensors_) {
            tensor.second.device_data =
                arena_bytes + static_cast<std::size_t>(tensor.second.arena_offset);
        }

        std::vector<ShardLoadOutcome> shard_outcomes(plan.shards.size());
        std::vector<std::exception_ptr> shard_exceptions(plan.shards.size());
        std::vector<ResidentLoadDiagnostic> worker_diagnostics(worker_count);
        std::vector<std::exception_ptr> worker_exceptions(worker_count);
        std::atomic<std::size_t> next_shard{0U};

        // Every thread owns one CUDA stream and two staging slots. Atomic shard
        // assignment keeps the pool fixed while allowing a worker that finishes
        // a smaller shard to claim more work.
        JoiningThreads threads;
        threads.reserve(worker_count);
        for (std::size_t worker_index = 0U;
             worker_index < worker_count; ++worker_index) {
            CudaWorkerResources* const resources =
                worker_resources[worker_index].get();
            threads.emplace_back([&, worker_index, resources]() {
                try {
                    const cudaError_t set_device_status =
                        cudaSetDevice(cuda_device);
                    if (set_device_status != cudaSuccess) {
                        worker_diagnostics[worker_index] = cuda_diagnostic(
                            set_device_status,
                            "cudaSetDevice(worker " +
                                std::to_string(worker_index) + ")");
                        return;
                    }
                    (void)cudaGetLastError();
                } catch (...) {
                    worker_exceptions[worker_index] =
                        std::current_exception();
                    return;
                }

                while (true) {
                    const std::size_t shard_index =
                        next_shard.fetch_add(1U, std::memory_order_relaxed);
                    if (shard_index >= open_shards.size()) {
                        return;
                    }
                    try {
                        UniqueFd af_alg_operation;
                        if (hash_backend.backend ==
                            ResidentSha256Backend::kLinuxAfAlg) {
                            af_alg_operation = std::move(
                                hash_backend.af_alg_operations[shard_index]);
                        }
                        shard_outcomes[shard_index] = load_one_shard(
                            open_shards[shard_index],
                            plan,
                            options,
                            hash_backend.backend,
                            std::move(af_alg_operation),
                            arena_bytes,
                            *resources);
                    } catch (...) {
                        shard_exceptions[shard_index] =
                            std::current_exception();
                        return;
                    }

                    // A failed shard may have outstanding async H2D work. Stop
                    // this worker so its staging memory is not reused; cleanup
                    // synchronizes the stream after all threads are joined.
                    if (!shard_outcomes[shard_index].ok()) {
                        return;
                    }
                }
            });
        }
        threads.join_all();

        for (std::size_t worker_index = 0U;
             worker_index < worker_count; ++worker_index) {
            if (worker_exceptions[worker_index]) {
                return load_failure(worker_exception_diagnostic(
                    worker_exceptions[worker_index],
                    {},
                    worker_index));
            }
            if (worker_diagnostics[worker_index].code !=
                ResidentLoadErrorCode::kNone) {
                return load_failure(
                    std::move(worker_diagnostics[worker_index]));
            }
        }

        for (std::size_t shard_index = 0U;
             shard_index < shard_outcomes.size(); ++shard_index) {
            if (shard_exceptions[shard_index]) {
                return load_failure(worker_exception_diagnostic(
                    shard_exceptions[shard_index],
                    plan.shards[shard_index].identity.filename,
                    shard_index));
            }
            if (!shard_outcomes[shard_index].attempted) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kAllocationFailure,
                    "resident loader worker did not process assigned shard",
                    "worker",
                    plan.shards[shard_index].identity.filename,
                    shard_index));
            }
            if (!shard_outcomes[shard_index].ok()) {
                return load_failure(
                    std::move(shard_outcomes[shard_index].diagnostic));
            }
        }

        for (ShardLoadOutcome& shard_outcome : shard_outcomes) {
            if (!add_stat(output.stats_.bytes_read,
                          shard_outcome.stats.bytes_read) ||
                !add_stat(output.stats_.bytes_copied,
                          shard_outcome.stats.bytes_copied) ||
                !add_stat(output.stats_.chunks,
                          shard_outcome.stats.chunks) ||
                !add_stat(output.stats_.memcpy_operations,
                          shard_outcome.stats.memcpy_operations)) {
                return load_failure(make_diagnostic(
                    ResidentLoadErrorCode::kArithmeticOverflow,
                    "aggregate loader statistics overflow uint64",
                    "statistics",
                    shard_outcome.stats.filename));
            }
            output.stats_.shards.emplace_back(
                std::move(shard_outcome.stats));
        }

        if (output.stats_.bytes_read != plan.source_bytes ||
            output.stats_.bytes_copied != plan.copied_bytes ||
            output.stats_.memcpy_operations != planned_memcpy_operations ||
            output.stats_.bytes_copied > output.stats_.bytes_read) {
            return load_failure(make_diagnostic(
                ResidentLoadErrorCode::kInvalidManifest,
                "completed loader statistics disagree with resident plan",
                "statistics",
                {},
                0U,
                std::to_string(plan.source_bytes) + "/" +
                    std::to_string(plan.copied_bytes),
                std::to_string(output.stats_.bytes_read) + "/" +
                    std::to_string(output.stats_.bytes_copied)));
        }
        output.stats_.bytes_skipped =
            output.stats_.bytes_read - output.stats_.bytes_copied;

        output.arena_ = arena.release();
        (void)cudaGetLastError();
        ResidentLoadResult result;
        result.value.emplace(std::move(output));
        return result;
    } catch (const std::bad_alloc&) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "allocation failed during resident weight load",
            "loader"));
    } catch (const std::length_error&) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "container size exceeded during resident weight load",
            "loader"));
    } catch (const std::filesystem::filesystem_error& error) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kIoFailure,
            "filesystem conversion failed during resident weight load",
            directory.string(),
            {},
            0U,
            {},
            error.what()));
    } catch (const std::system_error& error) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kAllocationFailure,
            "could not create or join resident loader worker",
            "worker threads",
            {},
            0U,
            {},
            error.what(),
            error.code().value()));
    }
}

ResidentLoadResult load_pinned_qwen36_27b(
    const std::filesystem::path& directory,
    const ResidentLoadOptions& options) {
    mw::ManifestResult manifest =
        mw::build_qwen36_27b_text_manifest(directory);
    if (!manifest) {
        return load_failure(from_manifest_failure(manifest));
    }
    if (manifest.value->summary.estimated_text_arena_bytes !=
            kPinnedQwen36_27BArenaBytes ||
        manifest.value->summary.raw_text_bytes !=
            mw::kPinnedQwen36_27BTextBytes) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kInvalidManifest,
            "pinned Qwen3.6-27B arena contract does not match compiled identity",
            "manifest.summary",
            {},
            0U,
            std::to_string(kPinnedQwen36_27BArenaBytes),
            std::to_string(
                manifest.value->summary.estimated_text_arena_bytes)));
    }
    ResidentLoadOptions text_options = options;
    text_options.payload = ResidentPayload::kText;
    return load_resident_weights(directory, *manifest.value,
                                 pinned_qwen36_27b_shards(), text_options);
}

ResidentLoadResult load_pinned_qwen36_27b_mtp(
    const std::filesystem::path& directory,
    const ResidentLoadOptions& options) {
    mw::ManifestResult manifest =
        mw::build_qwen36_27b_text_manifest(directory);
    if (!manifest) {
        return load_failure(from_manifest_failure(manifest));
    }

    PlanResult plan = build_resident_load_plan(
        *manifest.value, pinned_qwen36_27b_shards(), ResidentPayload::kMtp);
    if (!plan) {
        return load_failure(std::move(plan.diagnostic));
    }
    if (plan.value->arena_bytes != kPinnedQwen36_27BMtpArenaBytes ||
        plan.value->copied_bytes != mw::kPinnedQwen36_27BMtpBytes ||
        plan.value->tensors.size() != mw::kPinnedQwen36_27BMtpTensorCount) {
        return load_failure(make_diagnostic(
            ResidentLoadErrorCode::kInvalidManifest,
            "pinned Qwen3.6-27B MTP arena contract does not match compiled identity",
            "manifest.summary",
            {},
            0U,
            std::to_string(kPinnedQwen36_27BMtpArenaBytes) + " bytes / " +
                std::to_string(mw::kPinnedQwen36_27BMtpTensorCount) +
                " views",
            std::to_string(plan.value->arena_bytes) + " bytes / " +
                std::to_string(plan.value->tensors.size()) + " views"));
    }

    ResidentLoadOptions mtp_options = options;
    mtp_options.payload = ResidentPayload::kMtp;
    return load_resident_weights(directory, *manifest.value,
                                 pinned_qwen36_27b_shards(), mtp_options);
}

std::string_view to_string(ResidentLoadErrorCode code) noexcept {
    switch (code) {
        case ResidentLoadErrorCode::kNone:
            return "none";
        case ResidentLoadErrorCode::kInvalidOption:
            return "invalid_option";
        case ResidentLoadErrorCode::kInvalidManifest:
            return "invalid_manifest";
        case ResidentLoadErrorCode::kUnsafeShardPath:
            return "unsafe_shard_path";
        case ResidentLoadErrorCode::kDuplicateShard:
            return "duplicate_shard";
        case ResidentLoadErrorCode::kMissingShardIdentity:
            return "missing_shard_identity";
        case ResidentLoadErrorCode::kUnexpectedShardIdentity:
            return "unexpected_shard_identity";
        case ResidentLoadErrorCode::kInvalidShardIdentity:
            return "invalid_shard_identity";
        case ResidentLoadErrorCode::kArithmeticOverflow:
            return "arithmetic_overflow";
        case ResidentLoadErrorCode::kOpenDirectoryFailed:
            return "open_directory_failed";
        case ResidentLoadErrorCode::kOpenShardFailed:
            return "open_shard_failed";
        case ResidentLoadErrorCode::kShardNotRegular:
            return "shard_not_regular";
        case ResidentLoadErrorCode::kShardSizeMismatch:
            return "shard_size_mismatch";
        case ResidentLoadErrorCode::kIoFailure:
            return "io_failure";
        case ResidentLoadErrorCode::kSha256BackendUnavailable:
            return "sha256_backend_unavailable";
        case ResidentLoadErrorCode::kSha256Failure:
            return "sha256_failure";
        case ResidentLoadErrorCode::kSha256Mismatch:
            return "sha256_mismatch";
        case ResidentLoadErrorCode::kCudaFailure:
            return "cuda_failure";
        case ResidentLoadErrorCode::kInsufficientDeviceMemory:
            return "insufficient_device_memory";
        case ResidentLoadErrorCode::kAllocationFailure:
            return "allocation_failure";
    }
    return "unknown";
}

std::string_view to_string(const ResidentSha256Backend backend) noexcept {
    switch (backend) {
        case ResidentSha256Backend::kAuto:
            return "auto";
        case ResidentSha256Backend::kPortable:
            return "portable";
        case ResidentSha256Backend::kLinuxAfAlg:
            return "linux_af_alg";
    }
    return "unknown";
}

std::string_view to_string(const ResidentPayload payload) noexcept {
    switch (payload) {
        case ResidentPayload::kText:
            return "text";
        case ResidentPayload::kMtp:
            return "mtp";
    }
    return "unknown";
}

}  // namespace q3x::runtime
