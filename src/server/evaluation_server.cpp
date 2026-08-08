#include "q3x/server/evaluation_server.h"

#include "q3x/core/sha256.h"
#include "q3x/server/openai_protocol.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace q3x::server {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_milliseconds(
    const Clock::time_point begin, const Clock::time_point end) noexcept {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline constexpr std::size_t kMaximumHeaderBytes = 64U * 1024U;
inline constexpr std::size_t kMaximumBodyBytes = 4U * 1024U * 1024U;
// This bounded thread-per-connection ingress deliberately serves one request
// per socket. Waiting on idle HTTP keep-alive sockets would occupy every
// ingress thread and inject read-timeout-sized latency into concurrent evals.
inline constexpr std::size_t kMaximumRequestsPerConnection = 1U;
inline constexpr std::size_t kMaximumAcceptsPerPoll = 64U;

class UniqueFd final {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(const int value) noexcept : value_(value) {}
  ~UniqueFd() { reset(); }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : value_(other.release()) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return value_ >= 0;
  }
  [[nodiscard]] int release() noexcept {
    const int value = value_;
    value_ = -1;
    return value;
  }
  void reset(const int value = -1) noexcept {
    if (value_ >= 0) {
      (void)::close(value_);
    }
    value_ = value;
  }

 private:
  int value_ = -1;
};

template <typename T>
class BoundedQueue final {
 public:
  explicit BoundedQueue(const std::size_t capacity) : capacity_(capacity) {}

  [[nodiscard]] bool try_push(const T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || queue_.size() >= capacity_) {
      return false;
    }
    queue_.push_back(value);
    condition_.notify_one();
    return true;
  }

  [[nodiscard]] bool try_push(T&& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || queue_.size() >= capacity_) {
      return false;
    }
    queue_.push_back(std::move(value));
    condition_.notify_one();
    return true;
  }

  [[nodiscard]] bool pop(T& output) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return false;
    }
    output = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    condition_.notify_all();
  }

 private:
  const std::size_t capacity_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<T> queue_;
  bool closed_ = false;
};

struct HttpRequest {
  std::string method;
  std::string target;
  std::string content_type;
  std::string body;
  bool keep_alive = true;
};

struct HttpReadResult {
  std::optional<HttpRequest> value;
  int error_status = 0;
  std::string error_message;
  bool peer_closed = false;
};

struct InferenceJob {
  enum class StreamStartState : std::uint8_t {
    kPending,
    kSuccess,
    kEarlyError,
  };

  OpenAIRequest request;
  std::string id;
  std::string request_body_sha256;
  std::int64_t created = 0;
  Clock::time_point request_received_at;
  Clock::time_point admitted_at;
  Clock::time_point queued_at;
  std::atomic<bool> cancelled{false};
  std::mutex mutex;
  std::condition_variable condition;
  std::deque<std::string> events;
  std::size_t event_capacity = 0U;
  bool event_stream_closed = false;
  StreamStartState stream_start = StreamStartState::kPending;
  bool response_ready = false;
  int response_status = 500;
  std::string response_body;

  [[nodiscard]] bool push_event(std::string event,
                                const std::atomic<bool>& stopping) {
    std::unique_lock<std::mutex> lock(mutex);
    while (!cancelled.load(std::memory_order_relaxed) &&
           !stopping.load(std::memory_order_relaxed) &&
           events.size() >= event_capacity) {
      condition.wait_for(lock, std::chrono::milliseconds(100));
    }
    if (cancelled.load(std::memory_order_relaxed) ||
        stopping.load(std::memory_order_relaxed) || event_stream_closed) {
      return false;
    }
    events.push_back(std::move(event));
    if (stream_start == StreamStartState::kPending) {
      stream_start = StreamStartState::kSuccess;
    }
    condition.notify_all();
    return true;
  }

  void close_events() {
    std::lock_guard<std::mutex> lock(mutex);
    event_stream_closed = true;
    condition.notify_all();
  }

  void set_response(const int status, std::string body) {
    std::lock_guard<std::mutex> lock(mutex);
    response_status = status;
    response_body = std::move(body);
    response_ready = true;
    condition.notify_all();
  }
};

[[nodiscard]] std::string lowercase(std::string text) {
  for (char& character : text) {
    if (character >= 'A' && character <= 'Z') {
      character = static_cast<char>(character - 'A' + 'a');
    }
  }
  return text;
}

[[nodiscard]] std::string_view trim_ascii(std::string_view value) noexcept {
  while (!value.empty() &&
         (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1U);
  }
  while (!value.empty() &&
         (value.back() == ' ' || value.back() == '\t')) {
    value.remove_suffix(1U);
  }
  return value;
}

[[nodiscard]] bool is_json_content_type(
    const std::string_view value) noexcept {
  const std::size_t semicolon = value.find(';');
  if (trim_ascii(value.substr(0U, semicolon)) != "application/json") {
    return false;
  }
  if (semicolon == std::string_view::npos) {
    return true;
  }
  const std::string_view parameter = trim_ascii(value.substr(semicolon + 1U));
  return parameter == "charset=utf-8" || parameter == "charset=\"utf-8\"";
}

[[nodiscard]] bool parse_decimal_size(const std::string_view text,
                                      std::size_t& output) noexcept {
  if (text.empty()) {
    return false;
  }
  std::size_t value = 0U;
  for (const char character : text) {
    if (character < '0' || character > '9') {
      return false;
    }
    const std::size_t digit = static_cast<std::size_t>(character - '0');
    if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10U) {
      return false;
    }
    value = value * 10U + digit;
  }
  output = value;
  return true;
}

[[nodiscard]] bool wait_for_fd(const int fd, const short events,
                               const Clock::time_point deadline,
                               const std::atomic<bool>* const stopping) {
  while (Clock::now() < deadline) {
    if (stopping != nullptr && stopping->load(std::memory_order_relaxed)) {
      return false;
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - Clock::now());
    const int timeout = static_cast<int>(
        std::max<std::int64_t>(1, std::min<std::int64_t>(250,
                                                        remaining.count())));
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = events;
    const int status = ::poll(&descriptor, 1U, timeout);
    if (status > 0) {
      if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return false;
      }
#ifdef POLLRDHUP
      if ((descriptor.revents & POLLRDHUP) != 0) {
        return false;
      }
#endif
      if ((descriptor.revents & events) != 0) {
        return true;
      }
    } else if (status < 0 && errno != EINTR) {
      return false;
    }
  }
  return false;
}

[[nodiscard]] bool send_all(const int fd, const std::string_view bytes,
                            const std::uint32_t timeout_milliseconds,
                            const std::atomic<bool>* const stopping = nullptr) {
  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(timeout_milliseconds);
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    if (!wait_for_fd(fd, POLLOUT, deadline, stopping)) {
      return false;
    }
    const ssize_t sent = ::send(fd, bytes.data() + offset,
                                bytes.size() - offset, MSG_NOSIGNAL);
    if (sent > 0) {
      offset += static_cast<std::size_t>(sent);
    } else if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK ||
                           errno == EINTR)) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string_view status_text(const int status) noexcept {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 411:
      return "Length Required";
    case 413:
      return "Payload Too Large";
    case 415:
      return "Unsupported Media Type";
    case 429:
      return "Too Many Requests";
    case 500:
      return "Internal Server Error";
    case 503:
      return "Service Unavailable";
  }
  return "Error";
}

[[nodiscard]] bool send_fixed_response(
    const int fd, const int status, const std::string_view body,
    const bool keep_alive, const std::uint32_t timeout_milliseconds,
    const std::string_view extra_headers = {}) {
  std::string headers = "HTTP/1.1 " + std::to_string(status) + " ";
  headers += status_text(status);
  headers += "\r\nServer: qwen3x-orin\r\nContent-Type: application/json; "
             "charset=utf-8\r\nContent-Length: ";
  headers += std::to_string(body.size());
  headers += "\r\nConnection: ";
  headers += keep_alive ? "keep-alive" : "close";
  headers += "\r\n";
  headers.append(extra_headers.data(), extra_headers.size());
  headers += "\r\n";
  return send_all(fd, headers, timeout_milliseconds) &&
         send_all(fd, body, timeout_milliseconds);
}

[[nodiscard]] bool send_sse_headers(
    const int fd, const bool keep_alive,
    const std::uint32_t timeout_milliseconds) {
  std::string headers =
      "HTTP/1.1 200 OK\r\nServer: qwen3x-orin\r\n"
      "Content-Type: text/event-stream; charset=utf-8\r\n"
      "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
      "Transfer-Encoding: chunked\r\nConnection: ";
  headers += keep_alive ? "keep-alive" : "close";
  headers += "\r\n\r\n";
  return send_all(fd, headers, timeout_milliseconds);
}

[[nodiscard]] bool send_chunk(const int fd, const std::string_view payload,
                              const std::uint32_t timeout_milliseconds,
                              const std::atomic<bool>* const stopping) {
  std::ostringstream size;
  size << std::hex << payload.size();
  const std::string prefix = size.str() + "\r\n";
  return send_all(fd, prefix, timeout_milliseconds, stopping) &&
         send_all(fd, payload, timeout_milliseconds, stopping) &&
         send_all(fd, "\r\n", timeout_milliseconds, stopping);
}

[[nodiscard]] bool send_chunk_end(const int fd,
                                  const std::uint32_t timeout_milliseconds) {
  return send_all(fd, "0\r\n\r\n", timeout_milliseconds);
}

HttpReadResult read_http_request(
    const int fd, std::string& buffer,
    const std::uint32_t timeout_milliseconds,
    const std::atomic<bool>& stopping) {
  HttpReadResult result;
  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(timeout_milliseconds);
  std::size_t header_end = buffer.find("\r\n\r\n");
  while (header_end == std::string::npos) {
    if (buffer.size() >= kMaximumHeaderBytes) {
      result.error_status = 413;
      result.error_message = "HTTP headers exceed the fixed limit";
      return result;
    }
    if (!wait_for_fd(fd, POLLIN, deadline, &stopping)) {
      result.error_status = stopping.load(std::memory_order_relaxed) ? 503
                                                                     : 408;
      result.error_message = "request header read timed out";
      return result;
    }
    char bytes[8192];
    const ssize_t received = ::recv(fd, bytes, sizeof(bytes), 0);
    if (received == 0) {
      result.peer_closed = true;
      return result;
    }
    if (received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue;
      }
      result.peer_closed = true;
      return result;
    }
    buffer.append(bytes, static_cast<std::size_t>(received));
    header_end = buffer.find("\r\n\r\n");
  }
  if (header_end > kMaximumHeaderBytes) {
    result.error_status = 413;
    result.error_message = "HTTP headers exceed the fixed limit";
    return result;
  }

  const std::string_view header_block(buffer.data(), header_end);
  const std::size_t first_line_end = header_block.find("\r\n");
  const std::string_view request_line = header_block.substr(0U, first_line_end);
  const std::size_t first_space = request_line.find(' ');
  const std::size_t second_space =
      first_space == std::string_view::npos
          ? std::string_view::npos
          : request_line.find(' ', first_space + 1U);
  if (first_space == std::string_view::npos ||
      second_space == std::string_view::npos ||
      request_line.substr(second_space + 1U) != "HTTP/1.1") {
    result.error_status = 400;
    result.error_message = "only well-formed HTTP/1.1 requests are supported";
    return result;
  }
  HttpRequest request;
  request.method = std::string(request_line.substr(0U, first_space));
  request.target = std::string(request_line.substr(
      first_space + 1U, second_space - first_space - 1U));

  std::map<std::string, std::string, std::less<>> headers;
  std::size_t line_begin =
      first_line_end == std::string_view::npos ? header_block.size()
                                               : first_line_end + 2U;
  while (line_begin < header_block.size()) {
    const std::size_t line_end = header_block.find("\r\n", line_begin);
    const std::string_view line = header_block.substr(
        line_begin, (line_end == std::string_view::npos
                         ? header_block.size()
                         : line_end) -
                        line_begin);
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos || colon == 0U) {
      result.error_status = 400;
      result.error_message = "malformed HTTP header";
      return result;
    }
    std::string key = lowercase(std::string(line.substr(0U, colon)));
    std::string value(trim_ascii(line.substr(colon + 1U)));
    if (!headers.emplace(std::move(key), std::move(value)).second) {
      result.error_status = 400;
      result.error_message = "duplicate HTTP headers are not supported";
      return result;
    }
    if (line_end == std::string_view::npos) {
      break;
    }
    line_begin = line_end + 2U;
  }
  if (headers.find("transfer-encoding") != headers.end()) {
    result.error_status = 400;
    result.error_message = "request transfer-encoding is not supported";
    return result;
  }

  std::size_t content_length = 0U;
  const auto content_length_header = headers.find("content-length");
  if (request.method == "POST" &&
      content_length_header == headers.end()) {
    result.error_status = 411;
    result.error_message = "POST requires Content-Length";
    return result;
  }
  if (content_length_header != headers.end() &&
      (!parse_decimal_size(content_length_header->second, content_length) ||
       content_length > kMaximumBodyBytes)) {
    result.error_status = 413;
    result.error_message = "request body exceeds the fixed limit";
    return result;
  }
  const auto content_type = headers.find("content-type");
  if (content_type != headers.end()) {
    request.content_type = lowercase(content_type->second);
  }
  const auto connection = headers.find("connection");
  request.keep_alive = connection == headers.end() ||
                       lowercase(connection->second) != "close";

  const std::size_t message_bytes = header_end + 4U + content_length;
  while (buffer.size() < message_bytes) {
    if (!wait_for_fd(fd, POLLIN, deadline, &stopping)) {
      result.error_status = stopping.load(std::memory_order_relaxed) ? 503
                                                                     : 408;
      result.error_message = "request body read timed out";
      return result;
    }
    char bytes[8192];
    const ssize_t received = ::recv(fd, bytes, sizeof(bytes), 0);
    if (received == 0) {
      result.peer_closed = true;
      return result;
    }
    if (received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue;
      }
      result.peer_closed = true;
      return result;
    }
    buffer.append(bytes, static_cast<std::size_t>(received));
  }
  request.body.assign(buffer.data() + header_end + 4U, content_length);
  buffer.erase(0U, message_bytes);
  result.value.emplace(std::move(request));
  return result;
}

[[nodiscard]] bool peer_disconnected(const int fd) noexcept {
  pollfd descriptor{};
  descriptor.fd = fd;
  descriptor.events = POLLIN;
#ifdef POLLRDHUP
  descriptor.events |= POLLRDHUP;
#endif
  const int status = ::poll(&descriptor, 1U, 0);
  if (status <= 0) {
    return false;
  }
  if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
    return true;
  }
#ifdef POLLRDHUP
  if ((descriptor.revents & POLLRDHUP) != 0) {
    return true;
  }
#endif
  if ((descriptor.revents & POLLIN) != 0) {
    char byte = 0;
    return ::recv(fd, &byte, 1U, MSG_PEEK | MSG_DONTWAIT) == 0;
  }
  return false;
}

[[nodiscard]] OpenAIProtocolError simple_error(
    const int status, std::string code, std::string message) {
  OpenAIProtocolError error;
  error.http_status = status;
  error.code = std::move(code);
  error.message = std::move(message);
  return error;
}

[[nodiscard]] std::string sse_data(std::string payload) {
  return "data: " + std::move(payload) + "\n\n";
}

// Returns the complete valid UTF-8 prefix. An incomplete terminal code point
// is retained for the next token; malformed interior bytes fail closed.
[[nodiscard]] bool complete_utf8_prefix(const std::string_view bytes,
                                        std::size_t& prefix) noexcept {
  prefix = 0U;
  std::size_t index = 0U;
  while (index < bytes.size()) {
    const auto first = static_cast<unsigned char>(bytes[index]);
    std::size_t length = 0U;
    if (first <= 0x7fU) {
      length = 1U;
    } else if (first >= 0xc2U && first <= 0xdfU) {
      length = 2U;
    } else if (first >= 0xe0U && first <= 0xefU) {
      length = 3U;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      length = 4U;
    } else {
      return false;
    }
    if (bytes.size() - index < length) {
      return true;
    }
    for (std::size_t offset = 1U; offset < length; ++offset) {
      const auto continuation =
          static_cast<unsigned char>(bytes[index + offset]);
      if ((continuation & 0xc0U) != 0x80U) {
        return false;
      }
    }
    if (length == 3U) {
      const auto second = static_cast<unsigned char>(bytes[index + 1U]);
      if ((first == 0xe0U && second < 0xa0U) ||
          (first == 0xedU && second >= 0xa0U)) {
        return false;
      }
    }
    if (length == 4U) {
      const auto second = static_cast<unsigned char>(bytes[index + 1U]);
      if ((first == 0xf0U && second < 0x90U) ||
          (first == 0xf4U && second > 0x8fU)) {
        return false;
      }
    }
    index += length;
    prefix = index;
  }
  return true;
}

struct ObserverContext {
  std::shared_ptr<InferenceJob> job;
  const std::atomic<bool>* stopping = nullptr;
  std::string pending_utf8;
  bool stream = false;
  bool protocol_failure = false;
  std::string protocol_failure_message;
  std::optional<Clock::time_point> first_token_committed_at;
};

bool observe_gateway_token(
    void* const opaque,
    const runtime::ReferenceTokenEvent& token) noexcept {
  auto& context = *static_cast<ObserverContext*>(opaque);
  try {
    if (context.job->cancelled.load(std::memory_order_relaxed) ||
        context.stopping->load(std::memory_order_relaxed)) {
      return false;
    }
    if (!context.first_token_committed_at.has_value()) {
      context.first_token_committed_at = Clock::now();
    }
    if (!context.stream) {
      return true;
    }
    context.pending_utf8.append(token.text_delta.data(),
                                token.text_delta.size());
    std::size_t complete_bytes = 0U;
    if (!complete_utf8_prefix(context.pending_utf8, complete_bytes)) {
      context.protocol_failure = true;
      context.protocol_failure_message =
          "generated token bytes are not valid incremental UTF-8";
      return false;
    }
    const bool finished = token.is_stop_token ||
                          token.index + 1U ==
                              context.job->request.max_tokens;
    if (finished && complete_bytes != context.pending_utf8.size()) {
      context.protocol_failure = true;
      context.protocol_failure_message =
          "generation ended inside an incomplete UTF-8 code point";
      return false;
    }
    const std::string delta = context.pending_utf8.substr(0U, complete_bytes);
    context.pending_utf8.erase(0U, complete_bytes);
    std::optional<OpenAIFinishReason> finish_reason;
    if (token.is_stop_token) {
      finish_reason = OpenAIFinishReason::kStop;
    } else if (finished) {
      finish_reason = OpenAIFinishReason::kLength;
    }
    const std::string payload =
        context.job->request.endpoint == OpenAIEndpoint::kChatCompletions
            ? serialize_chat_chunk(
                  context.job->id, context.job->created,
                  context.job->request.model, delta, token.index == 0U,
                  finish_reason)
            : serialize_text_chunk(
                  context.job->id, context.job->created,
                  context.job->request.model, delta, finish_reason);
    if (!context.job->push_event(sse_data(payload), *context.stopping)) {
      context.job->cancelled.store(true, std::memory_order_relaxed);
      return false;
    }
    return true;
  } catch (const std::exception&) {
    context.protocol_failure = true;
    context.protocol_failure_message =
        "unexpected streaming observer failure";
    return false;
  } catch (...) {
    context.protocol_failure = true;
    context.protocol_failure_message =
        "unexpected streaming observer failure";
    return false;
  }
}

[[nodiscard]] OpenAIFinishReason finish_reason(
    const runtime::ReferenceStopReason reason) noexcept {
  return reason == runtime::ReferenceStopReason::kImEnd
             ? OpenAIFinishReason::kStop
             : OpenAIFinishReason::kLength;
}

[[nodiscard]] OpenAIUsage usage_for(
    const runtime::ReferenceGeneration& generation) noexcept {
  OpenAIUsage usage;
  usage.prompt_tokens = generation.prompt_token_ids.size();
  usage.completion_tokens = generation.generated_token_ids.size();
  usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
  return usage;
}

[[nodiscard]] std::uint64_t consumed_prompt_tokens(
    const runtime::ReferenceGeneration& generation) noexcept {
  const std::size_t comparable =
      std::min(generation.prompt_token_ids.size(), generation.steps.size());
  std::size_t consumed = 0U;
  while (consumed < comparable &&
         generation.steps[consumed].position == consumed &&
         generation.steps[consumed].input_token_id ==
             generation.prompt_token_ids[consumed]) {
    ++consumed;
  }
  return static_cast<std::uint64_t>(consumed);
}

[[nodiscard]] RequestPhaseEvidence measured_phase(
    std::string scope, const double milliseconds) {
  RequestPhaseEvidence phase;
  phase.scope = std::move(scope);
  phase.milliseconds = milliseconds;
  return phase;
}

[[nodiscard]] RequestPhaseEvidence unavailable_phase(
    std::string scope, std::string reason) {
  RequestPhaseEvidence phase;
  phase.scope = std::move(scope);
  phase.unavailable_reason = std::move(reason);
  return phase;
}

void emit_target_prefill_witness(
    const runtime::ReferenceEngine& engine,
    const std::shared_ptr<InferenceJob>& job,
    const ObserverContext& observer,
    const runtime::ReferenceGeneration& generation,
    const Clock::time_point execution_started_at,
    const Clock::time_point generation_started_at,
    const Clock::time_point generation_finished_at,
    const Clock::time_point response_ready_at) noexcept {
  try {
    TargetPrefillWitnessRecord record;
    record.request_id = job->id;
    record.request_body_sha256 = job->request_body_sha256;
    record.model = job->request.model;
    record.endpoint = job->request.endpoint;
    record.prompt_kind = job->request.prompt_kind;
    record.prompt_tokens = generation.prompt_token_ids.size();
    record.prompt_token_ids_u32le_sha256 =
        sha256_token_ids_u32le(generation.prompt_token_ids);
    record.consumed_prompt_tokens = consumed_prompt_tokens(generation);
    record.full_prompt_consumed =
        record.consumed_prompt_tokens == record.prompt_tokens;
    record.completion_tokens = generation.generated_token_ids.size();
    record.queue = measured_phase(
        "gateway_inference_queue",
        elapsed_milliseconds(job->queued_at, execution_started_at));
    record.admission =
        job->request.prompt_kind == OpenAIPromptKind::kTokenIds
            ? measured_phase(
                  "gateway_protocol_and_token_id_capacity",
                  elapsed_milliseconds(job->request_received_at,
                                       job->admitted_at))
            : unavailable_phase(
                  "capacity_admission",
                  "tokenization_and_capacity_not_separately_instrumented");
    record.generation = measured_phase(
        "engine_call_wall",
        elapsed_milliseconds(generation_started_at,
                             generation_finished_at));
    record.pure_prefill = measured_phase(
        "engine_prompt_prefill",
        generation.timing.prompt_prefill_milliseconds);
    record.finalize = measured_phase(
        "engine_finish_prefill",
        generation.timing.finish_prefill_milliseconds);
    record.ttft = observer.first_token_committed_at.has_value()
                      ? measured_phase(
                            "body_received_to_first_token_commit",
                            elapsed_milliseconds(
                                job->request_received_at,
                                *observer.first_token_committed_at))
                      : unavailable_phase(
                            "body_received_to_first_token_commit",
                            "first_token_commit_not_observed");
    record.first_byte = unavailable_phase(
        "external_first_response_byte", "socket_write_not_instrumented");
    record.decode = measured_phase(
        "engine_decode_after_first",
        generation.timing.decode_after_first_milliseconds);
    record.total = measured_phase(
        "body_received_to_response_enqueued",
        elapsed_milliseconds(job->request_received_at, response_ready_at));
    record.requested_prefill_chunk_size =
        generation.requested_prefill_chunk_size;
    record.effective_prefill_chunk_size =
        generation.effective_prefill_chunk_size;
    record.prefix_execution_count =
        generation.timing.prefix_execution_milliseconds.size();
    record.prefill_route_evidence = generation.prefill_route_evidence;
    record.projection_backend = engine.load_stats().projection_backend;
    std::cerr << serialize_target_prefill_witness(record) << '\n';
  } catch (...) {
    // Evidence is observational. Never convert an already successful model
    // response into an API failure because host-side serialization failed.
  }
}

void publish_job_error(const std::shared_ptr<InferenceJob>& job,
                       const OpenAIProtocolError& error,
                       const std::atomic<bool>& stopping) {
  const std::string body = serialize_openai_error(error);
  if (job->request.stream) {
    {
      std::lock_guard<std::mutex> lock(job->mutex);
      if (job->stream_start ==
          InferenceJob::StreamStartState::kPending) {
        job->stream_start =
            InferenceJob::StreamStartState::kEarlyError;
        job->response_status = error.http_status;
        job->response_body = body;
        job->response_ready = true;
        job->event_stream_closed = true;
        job->condition.notify_all();
        return;
      }
    }
    if (!job->cancelled.load(std::memory_order_relaxed)) {
      (void)job->push_event(sse_data(body), stopping);
    }
    job->close_events();
  } else {
    job->set_response(error.http_status, body);
  }
}

void execute_job(runtime::ReferenceEngine& engine,
                 const std::shared_ptr<InferenceJob>& job,
                 const EvaluationServerOptions& options,
                 const std::atomic<bool>& stopping) {
  if (job->cancelled.load(std::memory_order_relaxed) ||
      stopping.load(std::memory_order_relaxed)) {
    publish_job_error(job,
                      simple_error(503, "server_stopping",
                                   "server is stopping"),
                      stopping);
    return;
  }
  const Clock::time_point execution_started_at = Clock::now();

  ObserverContext observer;
  observer.job = job;
  observer.stopping = &stopping;
  observer.stream = job->request.stream;
  runtime::ReferenceGenerateOptions generate_options;
  generate_options.max_new_tokens = job->request.max_tokens;
  generate_options.prefill_chunk_size = options.prefill_chunk_size;
  generate_options.logits_mode =
      runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  generate_options.token_observer = observe_gateway_token;
  generate_options.token_observer_context = &observer;

  runtime::ReferenceGenerateResult generated;
  const Clock::time_point generation_started_at = Clock::now();
  switch (job->request.prompt_kind) {
    case OpenAIPromptKind::kChatMessages:
      generated = engine.generate_chat(job->request.messages,
                                       generate_options);
      break;
    case OpenAIPromptKind::kRawText:
      generated = engine.generate_prompt(job->request.prompt,
                                         generate_options);
      break;
    case OpenAIPromptKind::kTokenIds:
      generated = engine.generate_prompt_token_ids(
          job->request.prompt_token_ids, generate_options);
      break;
  }
  const Clock::time_point generation_finished_at = Clock::now();

  if (observer.protocol_failure) {
    publish_job_error(
        job,
        simple_error(500, "stream_encoding_error",
                     observer.protocol_failure_message),
        stopping);
    return;
  }
  if (job->cancelled.load(std::memory_order_relaxed) ||
      stopping.load(std::memory_order_relaxed)) {
    job->close_events();
    return;
  }
  if (!generated) {
    std::cerr << "evaluation request " << job->id
              << " failed stage=" << generated.diagnostic.stage
              << " code=" << runtime::to_string(generated.diagnostic.code)
              << " message=" << generated.diagnostic.message
              << " context=" << generated.diagnostic.context << '\n';
    const bool client_error =
        generated.diagnostic.code ==
            runtime::ReferenceEngineError::kInvalidArgument ||
        generated.diagnostic.code ==
            runtime::ReferenceEngineError::kCapacityExceeded ||
        generated.diagnostic.code ==
            runtime::ReferenceEngineError::kTokenizerFailure;
    publish_job_error(
        job,
        simple_error(client_error ? 400 : 500,
                     client_error ? "invalid_request"
                                  : "engine_error",
                     generated.diagnostic.code ==
                             runtime::ReferenceEngineError::kTokenizerFailure
                         ? "prompt formatting or tokenization failed"
                         : (client_error
                                ? generated.diagnostic.message
                                : "native inference failed; inspect server "
                                  "logs")),
        stopping);
    return;
  }
  if (generated.value->stop_reason ==
      runtime::ReferenceStopReason::kCancelled) {
    job->close_events();
    return;
  }

  const OpenAIUsage usage = usage_for(*generated.value);
  if (job->request.stream) {
    if (job->request.include_usage &&
        !job->push_event(
            sse_data(serialize_usage_chunk(
                job->request.endpoint, job->id, job->created,
                job->request.model, usage)),
            stopping)) {
      job->cancelled.store(true, std::memory_order_relaxed);
      job->close_events();
      return;
    }
    (void)job->push_event("data: [DONE]\n\n", stopping);
    job->close_events();
  } else {
    std::size_t complete_bytes = 0U;
    if (!complete_utf8_prefix(generated.value->generated_text,
                              complete_bytes) ||
        complete_bytes != generated.value->generated_text.size()) {
      publish_job_error(
          job,
          simple_error(500, "response_encoding_error",
                       "generated text is not complete valid UTF-8"),
          stopping);
      return;
    }
    const OpenAIFinishReason reason =
        finish_reason(generated.value->stop_reason);
    const std::string body =
        job->request.endpoint == OpenAIEndpoint::kChatCompletions
            ? serialize_chat_completion(
                  job->id, job->created, job->request.model,
                  generated.value->generated_text, reason, usage)
            : serialize_text_completion(
                  job->id, job->created, job->request.model,
                  generated.value->generated_text, reason, usage);
    job->set_response(200, body);
  }
  const Clock::time_point response_ready_at = Clock::now();
  emit_target_prefill_witness(
      engine, job, observer, *generated.value, execution_started_at,
      generation_started_at, generation_finished_at, response_ready_at);
}

void inference_worker(runtime::ReferenceEngine& engine,
                      BoundedQueue<std::shared_ptr<InferenceJob>>& queue,
                      const EvaluationServerOptions& options,
                      const std::atomic<bool>& stopping) {
  std::shared_ptr<InferenceJob> job;
  while (queue.pop(job)) {
    try {
      execute_job(engine, job, options, stopping);
    } catch (const std::exception& error) {
      std::cerr << "evaluation request " << job->id
                << " raised an unexpected exception: " << error.what()
                << '\n';
      try {
        publish_job_error(
            job,
            simple_error(500, "gateway_error",
                         "unexpected native gateway failure; inspect "
                         "server logs"),
            stopping);
      } catch (...) {
        job->cancelled.store(true, std::memory_order_relaxed);
        job->close_events();
      }
    } catch (...) {
      try {
        publish_job_error(
            job,
            simple_error(500, "gateway_error",
                         "unexpected native gateway failure"),
            stopping);
      } catch (...) {
        job->cancelled.store(true, std::memory_order_relaxed);
        job->close_events();
      }
    }
  }
}

[[nodiscard]] bool wait_for_response(
    const int fd, const std::shared_ptr<InferenceJob>& job,
    const std::atomic<bool>& stopping) {
  std::unique_lock<std::mutex> lock(job->mutex);
  while (!job->response_ready) {
    job->condition.wait_for(lock, std::chrono::milliseconds(100));
    if (job->cancelled.load(std::memory_order_relaxed) ||
        stopping.load(std::memory_order_relaxed) || peer_disconnected(fd)) {
      job->cancelled.store(true, std::memory_order_relaxed);
      job->condition.notify_all();
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool stream_response(
    const int fd, const std::shared_ptr<InferenceJob>& job,
    const EvaluationServerOptions& options,
    const std::atomic<bool>& stopping) {
  while (true) {
    std::string event;
    {
      std::unique_lock<std::mutex> lock(job->mutex);
      while (job->events.empty() && !job->event_stream_closed) {
        job->condition.wait_for(lock, std::chrono::milliseconds(100));
        if (stopping.load(std::memory_order_relaxed) ||
            peer_disconnected(fd)) {
          job->cancelled.store(true, std::memory_order_relaxed);
          job->condition.notify_all();
          return false;
        }
      }
      if (job->events.empty() && job->event_stream_closed) {
        return send_chunk_end(fd, options.write_timeout_milliseconds);
      }
      event = std::move(job->events.front());
      job->events.pop_front();
      job->condition.notify_all();
    }
    if (!send_chunk(fd, event, options.write_timeout_milliseconds,
                    &stopping)) {
      job->cancelled.store(true, std::memory_order_relaxed);
      job->condition.notify_all();
      return false;
    }
  }
}

[[nodiscard]] bool wait_for_stream_start(
    const int fd, const std::shared_ptr<InferenceJob>& job,
    const std::atomic<bool>& stopping, int& early_error_status,
    std::string& early_error_body, bool& ready_for_sse) {
  std::unique_lock<std::mutex> lock(job->mutex);
  while (job->stream_start ==
             InferenceJob::StreamStartState::kPending &&
         !job->event_stream_closed) {
    job->condition.wait_for(lock, std::chrono::milliseconds(100));
    if (job->cancelled.load(std::memory_order_relaxed) ||
        stopping.load(std::memory_order_relaxed) || peer_disconnected(fd)) {
      job->cancelled.store(true, std::memory_order_relaxed);
      job->condition.notify_all();
      return false;
    }
  }
  ready_for_sse =
      job->stream_start == InferenceJob::StreamStartState::kSuccess;
  if (!ready_for_sse) {
    if (job->stream_start !=
            InferenceJob::StreamStartState::kEarlyError ||
        !job->response_ready) {
      return false;
    }
    early_error_status = job->response_status;
    early_error_body = job->response_body;
  }
  return true;
}

void handle_connection(
    UniqueFd connection,
    BoundedQueue<std::shared_ptr<InferenceJob>>& inference_queue,
    const EvaluationServerOptions& options,
    const std::atomic<bool>& stopping,
    std::atomic<std::uint64_t>& next_request_id) {
  std::string buffer;
  for (std::size_t request_count = 0U;
       request_count < kMaximumRequestsPerConnection &&
       !stopping.load(std::memory_order_relaxed);
       ++request_count) {
    HttpReadResult read = read_http_request(
        connection.get(), buffer, options.read_timeout_milliseconds,
        stopping);
    if (!read.value.has_value()) {
      if (!read.peer_closed && read.error_status != 0) {
        const OpenAIProtocolError error = simple_error(
            read.error_status, "http_error", read.error_message);
        (void)send_fixed_response(
            connection.get(), read.error_status,
            serialize_openai_error(error), false,
            options.write_timeout_milliseconds);
      }
      return;
    }
    HttpRequest request = std::move(*read.value);
    const Clock::time_point request_received_at = Clock::now();
    const bool final_connection_request =
        !request.keep_alive ||
        request_count + 1U == kMaximumRequestsPerConnection;

    if (request.method == "GET" && request.target == "/healthz") {
      const std::string body = serialize_health_response(options.served_model);
      if (!send_fixed_response(connection.get(), 200, body,
                               !final_connection_request,
                               options.write_timeout_milliseconds) ||
          final_connection_request) {
        return;
      }
      continue;
    }
    if (request.method == "GET" && request.target == "/v1/models") {
      const auto created = static_cast<std::int64_t>(
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count());
      const std::string body =
          serialize_models_response(options.served_model, created);
      if (!send_fixed_response(connection.get(), 200, body,
                               !final_connection_request,
                               options.write_timeout_milliseconds) ||
          final_connection_request) {
        return;
      }
      continue;
    }

    const bool chat = request.target == "/v1/chat/completions";
    const bool completion = request.target == "/v1/completions";
    if (!chat && !completion) {
      const OpenAIProtocolError error = simple_error(
          404, "not_found", "the requested endpoint does not exist");
      (void)send_fixed_response(connection.get(), 404,
                                serialize_openai_error(error), false,
                                options.write_timeout_milliseconds);
      return;
    }
    if (request.method != "POST") {
      const OpenAIProtocolError error = simple_error(
          405, "method_not_allowed", "endpoint requires POST");
      (void)send_fixed_response(
          connection.get(), 405, serialize_openai_error(error), false,
          options.write_timeout_milliseconds, "Allow: POST\r\n");
      return;
    }
    if (!is_json_content_type(request.content_type)) {
      const OpenAIProtocolError error = simple_error(
          415, "unsupported_media_type",
          "Content-Type must be application/json");
      (void)send_fixed_response(connection.get(), 415,
                                serialize_openai_error(error), false,
                                options.write_timeout_milliseconds);
      return;
    }

    OpenAIParseResult parsed = parse_openai_request(
        request.body,
        chat ? OpenAIEndpoint::kChatCompletions
             : OpenAIEndpoint::kCompletions,
        options.served_model, options.maximum_output_tokens,
        options.max_sequence_length);
    if (!parsed) {
      if (!send_fixed_response(connection.get(), parsed.error.http_status,
                               serialize_openai_error(parsed.error),
                               !final_connection_request,
                               options.write_timeout_milliseconds) ||
          final_connection_request) {
        return;
      }
      continue;
    }
    const Clock::time_point admitted_at = Clock::now();

    auto job = std::make_shared<InferenceJob>();
    job->request = std::move(*parsed.value);
    job->request_body_sha256 = q3x::core::sha256(request.body).hex();
    const std::uint64_t sequence =
        next_request_id.fetch_add(1U, std::memory_order_relaxed);
    job->id = (chat ? "chatcmpl-q3x-" : "cmpl-q3x-") +
              std::to_string(sequence);
    job->created = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    job->event_capacity = options.stream_event_capacity;
    job->request_received_at = request_received_at;
    job->admitted_at = admitted_at;
    job->queued_at = Clock::now();
    if (!inference_queue.try_push(job)) {
      const bool server_stopping =
          stopping.load(std::memory_order_relaxed);
      const int status = server_stopping ? 503 : 429;
      const OpenAIProtocolError error = simple_error(
          status, server_stopping ? "server_stopping" : "queue_full",
          server_stopping ? "server is stopping"
                          : "the bounded inference queue is full");
      if (!send_fixed_response(connection.get(), status,
                               serialize_openai_error(error),
                               !final_connection_request,
                               options.write_timeout_milliseconds,
                               "Retry-After: 1\r\n") ||
          final_connection_request) {
        return;
      }
      continue;
    }

    if (job->request.stream) {
      int early_error_status = 500;
      std::string early_error_body;
      bool ready_for_sse = false;
      if (!wait_for_stream_start(
              connection.get(), job, stopping, early_error_status,
              early_error_body, ready_for_sse)) {
        return;
      }
      const bool sent = ready_for_sse
                            ? send_sse_headers(
                                  connection.get(),
                                  !final_connection_request,
                                  options.write_timeout_milliseconds) &&
                                  stream_response(connection.get(), job,
                                                  options, stopping)
                            : send_fixed_response(
                                  connection.get(), early_error_status,
                                  early_error_body,
                                  !final_connection_request,
                                  options.write_timeout_milliseconds);
      if (!sent || final_connection_request) {
        job->cancelled.store(true, std::memory_order_relaxed);
        job->condition.notify_all();
        return;
      }
    } else {
      if (!wait_for_response(connection.get(), job, stopping)) {
        return;
      }
      std::string body;
      int status = 500;
      {
        std::lock_guard<std::mutex> lock(job->mutex);
        body = job->response_body;
        status = job->response_status;
      }
      if (!send_fixed_response(connection.get(), status, body,
                               !final_connection_request,
                               options.write_timeout_milliseconds) ||
          final_connection_request) {
        return;
      }
    }
  }
}

void ingress_worker(
    BoundedQueue<UniqueFd>& connections,
    BoundedQueue<std::shared_ptr<InferenceJob>>& inference_queue,
    const EvaluationServerOptions& options,
    const std::atomic<bool>& stopping,
    std::atomic<std::uint64_t>& next_request_id) {
  UniqueFd connection;
  while (connections.pop(connection)) {
    try {
      handle_connection(std::move(connection), inference_queue, options,
                        stopping, next_request_id);
    } catch (const std::exception& error) {
      std::cerr << "evaluation ingress failure: " << error.what() << '\n';
    } catch (...) {
      std::cerr << "evaluation ingress failure: unknown exception\n";
    }
  }
}

[[nodiscard]] UniqueFd create_listener(
    const EvaluationServerOptions& options, std::string& error) {
  UniqueFd listener(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (!listener) {
    error = "socket failed: " + std::string(std::strerror(errno));
    return {};
  }
  const int enabled = 1;
  if (::setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &enabled,
                   sizeof(enabled)) != 0) {
    error = "setsockopt(SO_REUSEADDR) failed: " +
            std::string(std::strerror(errno));
    return {};
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(options.port);
  if (::inet_pton(AF_INET, options.bind_address.c_str(),
                  &address.sin_addr) != 1) {
    error = "bind address must be one numeric IPv4 address";
    return {};
  }
  if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address),
             sizeof(address)) != 0) {
    error = "bind failed: " + std::string(std::strerror(errno));
    return {};
  }
  if (::listen(listener.get(),
               static_cast<int>(std::min<std::size_t>(
                   options.accepted_connection_capacity, 128U))) != 0) {
    error = "listen failed: " + std::string(std::strerror(errno));
    return {};
  }
  const int flags = ::fcntl(listener.get(), F_GETFL, 0);
  if (flags < 0 || ::fcntl(listener.get(), F_SETFL, flags | O_NONBLOCK) != 0) {
    error = "failed to make listener nonblocking";
    return {};
  }
  return listener;
}

[[nodiscard]] bool valid_options(const EvaluationServerOptions& options,
                                 std::string& error) {
  std::size_t served_model_bytes = 0U;
  if (options.model_directory.empty() || options.bind_address.empty() ||
      options.served_model.empty() || options.max_sequence_length == 0U ||
      options.max_sequence_length >
          runtime::kAbsoluteRequestMaxSequenceLength ||
      options.maximum_output_tokens == 0U ||
      options.maximum_output_tokens > options.max_sequence_length ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size >
          runtime::kMaximumRequestPrefillChunkSize ||
      options.ingress_threads == 0U || options.ingress_threads > 64U ||
      options.accepted_connection_capacity == 0U ||
      options.inference_queue_capacity == 0U ||
      options.inference_queue_capacity > 62U ||
      options.ingress_threads < options.inference_queue_capacity + 2U ||
      options.stream_event_capacity == 0U ||
      options.read_timeout_milliseconds == 0U ||
      options.write_timeout_milliseconds == 0U ||
      !complete_utf8_prefix(options.served_model, served_model_bytes) ||
      served_model_bytes != options.served_model.size() ||
      !runtime::is_valid_projection_backend(options.projection_backend)) {
    error = "evaluation server options are outside fixed safe bounds";
    return false;
  }
  if (options.bind_address != "127.0.0.1") {
    error = "the unauthenticated evaluation gateway is restricted to "
            "127.0.0.1";
    return false;
  }
  return true;
}

class WorkerJoinGuard final {
 public:
  WorkerJoinGuard(
      std::atomic<bool>& stopping, BoundedQueue<UniqueFd>& connections,
      BoundedQueue<std::shared_ptr<InferenceJob>>& inference_queue,
      std::vector<std::thread>& ingress, std::thread& inference) noexcept
      : stopping_(stopping),
        connections_(connections),
        inference_queue_(inference_queue),
        ingress_(ingress),
        inference_(inference) {}

  ~WorkerJoinGuard() {
    stopping_.store(true, std::memory_order_relaxed);
    connections_.close();
    inference_queue_.close();
    for (std::thread& thread : ingress_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    if (inference_.joinable()) {
      inference_.join();
    }
  }

  WorkerJoinGuard(const WorkerJoinGuard&) = delete;
  WorkerJoinGuard& operator=(const WorkerJoinGuard&) = delete;

 private:
  std::atomic<bool>& stopping_;
  BoundedQueue<UniqueFd>& connections_;
  BoundedQueue<std::shared_ptr<InferenceJob>>& inference_queue_;
  std::vector<std::thread>& ingress_;
  std::thread& inference_;
};

}  // namespace

int run_evaluation_server(const EvaluationServerOptions& options,
                          std::atomic<bool>& stop_requested,
                          std::string& error_message) {
  if (!valid_options(options, error_message)) {
    return 2;
  }

  runtime::ReferenceEngineOptions engine_options;
  engine_options.projection_backend = options.projection_backend;
  engine_options.request_options.batch_size = 1U;
  engine_options.request_options.max_sequence_length =
      options.max_sequence_length;
  engine_options.request_options.prefill_chunk_size =
      options.prefill_chunk_size;
  engine_options.request_options.max_arena_bytes =
      options.request_max_arena_bytes;
  engine_options.request_options.min_free_bytes_after_create =
      options.request_min_free_bytes_after_create;
  engine_options.decode_graph_cache_policy =
      options.projection_backend ==
              runtime::ProjectionBackend::kSm87WeightOnly
          ? runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions
          : runtime::ReferenceDecodeGraphCachePolicy::kDisabled;

  std::cout << "loading resident model from " << options.model_directory
            << "\n";
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(options.model_directory,
                                       engine_options);
  if (!created) {
    error_message = "engine creation failed stage=" +
                    created.diagnostic.stage + " code=" +
                    std::string(runtime::to_string(
                        created.diagnostic.code)) +
                    " message=" + created.diagnostic.message +
                    " context=" + created.diagnostic.context;
    return 3;
  }
  runtime::ReferenceEngine engine = std::move(*created.value);

  UniqueFd listener = create_listener(options, error_message);
  if (!listener) {
    return 4;
  }

  BoundedQueue<UniqueFd> connections(
      options.accepted_connection_capacity);
  BoundedQueue<std::shared_ptr<InferenceJob>> inference_queue(
      options.inference_queue_capacity);
  std::atomic<std::uint64_t> next_request_id{1U};
  std::vector<std::thread> ingress;
  ingress.reserve(options.ingress_threads);
  std::thread inference;
  WorkerJoinGuard workers(stop_requested, connections, inference_queue,
                          ingress, inference);
  inference = std::thread(inference_worker, std::ref(engine),
                          std::ref(inference_queue), std::cref(options),
                          std::cref(stop_requested));
  for (std::size_t index = 0U; index < options.ingress_threads; ++index) {
    ingress.emplace_back(ingress_worker, std::ref(connections),
                         std::ref(inference_queue), std::cref(options),
                         std::cref(stop_requested),
                         std::ref(next_request_id));
  }

  const runtime::ReferenceEngineLoadStats& load = engine.load_stats();
  std::cout << "ready: http://" << options.bind_address << ':'
            << options.port << "/v1 model=" << options.served_model
            << " max_sequence_length=" << options.max_sequence_length
            << " prefill_chunk_size=" << options.prefill_chunk_size
            << " inference_workers=1 queue_capacity="
            << options.inference_queue_capacity
            << " fp8_prefill_supermatrix_sidecar_ms="
            << load.fp8_prefill_supermatrix_sidecar_milliseconds
            << " fp8_prefill_supermatrix_sidecars_enabled="
            << (load.fp8_prefill_supermatrix_sidecars_enabled ? 1 : 0)
            << " fp8_prefill_supermatrix_sidecar_projections="
            << load.fp8_prefill_supermatrix_sidecar_projections
            << " fp8_prefill_supermatrix_sidecar_bytes="
            << load.fp8_prefill_supermatrix_sidecar_bytes
            << " nvfp4_down_consumer_order_requested="
            << (load.nvfp4_down_consumer_order_sidecars_requested ? 1 : 0)
            << " nvfp4_down_consumer_order_enabled="
            << (load.nvfp4_down_consumer_order_sidecars_enabled ? 1 : 0)
            << " nvfp4_down_consumer_order_layers="
            << load.nvfp4_down_consumer_order_sidecar_layers
            << " nvfp4_down_consumer_order_bytes="
            << load.nvfp4_down_consumer_order_sidecar_bytes << '\n';

  bool fatal_accept_error = false;
  while (!stop_requested.load(std::memory_order_relaxed)) {
    pollfd descriptor{};
    descriptor.fd = listener.get();
    descriptor.events = POLLIN;
    const int polled = ::poll(&descriptor, 1U, 250);
    if (polled < 0) {
      if (errno == EINTR) {
        continue;
      }
      error_message = "listener poll failed: " +
                      std::string(std::strerror(errno));
      fatal_accept_error = true;
      break;
    }
    if (polled == 0 || (descriptor.revents & POLLIN) == 0) {
      continue;
    }
    for (std::size_t accepted_this_poll = 0U;
         accepted_this_poll < kMaximumAcceptsPerPoll &&
         !stop_requested.load(std::memory_order_relaxed);
         ++accepted_this_poll) {
      sockaddr_in peer{};
      socklen_t peer_length = sizeof(peer);
      const int accepted = ::accept4(
          listener.get(), reinterpret_cast<sockaddr*>(&peer), &peer_length,
          SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (accepted < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          break;
        }
        if (errno == EINTR) {
          if (stop_requested.load(std::memory_order_relaxed)) {
            break;
          }
          continue;
        }
        error_message = "accept failed: " +
                        std::string(std::strerror(errno));
        fatal_accept_error = true;
        break;
      }
      if (stop_requested.load(std::memory_order_relaxed)) {
        (void)::close(accepted);
        break;
      }
      UniqueFd connection(accepted);
      const int enabled = 1;
      (void)::setsockopt(connection.get(), IPPROTO_TCP, TCP_NODELAY,
                         &enabled, sizeof(enabled));
      if (!connections.try_push(std::move(connection))) {
        const OpenAIProtocolError error = simple_error(
            503, "ingress_full", "the bounded HTTP ingress is full");
        (void)send_fixed_response(
            accepted, 503, serialize_openai_error(error), false,
            std::min<std::uint32_t>(options.write_timeout_milliseconds,
                                    1'000U));
      }
    }
    if (fatal_accept_error) {
      break;
    }
  }

  listener.reset();
  return fatal_accept_error ? 5 : 0;
}

}  // namespace q3x::server
