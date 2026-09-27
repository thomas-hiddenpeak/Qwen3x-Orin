// Separate host-only test executable. No hook is compiled into the server.
#include "../src/server/evaluation_server.cpp"
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void utf8_contract() {
  using q3x::server::render_utf8_output;
  const std::string replacement = "\xef\xbf\xbd";
  for (const auto& test : std::vector<std::pair<std::string, std::string>>{
      {"hello", "hello"}, {"\xe4", replacement}, {"\xe4\xb8", replacement},
      {"\xe4\xb8\xad", "\xe4\xb8\xad"}, {"\xf0\x9f\x98\x80", "\xf0\x9f\x98\x80"},
      {"\xe4" "A", replacement + "A"}, {"\xff", replacement},
      {"\xed\xa0\x80", replacement + replacement + replacement},
      {"\xc0\xaf", replacement + replacement}}) {
    auto pending = test.first;
    require(render_utf8_output(pending, true) == test.second && pending.empty(),
            "nonstream UTF-8 replacement mismatch");
    for (std::size_t split = 0; split <= test.first.size(); ++split) {
      pending = test.first.substr(0, split);
      auto rendered = render_utf8_output(pending, false);
      require(pending.size() <= 3, "unbounded UTF-8 carry");
      pending += test.first.substr(split);
      rendered += render_utf8_output(pending, true);
      require(rendered == test.second && pending.empty(), "SSE/nonstream mismatch");
    }
  }
  using namespace q3x::server;
  std::atomic<bool> stopping{false};
  auto job = std::make_shared<InferenceJob>();
  job->request.max_tokens = 1; job->request.model = "test"; job->event_capacity = 16;
  ObserverContext observer; observer.job = job; observer.stopping = &stopping;
  observer.stream = true;
  q3x::runtime::ReferenceTokenEvent token; token.token_id = 160; token.text_delta = "\xe4";
  require(observe_gateway_token(&observer, token), "partial-byte output cap failed");
  require(!observer.protocol_failure && observer.pending_utf8.empty(), "terminal carry retained");
  require(job->events.front().find(replacement) != std::string::npos &&
          job->events.front().find("length") != std::string::npos,
          "missing replacement or length finish");
}
void health_contract() {
  using namespace q3x::server;
  using Error = q3x::runtime::ReferenceEngineError;
  EvaluationProductionRuntimeHealth healthy(true);
  for (Error error : {Error::kInvalidArgument, Error::kCapacityExceeded,
                      Error::kTokenizerFailure, Error::kCancelled}) {
    q3x::runtime::ReferenceEngineDiagnostic d; d.code = error;
    healthy.observe_failure(d);
    require(healthy.ready(), "client error/cancellation poisoned readiness");
  }
  for (Error error : {Error::kRunnerResetFailure, Error::kRunnerStepFailure,
                      Error::kMissingPrediction, Error::kMissingLogits,
                      Error::kMissingTiming, Error::kPrefillPlanUnavailable}) {
    EvaluationProductionRuntimeHealth health(true);
    q3x::runtime::ReferenceEngineDiagnostic d; d.code = error;
    health.observe_failure(d); require(!health.ready(), "fatal invariant remained healthy");
    d.code = Error::kNone; health.observe_failure(d);
    require(!health.ready(), "fatal health latch reopened");
  }
  q3x::runtime::ReferenceEngineDiagnostic d; d.code = Error::kCancelled; d.cuda_error = 700;
  healthy.observe_failure(d); require(!healthy.ready(), "failed cancellation cleanup stayed ready");
}
void backpressure_contract() {
  using namespace q3x::server;
  int fd[2];
  require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, fd) == 0, "socketpair");
  UniqueFd sender(fd[0]), stalled_reader(fd[1]);
  const int small_buffer = 1024;
  require(::setsockopt(sender.get(), SOL_SOCKET, SO_SNDBUF,
                      &small_buffer, sizeof(small_buffer)) == 0, "send buffer");
  const std::string payload(1024 * 1024, 'x');
  const auto start = Clock::now();
  require(!send_all(sender.get(), payload, 100), "non-reading client did not time out");
  require(Clock::now() - start < std::chrono::seconds(1), "write timeout was unbounded");
}
void ingress_contract() {
  using namespace q3x::server;
  EvaluationServerOptions options;
  std::atomic<bool> stopping{false}; std::atomic<std::uint64_t> ids{1};
  BoundedQueue<ReadyConnection> connections(16);
  BoundedQueue<std::shared_ptr<InferenceJob>> inference(1);
  EvaluationProductionRuntimeHealth health(true);
  std::string key = "test-secret";
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < 3; ++i)
    threads.emplace_back(ingress_worker, std::ref(connections), std::ref(inference),
                         std::cref(options), std::cref(key), std::cref(health),
                         std::cref(stopping), std::ref(ids));
  struct Cleanup {
    std::atomic<bool>& stop; BoundedQueue<ReadyConnection>& queue;
    std::vector<std::thread>& threads;
    ~Cleanup() { stop.store(true); queue.close(); for (auto& t : threads) t.join(); }
  } cleanup{stopping, connections, threads};
  std::vector<PendingConnection> pending;
  std::vector<UniqueFd> clients;
  for (int i = 0; i < 4; ++i) {
    int fd[2]; require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, fd) == 0, "socketpair");
    clients.emplace_back(fd[1]);
    pending.push_back({UniqueFd(fd[0]), {}, Clock::now() + std::chrono::seconds(10)});
  }
  // Empty socket, incomplete headers, incomplete body: none may hold a worker.
  for (const auto& request : std::vector<std::pair<int, std::string>>{
      {1, "POST /v1/completions HTTP/1.1\r\n"},
      {2, "POST /v1/completions HTTP/1.1\r\nContent-Length: 100\r\n\r\nx"},
      {3, "GET /healthz HTTP/1.1\r\nHost: test\r\n\r\n"}}) {
    require(::send(clients[request.first].get(), request.second.data(), request.second.size(), 0) > 0, "send");
  }
  stage_http_connections(pending, connections, options, stopping);
  pollfd poller{clients[3].get(), POLLIN, 0};
  require(::poll(&poller, 1, 750) == 1, "slow clients blocked health response");
  char response[8192]; auto n = ::recv(clients[3].get(), response, sizeof(response), 0);
  require(n > 0 && std::string_view(response, static_cast<std::size_t>(n)).find("200 OK") != std::string_view::npos,
          "health not ready");
  require(pending.size() == 3, "incomplete requests were dispatched");
  for (auto& item : pending) item.deadline = Clock::now();
  stage_http_connections(pending, connections, options, stopping);
  require(pending.empty(), "expired incomplete sockets retained");
}
}
int main() {
  try { utf8_contract(); health_contract(); backpressure_contract(); ingress_contract(); }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
  std::cout << "PASS: UTF-8 caps/splits, fatal health classification, slow-client isolation/read/write deadlines\n";
  return 0;
}
