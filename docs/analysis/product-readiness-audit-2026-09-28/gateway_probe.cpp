#include "src/server/evaluation_server.cpp"
#include <cassert>

int main() {
  using namespace q3x::server;
  std::atomic<bool> stopping{false};
  auto job=std::make_shared<InferenceJob>();
  job->request.max_tokens=1; job->request.stream=true;
  job->request.model="audit"; job->event_capacity=16;
  ObserverContext observer; observer.job=job; observer.stopping=&stopping; observer.stream=true;
  q3x::runtime::ReferenceTokenEvent token;
  token.text_delta=std::string_view("\xe4",1);
  bool accepted=observe_gateway_token(&observer,token);
  std::cout << "utf8_partial_at_cap accepted=" << accepted << " protocol_failure=" << observer.protocol_failure << " reason=" << observer.protocol_failure_message << '\n';
  assert(!accepted && observer.protocol_failure);
  auto complete_job=std::make_shared<InferenceJob>();
  complete_job->request.max_tokens=1; complete_job->request.stream=true;
  complete_job->request.model="audit"; complete_job->event_capacity=16;
  ObserverContext complete; complete.job=complete_job; complete.stopping=&stopping; complete.stream=true;
  token.text_delta=std::string_view("\xe4\xb8\xad",3);
  assert(observe_gateway_token(&complete,token));
  std::cout << "utf8_complete_at_cap accepted=1\n";

  EvaluationServerOptions options;
  BoundedQueue<UniqueFd> connections(options.accepted_connection_capacity);
  BoundedQueue<std::shared_ptr<InferenceJob>> inference(options.inference_queue_capacity);
  EvaluationProductionRuntimeHealth health(true);
  std::atomic<std::uint64_t> ids{1};
  const std::string api_key="audit-secret";
  std::vector<std::thread> threads;
  for(std::size_t i=0;i<options.ingress_threads;++i)
    threads.emplace_back(ingress_worker,std::ref(connections),std::ref(inference),std::cref(options),std::cref(api_key),std::cref(health),std::cref(stopping),std::ref(ids));
  int sockets[4][2];
  for(int i=0;i<4;++i){
    assert(socketpair(AF_UNIX,SOCK_STREAM|SOCK_NONBLOCK,0,sockets[i])==0);
    assert(connections.try_push(UniqueFd(sockets[i][0])));
  }
  const char request[]="GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n";
  assert(send(sockets[3][1],request,sizeof(request)-1,0)==sizeof(request)-1);
  pollfd p{sockets[3][1],POLLIN,0};
  const auto begin=Clock::now();
  int before=poll(&p,1,750);
  std::cout << "health_with_three_idle_unauthenticated_connections poll=" << before << " elapsed_ms=" << elapsed_milliseconds(begin,Clock::now()) << " configured_read_timeout_ms=" << options.read_timeout_milliseconds << '\n';
  assert(before==0);
  close(sockets[0][1]);
  int after=poll(&p,1,2000);
  char response[8192];int n=recv(sockets[3][1],response,sizeof(response),0);
  assert(after==1 && n>0 && std::string_view(response,n).find("200 OK")!=std::string_view::npos);
  std::cout << "health_after_freeing_one_worker status=200\n";
  stopping.store(true); connections.close(); inference.close();
  for(int i=1;i<4;++i)close(sockets[i][1]);
  for(auto& t:threads)t.join();
}
