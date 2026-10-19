#include "runtime/mtp_tree_control.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
namespace m=q3x::runtime::mtp_detail;
unsigned checks=0;
void req(bool b){++checks;if(!b){std::cerr<<"tree assertion "<<checks<<"\n";std::abort();}}
struct State {
  std::uint64_t gdn=17;
  std::array<unsigned,3> conv{3,5,7};
  std::vector<unsigned> kv{19,23,29};
  bool operator==(const State& b) const {return gdn==b.gdn&&conv==b.conv&&kv==b.kv;}
};
void consume(State& s,unsigned token){s.gdn=s.gdn*6364136223846793005ULL+token+31*s.conv[0]+s.kv[s.kv.size()/2];s.conv={s.conv[1],s.conv[2],token};s.kv.push_back(token);}
unsigned predict(const State& s){return unsigned((s.gdn>>13)%60000);}
struct Backend: m::TreeRoundBackend {
  State state,draft_state,entry;std::array<State,4> nodes;
  m::TreeParents parents{};std::vector<unsigned> committed_nodes;
  unsigned mode=0,mismatch=4,count=0,staged=0,bounded=3,fault=0,invalid=0,commits=0,verify_calls=0,aborts=0;
  bool branch=true,active=false,poisoned=false;
  bool begin(unsigned n) noexcept override {active=true;entry=state;staged=n;count=0;commits=0;committed_nodes.clear();parents.fill(m::kTreeEntryParent);return fault!=1;}
  bool propose(unsigned seed,unsigned n,unsigned* ids) noexcept override {
    if(fault==2)return false;count=n;State cursor=entry;consume(cursor,seed);unsigned first=predict(cursor);
    for(unsigned i=0;i<n;++i){if(i)consume(cursor,ids[i-1]);ids[i]=predict(cursor);if((i==0&&mode!=0)||i==mismatch)ids[i]=(ids[i]+1)%60000;parents[i+1]=i;}
    if(branch&&n>=2){ids[n-1]=mode==1?first:(first+2)%60000;parents[n]=0;}
    if(invalid==1)parents[0]=0;
    if(invalid==2)parents[n]=n;
    if(invalid==3&&n>=2)ids[n-1]=ids[0];
    if(invalid==4)ids[0]=65536;
    if(invalid==5&&n==3)parents[2]=0; // Unsupported broader topology.
    draft_state=cursor;return true;
  }
  bool propose_bounded(unsigned seed,unsigned maximum,unsigned* ids,unsigned& actual) noexcept override {
    actual=bounded;if(!actual||actual>maximum)return true;return propose(seed,actual,ids);
  }
  bool proposal_parents(unsigned n,m::TreeParents& p) const noexcept override {req(n==count);p=parents;return fault!=3;}
  bool verify(unsigned seed,const unsigned* ids,unsigned n,unsigned* predictions) noexcept override {
    ++verify_calls;if(fault==4)return false;req(n+1<=staged);
    for(unsigned i=0;i<=n;++i){nodes[i]=i?nodes[parents[i]]:entry;consume(nodes[i],i?ids[i-1]:seed);predictions[i]=predict(nodes[i]);}
    state=nodes[n];if(invalid==6)predictions[n]=65536;return true;
  }
  bool commit_prefix(unsigned,unsigned) noexcept override {req(false);return false;}
  bool commit_node(unsigned node,unsigned depth,unsigned pending) noexcept override {
    if(fault==5&&depth==1)return false;if(fault==8&&depth==2)return false;
    req(depth==commits+1&&node<=count);req(depth==1?node==0:parents[node]==committed_nodes.back());
    state=nodes[node];draft_state=state;req(pending==predict(state));committed_nodes.push_back(node);++commits;return true;
  }
  bool finish() noexcept override {if(fault==6)return false;active=false;return true;}
  bool abort() noexcept override {++aborts;active=false;poisoned=true;return fault!=7;}
};
struct Observer {
  Backend& b;State scalar;unsigned pending,seen=0,cancel_after=99;
  static bool call(void* opaque,unsigned token) noexcept {
    auto& o=*static_cast<Observer*>(opaque);consume(o.scalar,o.pending);req(token==predict(o.scalar));
    req(o.b.state==o.scalar&&o.b.draft_state==o.scalar);req(o.b.commits==o.seen+1);
    o.pending=token;++o.seen;return o.seen<o.cancel_after;
  }
};
m::RoundOptions options(unsigned length,unsigned remaining,unsigned seed){return {length,seed,remaining,remaining,65536,65535,false};}
void planner(){
  const unsigned ids[3]={9,11,13};const double p[3]={-.5,-.75,-1.};
  for(unsigned n=1;n<=3;++n){
    double joint=0;for(unsigned i=0;i<n;++i)joint+=p[i];
    for(double alt:{joint-.25,joint,std::nextafter(joint,0.),-.5}){
      m::BudgetedProposal plan;
      if(alt>p[0]){req(!m::plan_budgeted_proposal(ids,p,n,10,alt,65536,plan));continue;}
      req(m::plan_budgeted_proposal(ids,p,n,10,alt,65536,plan));
      req(plan.count==n&&plan.parents[0]==m::kTreeEntryParent&&plan.depth[0]==0);
      req(plan.alternate_selected==(n>=2&&alt>joint));
      for(unsigned i=1;i<=n;++i){req(plan.parents[i]<i);req(plan.depth[i]==plan.depth[plan.parents[i]]+1);}
      if(plan.alternate_selected){req(plan.tokens[n-1]==10&&plan.parents[n]==0&&plan.depth[n]==1);}
      else for(unsigned i=0;i<n;++i)req(plan.tokens[i]==ids[i]);
    }
  }
  for(unsigned bad=0;bad<10;++bad){
    double logs[3]={-.5,-.75,-1.};double alt=-1;unsigned n=3,a=10,v=65536;const double* lp=logs;const unsigned* ip=ids;
    switch(bad){case 0:lp=nullptr;break;case 1:ip=nullptr;break;case 2:n=0;break;case 3:n=4;break;case 4:a=9;break;case 5:a=v;break;case 6:logs[1]=std::numeric_limits<double>::quiet_NaN();break;case 7:alt=-INFINITY;break;case 8:alt=-.25;break;case 9:logs[1]=.1;break;}
    m::BudgetedProposal plan;plan.count=77;req(!m::plan_budgeted_proposal(ip,lp,n,a,alt,v,plan));req(plan.count==77);
  }
}
void replay(){
  for(unsigned length:{2U,3U})for(bool branch:{false,true})for(unsigned mode=0;mode<3;++mode)
  for(unsigned mismatch=0;mismatch<=3;++mismatch)for(unsigned limit=1;limit<=8;++limit)
  for(unsigned cancel=1;cancel<=5;++cancel){
    Backend b;b.branch=branch;b.mode=mode;b.mismatch=mismatch;
    const unsigned seed=7;Observer o{b,b.state,seed};o.cancel_after=cancel;
    auto r=m::run_tree_round(options(length,limit,seed),b,Observer::call,&o);
    req(r.ok()&&!b.poisoned&&!b.active&&r.output_count==o.seen);
    req(r.committed_target_rows==r.output_count&&r.output_count<=limit);
    req(r.verified_rows==r.proposed_tokens+1&&r.proposed_tokens==std::min(length,limit-1));
    req(r.accepted_tokens<=r.proposed_tokens&&r.output_count<=r.accepted_tokens+1);
    if(branch&&mode==1&&limit>=3&&cancel>=2){req(b.committed_nodes.size()==2&&b.committed_nodes[1]==r.proposed_tokens);}
    // Continue from the committed state with the last output pending. This
    // catches hidden sibling-history contamination beyond the first observer.
    if(r.status==m::RoundStatus::kContinue){Observer next{b,o.scalar,o.pending};b.mismatch=4;b.mode=0;
      auto r2=m::run_tree_round(options(length,5,o.pending),b,Observer::call,&next);req(r2.ok());}
  }
}
void stops_and_failures(){
  for(unsigned length:{2U,3U})for(unsigned mode=0;mode<3;++mode)for(unsigned at=1;at<=4;++at){
    Backend b;b.mode=mode;State cursor=b.state;unsigned token=7,stop=0;
    for(unsigned i=0;i<at;++i){consume(cursor,token);token=predict(cursor);stop=token;}
    auto opts=options(length,8,7);opts.stop_token=stop;Observer o{b,b.state,7};o.cancel_after=at;
    auto r=m::run_tree_round(opts,b,Observer::call,&o);req(r.ok());
    if(o.seen==at)req(r.status==m::RoundStatus::kStop); // EOS wins same-token cancellation.
  }
  for(unsigned fault=1;fault<=8;++fault){Backend b;b.fault=fault;b.mode=1;Observer o{b,b.state,7};
    if(fault==7)b.invalid=4;
    auto r=m::run_tree_round(options(3,8,7),b,Observer::call,&o);
    req(!r.ok()&&b.poisoned&&b.aborts==1);req(r.abort_succeeded==(fault!=7));
    req(o.seen==(fault==6?2U:fault==8?1U:0U));
  }
  for(unsigned invalid=1;invalid<=6;++invalid){Backend b;b.invalid=invalid;Observer o{b,b.state,7};
    auto r=m::run_tree_round(options(3,8,7),b,Observer::call,&o);req(r.status==m::RoundStatus::kInvalidPrediction&&b.poisoned&&!o.seen);
    req(b.verify_calls==(invalid==6?1U:0U));
  }
  for(unsigned length:{2U,3U})for(unsigned actual=0;actual<=4;++actual){Backend b;b.bounded=actual;auto opts=options(length,8,7);opts.confidence_lookahead=true;
    Observer o{b,b.state,7};auto r=m::run_tree_round(opts,b,Observer::call,&o);
    req(r.ok()==(actual>=1&&actual<=length));if(r.ok())req(r.proposed_tokens==actual);
  }
}
void invalid_options(){
  for(unsigned kind=0;kind<10;++kind){
    Backend b;auto opts=options(3,8,7);Observer observer{b,b.state,7};
    switch(kind){case 0:opts.draft_length=1;break;case 1:opts.draft_length=4;break;
      case 2:opts.vocabulary_size=0;break;case 3:opts.seed_token=65536;break;
      case 4:opts.stop_token=65536;break;case 5:opts.seed_token=opts.stop_token;break;
      case 6:opts.remaining_output=0;break;case 7:opts.available_target_rows=7;break;
      case 8:opts.draft_length=0;break;case 9:break;}
    auto r=kind==9?m::run_tree_round(opts,b,nullptr,&observer):m::run_tree_round(opts,b,Observer::call,&observer);
    req(r.status==m::RoundStatus::kInvalidOptions&&!b.active&&!b.poisoned&&b.aborts==0&&observer.seen==0);
  }
}
int main(){planner();replay();stops_and_failures();invalid_options();std::cout<<"mtp_tree_control_checks="<<checks<<"\n";}
