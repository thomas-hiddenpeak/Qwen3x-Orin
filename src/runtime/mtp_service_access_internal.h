#pragma once

namespace q3x::runtime {
class ReferenceEngine;
namespace mtp_detail {
class Service;
// Source-private service borrowing. The engine owns and outlives the service;
// HTTP requests are serialized. This exposes no runner/state/model test access.
class ServiceAccess {
 public:
  static Service* service(ReferenceEngine&) noexcept;
};
}  // namespace mtp_detail
}  // namespace q3x::runtime
