"""Distributed active-token transitions for Mixture-of-Recursions."""

from .all_to_all import (
    ActiveTokenDispatcher,
    BalancedTargetPlan,
    build_balanced_target_plan,
    variable_all_to_all,
)
from .backends import (
    ActiveDispatchBackend,
    BalancedReferenceBackend,
    MagiCanonicalBackend,
    MagiDirectBackend,
    MagiDirectPlan,
    StaticReferenceBackend,
    decode_magi_direct_plan,
    get_dispatch_backend,
)
from .counters import (
    CommunicationCounters,
    CommunicationSnapshot,
    count_unexpected_real_token_ids,
)
from .group import TPxCPRouteGroup
from .layout import ActiveTokenBatch, ActiveTokenLayout
from .packing import (
    CanonicalActivePacking,
    active_sequence_alignment,
    pack_active_sequences,
    pack_route_plan_canonical,
)
from .parking import EarlyExitParking, ParkingTicket
from .route_plan import batch_from_route_plan, target_plan_from_route_plan
from .routing import (
    DistributedDepthRouterOutput,
    assert_route_plan_consistent,
    create_dense_dp_route_group,
    distributed_depth_route,
)
from .static_qkv import StaticGatheredQKV, gather_static_active_qkv
from .transition import ActiveTokenTransition, TransitionResult

__all__ = [
    "ActiveDispatchBackend",
    "ActiveTokenBatch",
    "ActiveTokenDispatcher",
    "ActiveTokenLayout",
    "ActiveTokenTransition",
    "BalancedReferenceBackend",
    "BalancedTargetPlan",
    "CanonicalActivePacking",
    "CommunicationCounters",
    "CommunicationSnapshot",
    "DistributedDepthRouterOutput",
    "EarlyExitParking",
    "MagiCanonicalBackend",
    "MagiDirectBackend",
    "MagiDirectPlan",
    "ParkingTicket",
    "StaticGatheredQKV",
    "StaticReferenceBackend",
    "TPxCPRouteGroup",
    "TransitionResult",
    "active_sequence_alignment",
    "assert_route_plan_consistent",
    "batch_from_route_plan",
    "build_balanced_target_plan",
    "count_unexpected_real_token_ids",
    "create_dense_dp_route_group",
    "decode_magi_direct_plan",
    "distributed_depth_route",
    "gather_static_active_qkv",
    "get_dispatch_backend",
    "pack_active_sequences",
    "pack_route_plan_canonical",
    "target_plan_from_route_plan",
    "variable_all_to_all",
]
