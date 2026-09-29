"""Deterministic fault-injection gates for concurrency regression verification.

Disabled unless ``KEYVAULT_FAULT_INJECTION=1`` **and** the request carries the
matching test headers; in a normal deployment every call is a constant-time
no-op. The gates let the regression harness (``scripts/race_recovery.py``)
pin the exact interleaving that previously corrupted ``/recoveries/shares``:

* one request parked **while holding the session row lock** after its verified
  shard entered the in-memory pool (another request is then queued on the lock);
* that request failing from the audit-write / reconstruction path;
* a second request parked right after it acquires the released lock — i.e. at
  the precise point where, in the buggy build, it could observe the first
  request's uncommitted in-memory share, or replace it and have a stale rollback
  delete the newer share.

Gate names used by the share-submit flow:

* ``submit.after_add``   — after the verified shard was put in the pending
                          pool, still inside the transaction, row lock held.
* ``submit.abort_undo`` — on the exception path, **after** this request's own
                          in-memory insertion was retracted but still inside
                          the transaction (the row lock is not yet released).
* ``submit.after_lock`` — immediately after acquiring the session row lock.
* ``submit.before_reconstruct`` — threshold reached, right before combining.

The same gates exist for the cross-policy migration share flow under the
``migration.*`` prefix (``migration.after_lock`` / ``migration.after_add`` /
``migration.before_resplit``), pinning the identical rollback/lock interleaving
on in-memory source-shard accumulation and target re-sharding.
``scripts/race_migration.py`` additionally parks a threshold-reaching submit at
``migration.before_resplit`` (row lock held, pending shares lost) while a
second submitter queues on the lock, proving the queued request can only fail
fast with ``MIGRATION_FAILED`` once the gap detector commits the ``failed``
terminal state — never complete the doomed session.

Logical-destruction interleavings (v1.6, ``scripts/race_destruction.py``) add:

* ``destruction.after_lineage_lock`` — after every custody row of the migration
  lineage is locked but before the in-flight recovery/migration session rows are
  locked. A request parked here lets a migration-completing share submit or a
  recovery shard claim queue on its locks, pinning the deterministic ordering of
  destruction versus in-flight recovery/migration and proving the queued, older
  session can no longer hand out shard plaintext / wrapped ciphertext nor insert
  a missed descendant once destruction wins.
* ``destruction.after_idem_precheck`` — after the idempotency-receipt precheck
  passes (no committed receipt for the key) but before lineage expansion. Two
  concurrent ``POST /destructions`` carrying the same Idempotency-Key can be
  parked here simultaneously and then released together, pinning the exact
  interleaving in which both requests pass their prechecks and the loser only
  finds the winner at the ``uq_destruction_idem`` unique index: exactly one
  ``201`` and one stable ``409 IDEMPOTENCY_CONFLICT`` (or a ``200`` same-receipt
  replay when both target the same root), with the loser's whole transaction
  rolled back.
* ``destruction.before_receipt_insert`` — fail point on the receipt-write path
  (after lineage custody/session mutations, before the INSERT): ``RAISE``
  injects a generic failure and ``UNIQUE_VIOLATION_OTHER`` injects a 23505 that
  belongs to a *different* constraint. Both must roll the whole destruction
  back and surface as a 500 — never misclassified as IDEMPOTENCY_CONFLICT.

Header protocol (only honored when fault injection is enabled):

* ``X-Fault-Gate: <name>``        park at that gate until released
* ``X-Fault-Fail: <name>:<code>`` at that gate raise ``ApiError(code)`` before
                                  parking (used with a gate name that is
                                  visited by the same request; the harness
                                  typically pairs fail-at after_add with
                                  park-at abort_undo).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import asyncpg
from fastapi import Request

from .errors import ApiError, E_INTEGRITY


@dataclass
class _Gate:
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    parkers: int = 0


class FaultBoard:
    """Per-process gate registry; one gate per (name, token) pair."""

    def __init__(self) -> None:
        self._gates: dict[str, _Gate] = {}
        self._lock = asyncio.Lock()

    async def _gate(self, key: str) -> _Gate:
        async with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = _Gate()
                self._gates[key] = gate
            return gate

    async def wait_reached(self, key: str, timeout: float = 10.0) -> None:
        gate = await self._gate(key)
        await asyncio.wait_for(gate.reached.wait(), timeout=timeout)

    async def parked_count(self, key: str) -> int:
        """How many tasks are currently parked waiting at gate ``key``.

        ``reached`` is set when a task starts parking; ``release`` is cleared by
        :meth:`reset`. Callers that want to synchronize N concurrent parkers
        (one shared gate key, N requests) poll this until all N are inside the
        park point before releasing.
        """
        gate = await self._gate(key)
        if gate.release.is_set():
            return 0
        # reached 被 set 即表示至少有任务进入 park；多任务共用同一 gate 时
        # 事件无法区分数量，因此这里额外维护 per-key 的计数（见 park）。
        return gate.parkers

    async def reset(self, key: str) -> None:
        gate = await self._gate(key)
        gate.reached.clear()
        gate.release.clear()
        gate.parkers = 0

    async def release_gate(self, key: str) -> None:
        gate = await self._gate(key)
        gate.release.set()

    async def park(self, key: str, timeout: float = 30.0) -> None:
        """Block at gate ``key`` until :meth:`release_gate` is called."""
        gate = await self._gate(key)
        gate.parkers += 1
        gate.reached.set()
        try:
            await asyncio.wait_for(gate.release.wait(), timeout=timeout)
        finally:
            gate.parkers -= 1


def enabled(request: Request) -> bool:
    settings = getattr(request.app.state, "fault_enabled", False)
    return bool(settings)


async def maybe_raise(request: Request, gate: str) -> None:
    """Raise the injected fault when X-Fault-Fail points at ``gate``.

    Spec ``<gate>:<ERROR_CODE>`` raises a deterministic ``ApiError`` (mirrors a
    failed audit append / reconstruction); ``<gate>:RAISE`` raises a plain
    ``RuntimeError`` (mirrors a connection abort hitting the generic path);
    ``<gate>:UNIQUE_VIOLATION_OTHER`` raises an asyncpg 23505 carrying a
    different constraint name, proving constraint errors unrelated to the
    idempotency-receipt index are never misreported as IDEMPOTENCY_CONFLICT.
    """
    if not enabled(request):
        return
    spec = request.headers.get("x-fault-fail")
    if not spec:
        return
    name, _, code = spec.partition(":")
    if name != gate:
        return
    if code == "RAISE":
        raise RuntimeError(f"injected generic fault at {gate}")
    if code == "UNIQUE_VIOLATION_OTHER":
        # 构造一个 23505 但 constraint_name 指向别的约束的异常：
        # 回执写入路径必须把它当普通数据库错误上抛（整体回滚 → 500），
        # 绝不可仅凭 pgcode=23505 就判成幂等冲突。
        exc = asyncpg.exceptions.UniqueViolationError(
            'injected duplicate key value violates unique constraint "shards_pkey"'
        )
        # asyncpg 的 PostgresError 属性由服务器报文填充，这里手工补一个
        # “别的约束”名称，模拟命中非幂等唯一索引的真实 23505。
        object.__setattr__(exc, "constraint_name", "shards_pkey")
        object.__setattr__(exc, "table_name", "shards")
        raise exc
    raise ApiError(code or E_INTEGRITY, basis={"reason": "injected_fault", "gate": gate})


async def maybe_park(request: Request, gate: str) -> None:
    """Park when X-Fault-Gate names ``gate``; keyed by X-Fault-Key for isolation.

    The header may list several gates, comma-separated, so one request can be
    pinned at more than one point of its lifecycle (e.g. ``after_add`` and the
    later ``abort_undo`` on the failure path).
    """
    if not enabled(request):
        return
    wanted = {g.strip() for g in request.headers.get("x-fault-gate", "").split(",") if g.strip()}
    if gate not in wanted:
        return
    board: FaultBoard = request.app.state.fault_board
    token = request.headers.get("x-fault-key", "default")
    await board.park(f"{gate}:{token}")
