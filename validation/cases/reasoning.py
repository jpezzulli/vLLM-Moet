"""Frozen DS4Flash-0731 staged reasoning-quality suite.

This module is data only.  Prompts and keys were approved before execution.
"""

SYSTEM = (
    "You are completing a closed-book reasoning evaluation. You have no tools "
    "or external sources. Use only facts in the packet. Show enough reasoning "
    "to audit, distinguish facts from assumptions, preserve every stated "
    "constraint, and finish with: (1) a direct answer, (2) the key supporting "
    "evidence, and (3) confidence from 0–100% plus what fact would most change "
    "it. Do not invent missing facts."
)

CASES = [
    {
        "id": "C1",
        "title": "Tunnel latency diagnosis",
        "weight": 14,
        "prompt": """At 09:55 an HTTPS service was moved behind an IP-in-IP tunnel that adds exactly 96 bytes per packet.

The physical egress MTU is 1500 bytes. The inner interface remained at MTU 1500. The firewall drops every ICMP \"packet too big\"/\"fragmentation needed\" message. The TCP implementation reduces its segment size after its first retransmission timeout when it suspects an MTU black hole.

Beginning at 10:00, median latency remained 75 ms but p95 became approximately 1.1 seconds. A request is slow exactly when its first server flight contains an inner IP packet between 1405 and 1500 bytes. Flights whose largest inner packet is at most 1404 bytes are fast.

A capture before tunnel encapsulation sees the large original packet once and a smaller retransmission about one second later. A capture on the physical egress sees only the smaller retransmission. DNS remains 3 ms. CPU, queue depth, and garbage-collection pauses remain normal. The old application build is equally affected. Bypassing the tunnel immediately restores normal latency.

A router fan warning appeared at 10:01, but forwarding temperature and packet-drop counters remain within their stated operating limits.

Identify the primary mechanism, reconcile the two captures and size boundary, reject unsupported alternatives, and give the smallest corrective action and verification.""",
        "expected": [
            "PMTU black hole from 96-byte encapsulation, 1500 physical MTU, and blocked PTB",
            "1500 - 96 = 1404 explains boundary",
            "capture placement explains missing original on physical egress",
            "RTO and smaller retry explain delay",
            "set/clamp inner MTU or MSS <=1404 or allow valid PTB, then verify boundary traffic",
            "reject fan, DNS, application, CPU, and GC as primary causes",
        ],
        "fatal": "Selects application, DNS, CPU, or fan as primary cause, or recommends only increasing timeout.",
    },
    {
        "id": "C2",
        "title": "Authoritative ordering versus client logs",
        "weight": 12,
        "prompt": """A controller's signed audit sequence increments when an operation is applied. Valid sequence entries cannot be reordered. Alarm decisions use the controller state at the alarm's sequence number.

Controller records:
- Sequence 441, 14:03:12.400: configuration checksum is X.
- Sequence 442, 14:03:12.600: pressure alarm first activates.
- Sequence 443, 14:03:13.100: configuration write changes X to Y.

Configuration Y can produce this alarm condition; X cannot produce it through that mechanism. The operator workstation says the write was issued at 14:03:10.900. Workstation clocks may differ from the controller by up to four seconds, and commands can remain queued before application. The workstation UI displays the requested value optimistically before controller acknowledgement. A witness saw Y on that UI before hearing the alarm.

The SIEM received the workstation's \"write requested\" event before it received the alarm event. SIEM receipt order is explicitly not application order. All controller signatures and checksum records validate.

Decide whether the X-to-Y write caused the first alarm, explain every apparent conflict, and state what remains unknown.""",
        "expected": [
            "write cannot have caused first alarm through stated mechanism",
            "authoritative apply order is checksum X at 441, alarm 442, write Y 443",
            "workstation time/UI and SIEM establish intent or reporting, not application",
            "actual alarm cause remains unknown",
        ],
        "fatal": "Claims the write caused the first alarm or invents another cause.",
    },
    {
        "id": "C3",
        "title": "Production optimization with grade mix",
        "weight": 14,
        "prompt": """Production occurs only in whole 100-input batches. Every failed original unit is reworked exactly once, and rework recovers exactly 50% of failed units.

Source A has 92% original yield, costs $4 per original input, charges $1 to rework each failed original, uses 30 energy units per batch, and permits at most seven batches. Each completed A batch consequently has 96 accepted units, of which 72 are Grade H, and costs $408.

Source B has 80% original yield, costs $3 per original input, charges $1 to rework each failed original, uses 45 energy units per batch, and permits at most six batches. Each completed B batch consequently has 90 accepted units, of which 18 are Grade H, and costs $320.

At least 1,000 accepted units are required. Grade H must constitute at least 50% of all accepted units; accepted units cannot be discarded to manipulate that percentage. Total energy may not exceed 420 units. Minimize incremental cost. A previously paid $12,000 audit fee cannot be recovered and is identical under every plan.

Find the optimum and prove it respects all constraints. Also determine whether any feasible plan remains if A's capacity falls to five batches.""",
        "expected": [
            "5A+6B has 1020 accepted but only 468 H = 45.9%, so fails",
            "optimum 6A+5B: 1026 accepted, 522 H = 50.9%, energy 405, cost 4048",
            "7A+4B feasible but costs 4136",
            "with A max 5 no feasible plan; only quantity-capable 5A+6B fails grade mix",
            "sunk audit fee irrelevant",
        ],
        "fatal": "Uses fractional batches, discards output, violates quantity/grade/energy/capacity, or claims feasibility with only five A batches.",
    },
    {
        "id": "C4",
        "title": "Constrained non-preemptive schedule",
        "weight": 14,
        "prompt": """Time begins at t=0. There is one GPU and one specialist; each can serve only one job at a time. GPU jobs are non-preemptive. GPU maintenance occupies [6,8), and no GPU job may straddle it. CPU-only work consumes neither constrained resource.

Mandatory jobs:
- B: GPU plus specialist, duration 2, release 0, finish by t=3.
- E: GPU plus specialist, duration 2, after B, finish by t=6.
- H: GPU, duration 2, after E.
- C: specialist, duration 4, after B.
- A: GPU, duration 3, release 0.
- D: GPU, duration 4, after both A and C.
- F: CPU, duration 2, after both D and H.

Optional X uses the GPU for one hour and must finish by t=5. The primary objective is to finish mandatory F as early as possible. The secondary objective is to accept X only if it does not delay F or violate another constraint.

Give an optimal schedule, decide X, and prove the makespan rather than merely presenting a feasible schedule.""",
        "expected": [
            "B 0-2, E 2-4, H 4-6, C 4-8, maintenance 6-8, A 8-11, D 11-15, F 15-17",
            "minimum makespan 17",
            "A+D consume seven post-maintenance GPU hours for a 17 finish, so H must be pre-maintenance",
            "B+E+H consume all six pre-maintenance GPU hours",
            "reject X; accepting it pushes H post-maintenance and F to at least 19",
        ],
        "fatal": "Accepts X, overlaps specialist jobs, places H after maintenance while claiming makespan 17, violates maintenance/deadlines, or claims less than 17.",
    },
    {
        "id": "C5",
        "title": "Idempotent transfer implementation review",
        "weight": 16,
        "prompt": """Review this pseudocode:

transfer(:request_id, :src_id, :dst_id, :cents):
    old = SELECT l.result, l.src_id, l.dst_id, l.cents
          FROM ledger AS l
          WHERE l.request_id = :request_id
    if old exists:
        return old.result

    try:
        WITH TRANSACTION READ COMMITTED:
            s = SELECT a.balance FROM accounts AS a
                WHERE a.account_id = :src_id FOR UPDATE
            if s.balance < :cents:
                return \"insufficient\"
            UPDATE accounts AS a SET balance = balance - :cents
                WHERE a.account_id = :src_id
            UPDATE accounts AS a SET balance = balance + :cents
                WHERE a.account_id = :dst_id
            INSERT ledger(request_id, src_id, dst_id, cents, result)
                VALUES (:request_id, :src_id, :dst_id, :cents, \"ok\")
        return \"ok\"
    except UniqueViolation:
        return SELECT l.result FROM ledger AS l
               WHERE l.request_id = :request_id

Database facts: transactions are atomic; return inside the transaction commits; FOR UPDATE locks selected rows until completion; a unique conflict waits for the winner and then raises, rolling back the loser; updating a nonexistent account affects zero rows without error; opposite-direction transfers can deadlock when they lock accounts in opposite order; deadlock abort is exposed to the caller with no automatic retry. Inputs may contain negative amounts, identical source/destination, reused request IDs with different parameters, and missing accounts.

Required behavior: positive integer cents and distinct existing accounts; no money creation or destruction; every outcome including insufficient funds is idempotent by request ID; reusing an ID with different parameters is rejected; valid concurrent transfers do not expose an avoidable deadlock failure.

Identify the correctness failures and propose the smallest sound transaction ordering, including idempotency ownership and account locking.""",
        "expected": [
            "outside-transaction idempotency lookup insufficient",
            "concurrent duplicate can return insufficient after same request already succeeded",
            "ledger must retain and compare request fingerprint",
            "validate positive integer, distinct IDs, and both accounts exist",
            "lock both accounts in canonical ID order",
            "atomically claim request ID/fingerprint, then lock/check/update, persist every outcome, commit",
            "explicit retry or avoidance for remaining deadlocks",
        ],
        "fatal": "Declares code safe due to transaction/unique key or proposes a patch retaining inconsistent duplicates, money loss, or request-ID substitution.",
    },
    {
        "id": "C6",
        "title": "Release decision with distractions",
        "weight": 10,
        "prompt": """The shipping lot is the lot identity created by the most recent rework. Test results attach only to the exact full lot ID and do not carry across rework. Release requires valid T1 and T2 results for the shipping lot. Tests remain valid for 30 days. A waiver can replace T2 only when the lab marks T2 unavailable; it cannot replace T1.

Lot L17 passed T1 ten days ago. L17 was reworked two days ago, creating shipping lot L17-R. L17-R passed T2 yesterday. The handheld dashboard displays only the first three characters of lot IDs, so both appear as L17. Signed MES records retain complete IDs and are authoritative. The T1 lab is available. A manager signed a generic \"ship as needed\" note. The supplier has had no failures in five years, and a major customer has an urgent deadline.

Decide whether L17-R may be released and explain which evidence and rules control.""",
        "expected": [
            "do not release L17-R",
            "it lacks T1 for exact post-rework lot",
            "old T1 does not carry forward",
            "waiver cannot replace T1 and T2 is not unavailable",
            "truncated display, history, urgency, and generic note do not satisfy rule",
        ],
        "fatal": "Authorizes release.",
    },
    {
        "id": "C7",
        "title": "Posterior probability without independence",
        "weight": 10,
        "prompt": """P(D)=0.10. Conditional test rates are P(A+|D)=0.80, P(B+|D)=0.70, P(A+|not D)=0.20, and P(B+|not D)=0.10. A subject receives both positive results. No conditional-independence assumption is supplied, and no information about the joint behavior of A and B is available.

Determine what posterior can and cannot be calculated, give the sharp possible range, and optionally show the independence-only estimate clearly labeled as an assumption.""",
        "expected": [
            "exact posterior not identifiable",
            "joint positive given D in [0.50,0.70] and given not-D in [0,0.10]",
            "sharp posterior range about 35.7%-100%",
            "independence-only estimate about 75.7%, explicitly unjustified by packet",
        ],
        "fatal": "Presents 75.7% or another single posterior as determined.",
    },
    {
        "id": "C8",
        "title": "Fixed multi-turn correction",
        "weight": 10,
        "prompt": """Candidate causes are A, B, and C. Initial scores are A=1, B=0, C=0.

Evidence contributions:
- E1: A +3, B +1, C 0.
- E2: A -2, B +3, C +1.
- E3: A +1, B 0, C +4.

Select a unique cause only when its score exceeds the second-place score by at least 2. Otherwise report as ambiguous every cause within one point of the highest score.

Choose the lowest-cost containment safe for every cause in the declared set:
- W: safe for A only, cost 2.
- X: safe for B and C, cost 3.
- Y: safe for C only, cost 1.
- Z: safe for B only, cost 1.

Calculate the scores, apply the decision rule, and choose containment.""",
        "correction": "Correction: E3 was attached to the wrong device. Its verified contributions are A=+5, B=0, C=-3. All other facts and rules remain unchanged. Recalculate from the beginning, explicitly state what conclusion you retract, what remains valid, and the resulting containment action.",
        "expected": [
            "first: A3 B4 C5, ambiguous B/C, choose X",
            "correction: A7 B4 C-2, uniquely A, choose W",
            "retract B/C ambiguity and X",
            "retain initial scores, E1, E2, decision rule, action table",
        ],
        "fatal": "Retains B/C or X after correction, ignores correction, or silently changes unaffected evidence.",
    },
]

CASE_ORDER = [case["id"] for case in CASES]
