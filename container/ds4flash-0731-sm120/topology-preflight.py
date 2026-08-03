#!/usr/bin/env python3
"""Fail-closed GPU/NUMA topology preflight for the mapped-W2 container."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_BDF_RE = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4,8}):(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-7])$"
)


def _parse_list(value: str) -> frozenset[int]:
    result: set[int] = set()
    for part in value.strip().split(","):
        if not part:
            continue
        bounds = part.split("-", 1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        if stop < start:
            raise RuntimeError(f"invalid range {part!r}")
        result.update(range(start, stop + 1))
    if not result:
        raise RuntimeError(f"empty topology list {value!r}")
    return frozenset(result)


def _status_list(name: str) -> tuple[str, frozenset[int]]:
    prefix = f"{name}:"
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(prefix):
            raw = line.split(":", 1)[1].strip()
            return raw, _parse_list(raw)
    raise RuntimeError(f"/proc/self/status does not expose {name}")


def _normalize_bdf(value: str) -> str:
    match = _BDF_RE.fullmatch(value.strip())
    if match is None:
        raise RuntimeError(f"invalid PCI bus ID {value!r}")
    domain = match.group("domain")[-4:]
    return (
        f"{domain}:{match.group('bus')}:{match.group('device')}."
        f"{match.group('function')}"
    ).lower()


def _format_list(values: frozenset[int]) -> str:
    ordered = sorted(values)
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _visible_gpu() -> tuple[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            "the container requires exactly one CDI-visible GPU; "
            f"nvidia-smi reported {len(rows)}"
        )
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2 or not fields[0].startswith("GPU-"):
        raise RuntimeError(f"unexpected nvidia-smi GPU identity row: {rows[0]!r}")
    return fields[0], _normalize_bdf(fields[1])


def main() -> None:
    uuid, pci = _visible_gpu()
    pci_override = os.environ.get("VLLM_MOE_W2_MAPPED_PCI")
    if pci_override and _normalize_bdf(pci_override) != pci:
        raise RuntimeError(
            "VLLM_MOE_W2_MAPPED_PCI conflicts with the only visible GPU: "
            f"override={pci_override}, visible={pci}"
        )

    device_path = Path("/sys/bus/pci/devices") / pci
    if not device_path.is_dir():
        raise RuntimeError(f"PCI sysfs device is unavailable for visible GPU {pci}")
    reported_node = int((device_path / "numa_node").read_text().strip())
    cpus_raw, allowed_cpus = _status_list("Cpus_allowed_list")
    mems_raw, allowed_nodes = _status_list("Mems_allowed_list")

    node_override_raw = os.environ.get("VLLM_MOE_W2_MAPPED_NUMA_NODE")
    node_override = int(node_override_raw) if node_override_raw is not None else None
    if reported_node >= 0:
        selected_node = reported_node
        if node_override is not None and node_override != reported_node:
            raise RuntimeError(
                f"visible GPU {uuid} is local to NUMA node {reported_node}, but "
                f"VLLM_MOE_W2_MAPPED_NUMA_NODE requested {node_override}"
            )
        resolution = "GPU PCI sysfs"
    elif node_override is not None:
        selected_node = node_override
        resolution = "explicit VLLM_MOE_W2_MAPPED_NUMA_NODE override"
    elif len(allowed_nodes) == 1:
        selected_node = next(iter(allowed_nodes))
        resolution = "sole effective memory node fallback"
    else:
        raise RuntimeError(
            f"visible GPU {uuid} at {pci} reports NUMA node -1 while effective "
            f"memory nodes are {mems_raw}; set VLLM_MOE_W2_MAPPED_NUMA_NODE "
            "explicitly"
        )

    if selected_node not in allowed_nodes:
        raise RuntimeError(
            f"selected NUMA node {selected_node} is outside effective memory "
            f"nodes {mems_raw}"
        )
    node_path = Path("/sys/devices/system/node") / f"node{selected_node}"
    if not node_path.is_dir():
        raise RuntimeError(f"selected NUMA node {selected_node} does not exist")
    local_cpus_raw = (node_path / "cpulist").read_text().strip()
    local_cpus = _parse_list(local_cpus_raw)
    effective_local_cpus = local_cpus & allowed_cpus
    if not effective_local_cpus:
        raise RuntimeError(
            f"NUMA node {selected_node} CPUs {local_cpus_raw} do not intersect "
            f"effective CPU affinity {cpus_raw}"
        )

    print(
        "mapped-W2 topology: "
        f"gpu_uuid={uuid} pci_bus_id={pci} reported_numa_node={reported_node} "
        f"selected_numa_node={selected_node} resolution={resolution!r}"
    )
    print(
        "mapped-W2 affinity: "
        f"cpus_allowed={cpus_raw} mems_allowed={mems_raw} "
        f"node_cpus={local_cpus_raw} effective_node_cpus="
        f"{_format_list(frozenset(effective_local_cpus))}"
    )
    print(
        "mapped-W2 overrides: "
        f"pci={pci_override or '<automatic>'} "
        f"numa_node={node_override_raw or '<automatic>'}"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"mapped-W2 topology preflight failed: {exc}") from exc
    except RuntimeError as exc:
        raise SystemExit(f"mapped-W2 topology preflight failed: {exc}") from exc
