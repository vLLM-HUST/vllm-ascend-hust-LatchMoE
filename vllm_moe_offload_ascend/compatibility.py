"""Compatibility-profile helpers shared by installation and launch checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    torch_version: str
    torch_npu_version: str
    cann_version: str
    qualification: str


def public_version(raw_version: str | None) -> str | None:
    if raw_version is None:
        return None
    return raw_version.split("+", 1)[0]


def runtime_profiles(lock: Mapping[str, str]) -> tuple[RuntimeProfile, ...]:
    profiles: list[RuntimeProfile] = []
    prefix = "runtime_profile."
    for key, value in sorted(lock.items()):
        if not key.startswith(prefix):
            continue
        fields = value.split("|")
        if len(fields) != 4 or not all(field.strip() for field in fields):
            raise ValueError(
                f"invalid runtime profile {key}: expected "
                "torch|torch-npu|CANN|qualification"
            )
        profiles.append(
            RuntimeProfile(
                name=key.removeprefix(prefix),
                torch_version=fields[0].strip(),
                torch_npu_version=fields[1].strip(),
                cann_version=fields[2].strip(),
                qualification=fields[3].strip(),
            )
        )
    if profiles:
        return tuple(profiles)

    legacy = (
        lock.get("torch_version"),
        lock.get("torch_npu_version"),
        lock.get("cann_version"),
    )
    if all(legacy):
        return (
            RuntimeProfile(
                name="default",
                torch_version=str(legacy[0]),
                torch_npu_version=str(legacy[1]),
                cann_version=str(legacy[2]),
                qualification="legacy_lock",
            ),
        )
    return ()


def match_runtime_profile(
    lock: Mapping[str, str],
    *,
    torch_version: str | None,
    torch_npu_version: str | None,
    cann_version: str | None,
) -> RuntimeProfile | None:
    actual = (
        public_version(torch_version),
        public_version(torch_npu_version),
        cann_version,
    )
    for profile in runtime_profiles(lock):
        expected = (
            profile.torch_version,
            profile.torch_npu_version,
            profile.cann_version,
        )
        if actual == expected:
            return profile
    return None


def describe_runtime_profiles(lock: Mapping[str, str]) -> str:
    return "; ".join(
        f"{profile.name}(torch={profile.torch_version}, "
        f"torch-npu={profile.torch_npu_version}, CANN={profile.cann_version})"
        for profile in runtime_profiles(lock)
    )
