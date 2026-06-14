"""Environment doctor: report the full compute stack and NVDEC status.

Run with:  python -m sentinelai.doctor

Exit code is 0 if the environment is usable (torch importable + ffmpeg present),
1 otherwise. NVDEC absence is NOT a failure on a CPU host — it is expected and
reported as a software-decode fallback.
"""

from __future__ import annotations

import platform
import sys

from . import video


def _line(label: str, value: str, ok: bool | None = None) -> str:
    mark = "  " if ok is None else ("✅" if ok else "⚠️ ")
    return f"{mark} {label:<22} {value}"


def main() -> int:
    print("=" * 60)
    print("SentinelAI environment doctor")
    print("=" * 60)

    usable = True

    # --- Host / Python ---
    print("\n[ Host ]")
    print(_line("Python", platform.python_version()))
    print(_line("Platform", f"{platform.system()} {platform.release()}"))
    print(_line("Interpreter", sys.executable))

    # --- PyTorch / CUDA ---
    print("\n[ PyTorch / CUDA ]")
    try:
        from .hardware import get_device_info

        info = get_device_info()
        print(_line("torch", info.torch_version, ok=True))
        if info.cuda_available:
            print(_line("CUDA", f"available (built for cu{info.torch_cuda_version})", ok=True))
            print(_line("GPU", info.name, ok=True))
            print(_line("compute device", "cuda", ok=True))
        else:
            print(_line("CUDA", "not available — CPU-only torch", ok=False))
            print(_line("compute device", "cpu"))
    except Exception as e:  # torch missing or broken
        usable = False
        print(_line("torch", f"NOT importable: {e}", ok=False))

    # --- FFmpeg / NVDEC ---
    print("\n[ FFmpeg / NVDEC ]")
    ff = video.probe_ffmpeg(selftest=True)
    if ff.path is None:
        usable = False
        print(_line("ffmpeg", "NOT found on PATH", ok=False))
    else:
        print(_line("ffmpeg", f"{ff.path} ({ff.version})", ok=True))
        print(_line("hwaccels", ", ".join(ff.hwaccels) or "(none)"))
        print(_line("cuvid decoders", ", ".join(ff.cuvid_decoders) or "(none)"))
        if ff.nvdec_verified:
            print(_line("NVDEC", "verified (encode→GPU-decode roundtrip OK)", ok=True))
        elif ff.nvdec_capable:
            print(_line("NVDEC", "build looks capable but roundtrip FAILED", ok=False))
        else:
            print(_line("NVDEC", "unavailable — software decode fallback", ok=False))

    # --- Resolved decode path ---
    print("\n[ Resolved decode path (h264) ]")
    args = video.recommended_decode_args("h264", info=ff)
    print(_line("ffmpeg input flags", " ".join(args) if args else "(software default)"))

    print("\n" + "=" * 60)
    print("RESULT:", "OK — environment usable" if usable else "PROBLEM — see ⚠️ above")
    print("=" * 60)
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
