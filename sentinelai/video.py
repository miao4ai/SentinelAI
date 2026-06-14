"""FFmpeg / NVDEC capability detection and decode-flag selection.

The canonical decode path goes through the *system* ffmpeg binary so that NVDEC
works via the NVIDIA driver's libnvcuvid at runtime. We detect capability and
hand back the right input flags; callers stay identical on CPU and GPU hosts.

Nothing here imports torch — capability detection must work before/without it.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

# ffmpeg's CUVID (NVDEC) decoder name per codec.
_CUVID_DECODER = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
    "mpeg4": "mpeg4_cuvid",
}


@dataclass(frozen=True)
class FFmpegInfo:
    path: str | None
    version: str | None
    hwaccels: list[str] = field(default_factory=list)
    cuvid_decoders: list[str] = field(default_factory=list)
    # True only after an actual encode→NVDEC-decode roundtrip succeeds.
    nvdec_verified: bool = False

    @property
    def has_cuda_hwaccel(self) -> bool:
        return "cuda" in self.hwaccels

    @property
    def nvdec_capable(self) -> bool:
        """Build/driver *look* able to do NVDEC (not a runtime guarantee)."""
        return self.has_cuda_hwaccel and bool(self.cuvid_decoders)


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def probe_ffmpeg(selftest: bool = True) -> FFmpegInfo:
    """Inspect the system ffmpeg for NVDEC capability.

    If ``selftest`` and the build looks NVDEC-capable, run a real
    encode→hardware-decode roundtrip to confirm the driver actually works.
    """
    path = ffmpeg_path()
    if path is None:
        return FFmpegInfo(path=None, version=None)

    version = _run([path, "-hide_banner", "-version"]).stdout.splitlines()
    version = version[0] if version else None

    hwaccels = []
    out = _run([path, "-hide_banner", "-hwaccels"]).stdout
    for line in out.splitlines()[1:]:  # first line is a header
        tok = line.strip()
        if tok:
            hwaccels.append(tok)

    cuvid = []
    out = _run([path, "-hide_banner", "-decoders"]).stdout
    for line in out.splitlines():
        for name in line.split():
            if name.endswith("_cuvid"):
                cuvid.append(name)

    info = FFmpegInfo(
        path=path,
        version=version,
        hwaccels=hwaccels,
        cuvid_decoders=sorted(set(cuvid)),
    )

    if selftest and info.nvdec_capable:
        verified = selftest_nvdec(path)
        info = FFmpegInfo(
            path=path,
            version=version,
            hwaccels=hwaccels,
            cuvid_decoders=info.cuvid_decoders,
            nvdec_verified=verified,
        )
    return info


def selftest_nvdec(path: str | None = None) -> bool:
    """Encode a tiny H.264 clip, decode it via NVDEC. True iff both succeed.

    This is the only honest test of NVDEC: it exercises libnvcuvid through the
    driver. Returns False on any failure (no GPU, missing decoder, driver issue).
    """
    path = path or ffmpeg_path()
    if path is None:
        return False
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        enc = _run([
            path, "-hide_banner", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp.name,
        ])
        if enc.returncode != 0:
            return False
        dec = _run([
            path, "-hide_banner",
            "-hwaccel", "cuda", "-c:v", "h264_cuvid",
            "-i", tmp.name, "-f", "null", "-",
        ])
        return dec.returncode == 0


def recommended_decode_args(codec: str = "h264", info: FFmpegInfo | None = None) -> list[str]:
    """Input-side ffmpeg flags: NVDEC when available, else software fallback.

    Place these *before* the ``-i <input>`` argument. An empty list means "let
    ffmpeg pick the default software decoder".
    """
    info = info if info is not None else probe_ffmpeg(selftest=True)
    decoder = _CUVID_DECODER.get(codec.lower())
    if info.nvdec_verified and decoder and decoder in info.cuvid_decoders:
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder]
    return []
