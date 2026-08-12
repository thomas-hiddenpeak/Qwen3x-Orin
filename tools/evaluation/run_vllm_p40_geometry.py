#!/usr/bin/env python3
"""Run the one bounded stock-vLLM P40 macrochunk geometry witness.

This is deliberately not a general benchmark launcher.  It freezes the first
target-like whole-P40 scheduled-token budget selected by
WP-PREFILL-REFERENCE-TRANSLATION-v1 and fails closed around every source of
evidence drift: repository identity, corpus identity, server command,
clean-host ownership, warmup, external API result, Prometheus deltas, and
process-group cleanup.

The model directory and vLLM environment are external read-only inputs.  All
generated state, including Python/JIT/tool caches, is redirected below the
repository's ignored .q3x-work tree.  The Jetson nvidia-smi implementation is
never consulted; resource admission is delegated to orin_perf_preflight.py,
which uses tegrastats plus CPU/process and GPU-device-handle inspection.
"""

from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import datetime as dt
import email.parser
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK_ROOT = REPOSITORY_ROOT / ".q3x-work"
PREFLIGHT_TOOL = REPOSITORY_ROOT / "tools/evaluation/orin_perf_preflight.py"

MODEL_NAME = "qwen3.6-27b-nvfp4"
DEFAULT_MODEL_DIR = pathlib.Path(
    "/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4"
)
DEFAULT_VLLM_BIN = pathlib.Path("/home/rm01/vllmEvn/.venv/bin/vllm")
CONTROLLED_PATH = ":".join(
    (
        "/home/rm01/vllmEvn/.venv/bin",
        "/home/rm01/.local/bin",
        "/usr/local/cuda-13.3/bin",
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    )
)
EVALSCOPE_REQUIREMENT = "evalscope[perf]==1.9.1"
EVALSCOPE_VERSION = "1.9.1"
UVX_VERSION = "uvx 0.11.23 (aarch64-unknown-linux-gnu)"
UVX_SHA256 = "31450c5331a5999983911d93174457184e121423b133a1e73c5b5039deff0617"
EVALSCOPE_BASE_PYTHON = pathlib.Path(
    "/home/rm01/.local/share/uv/python/"
    "cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
)
EVALSCOPE_BASE_PYTHON_SHA256 = (
    "026f181cf0cc0d61cf620ca505994e10ab157a314e9a7d9434af6448546397f5"
)
EVALSCOPE_PYTHON_VERSION = (3, 13, 14)
EVALSCOPE_DISTRIBUTION_COUNT = 120
EVALSCOPE_RECORD_ROWS = 25_522
EVALSCOPE_HASHED_RECORD_ROWS = 25_402
EVALSCOPE_DISTRIBUTION_MANIFEST_SHA256 = (
    "143be983a531b71585fc0d2ed47b82aae8e00541d607fa58dc91727c7a414374"
)
EVALSCOPE_SITE_PACKAGES_SHA256 = (
    "65afd31f1670d288ef6d74f8366bbcf68c35490c0b3091315b0c31f7d141516f"
)
EVALSCOPE_SITE_PACKAGES_FILES = 25_473
EVALSCOPE_SITE_PACKAGES_BYTES = 812_725_950
EVALSCOPE_EXECUTABLE_SHA256 = (
    "1d07b710954b6102123ad63ae63c4b782799d06c00f6e06bd08903f4101fcb89"
)
VLLM_VERSION = "0.26.0"
VLLM_BINARY_SHA256 = (
    "52c6299affc4b84979fb11b9537e531b055a532bba54ee0e894d1e910014936d"
)

VLLM_SOURCE_SHA256 = {
    "_C_stable_libtorch.abi3.so": (
        "9eb23b42e054be399242372b84cf8f14caf901dd298edd66779f313411236c5a"
    ),
    "_custom_ops.py": (
        "16c78d17be2779361043e7a3a121bbc2d6d72cb631f5c267961bbd5971ca5336"
    ),
    "config/vllm.py": (
        "d4c31282c664c91bee81267a2134a9609d16c33e88cc40e0f020256346510a87"
    ),
    "engine/arg_utils.py": (
        "e46d992fb2553f7ecd71d9b64459ecbc7daa740305b19e862493389d1b68d11a"
    ),
    "entrypoints/openai/completion/serving.py": (
        "2590281b465d8a9e31252bd92407661f3af9adeddf7bd52c80395cc9fbb9e19c"
    ),
    "config/scheduler.py": (
        "a816cf79a3e74ffc0984f9bebb274275b26f46be8b28cb77a29388a0996263c8"
    ),
    "model_executor/kernels/linear/__init__.py": (
        "22cb0fb48a24ab4d00d0f05b27f0a4ea7db2abc345559de0b9ca5660f6833031"
    ),
    "model_executor/kernels/linear/nvfp4/marlin.py": (
        "6dd9892f215049f24bd6479cca32463a6256baba5957b3ed549e35f210502e06"
    ),
    "model_executor/kernels/linear/scaled_mm/marlin.py": (
        "eb4f0c3494d320ed71cc06ecff40eafa6695bbe60d10c178d67df0484bdf0b8f"
    ),
    "model_executor/models/qwen3_5.py": (
        "da738a2017741b8df66cbdb2ce5301c5cdec87ffacb2279f85245640eb33c496"
    ),
    "model_executor/models/qwen3_next.py": (
        "b642c10eb68978ca0df25f92ea866add08b31e654d99e78eaac0195d8bc6c74b"
    ),
    "model_executor/models/registry.py": (
        "27f6675a032ba82780c4b6038c05e6aad3988879b44f3582f6d5a273847da2f0"
    ),
    "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py": (
        "1227d6f385a52296e9f08223544b1c5fdc7e8d9aa09a848e7a8e522a8dc51214"
    ),
    "model_executor/layers/quantization/modelopt.py": (
        "6732d6c46718f9f0a98ca1fd04ed730ac299c4f506b62adfd5029c8112a93207"
    ),
    "model_executor/layers/quantization/utils/marlin_utils_fp8.py": (
        "c8751930b2d9e9d79451c410b1094e7bd7ee364d554b2e2dea5f2a19cf3fc17f"
    ),
    "platforms/cuda.py": (
        "ea6cade0bc560dcafea09bef8d30249b949e322ed8696ff02c6d20fa250640d2"
    ),
    "third_party/deep_gemm/__init__.py": (
        "0e563c954d08b1fcf53f77c6ba403df060fa88571e0959cfab721645ef6c8ae9"
    ),
    "utils/deep_gemm.py": (
        "c7adac1775842e9b0d5a1be9a323abb2840b5a93cc2a02d4f232cc3cfb0685a4"
    ),
    "v1/core/sched/scheduler.py": (
        "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"
    ),
    "v1/metrics/loggers.py": (
        "d5a91626cf79118f98ae6438c842cb30fc44ca14463c7e2af5871e3da8e51bcf"
    ),
    "v1/worker/gpu_model_runner.py": (
        "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0"
    ),
}

PACKAGE_METADATA_SHA256 = {
    "flashinfer_python-0.6.14.dist-info/METADATA": (
        "09f1467a9d9eeee956fc55ad5ecdc25364249fc51502bbe10492ae36de27e35c"
    ),
    "flashinfer_python-0.6.14.dist-info/RECORD": (
        "1958ef748461a095f0be0005339706362f8461475cead3d1c65ccfd824042437"
    ),
    "triton-3.6.0.dist-info/METADATA": (
        "5c9071db9b057bec89cd6d7f76f61e5f28a5e8d3717ff6c85420cf8015ee0321"
    ),
    "triton-3.6.0.dist-info/RECORD": (
        "761282d387bcc1358d0ecef39feb46c8dae673d06d1480e36170b5c8586f9ce1"
    ),
    "torch-2.11.0.dist-info/METADATA": (
        "d65e0ab5a65ced0dce799a2f6f32bff57b3d3e9d050069f607ef71eb24dd9976"
    ),
    "torch-2.11.0.dist-info/RECORD": (
        "eaf8be311fe9178aadd5ebc7be49f7d877c4ac0d397d3326bf1f73c6b0e47130"
    ),
    "vllm-0.26.0.dist-info/METADATA": (
        "3d4bb60795893f064bf3d3b9dce702ff853131e9cc489047762dff6959215cf1"
    ),
    "vllm-0.26.0.dist-info/RECORD": (
        "219c106b78d05a434786b931eddce17e751b05654bd903d68aeeec9adea8b2df"
    ),
}

PACKAGE_TREE_SHA256 = {
    "flashinfer": (
        "b1a78b966855afb6c6857138a91d6f0182abac2ba75c728dc9733396383ad1fc"
    ),
    "triton": (
        "4d4b0b5cf501d69aaa8e05f3cafc907a3e0740b1c3b24b280079b94864dfadaf"
    ),
    "vllm/third_party/flash_linear_attention": (
        "4a9e4f50d57f8711a40ded23125323be1e8fa83eb7b0fb1ca385d0b0df72a5aa"
    ),
}

MODEL_FILE_SHA256 = {
    "config.json": (
        "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"
    ),
    "hf_quant_config.json": (
        "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1"
    ),
    "model.safetensors.index.json": (
        "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"
    ),
    "model-00001-of-00003.safetensors": (
        "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d"
    ),
    "model-00002-of-00003.safetensors": (
        "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d"
    ),
    "model-00003-of-00003.safetensors": (
        "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
}

JETSON_MODEL = "NVIDIA Jetson AGX Orin Developer Kit"
JETSON_ARCHITECTURE = "aarch64"
JETSON_COMPUTE_CAPABILITY = (8, 7)
JETSON_MULTIPROCESSOR_COUNT = 16
JETSON_TOTAL_DEVICE_MEMORY = 65_932_095_488
JETSON_ONLINE_CPUS = tuple(range(12))
JETSON_CPU_KHZ = 2_201_600
JETSON_GPU_HZ = 1_300_500_000
JETSON_EMC_HZ = 3_200_000_000
JETSON_MAX_TEMPERATURE_MILLIC = 70_000
THERMAL_COOLDOWN_TARGET_MILLIC = 65_000
THERMAL_COOLDOWN_STABLE_SAMPLES = 3
THERMAL_COOLDOWN_INTERVAL_SECONDS = 5.0
THERMAL_COOLDOWN_TIMEOUT_SECONDS = 900.0
JETSON_L4T_SHA256 = "626bc5b28a18d7cab5f50ed10d436181991f29fa9d41a1ee359c900cbed5704b"
JETSON_DRIVER_SHA256 = (
    "ca994a48158f4a3a851abdec36a32978d7fe3798c15cd536e5b5fb07709a5311"
)
CUDA_VERSION_JSON_SHA256 = (
    "d2454ef27430ce41a2b7f1c67b7779a1d718994fbe3f830ee5c898373b65ab94"
)
TORCH_VERSION = "2.11.0+cu130"
TORCH_CUDA_VERSION = "13.0"
TORCH_GIT_VERSION = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
TORCH_DRIVER_API_VERSION = 13_020
TORCH_RUNTIME_API_VERSION = 13_000
VLLM_RUNTIME_FILE_SHA256 = {
    "torch/_C.cpython-313-aarch64-linux-gnu.so": (
        "8fcd72c89a7f8ceb3fb106829dca55295928a904eded268d953bf0fa26088e31"
    ),
    "torch/lib/libtorch.so": (
        "ab07676deb8234b91ece5b8674ff30b600449acef88d56f3f719abd668771723"
    ),
    "torch/lib/libtorch_cpu.so": (
        "af8453bb5bfffe061a27b21975467473b8f3e4daa86403c6d1259a7283e3ddb1"
    ),
    "torch/lib/libtorch_cuda.so": (
        "b62f38ee4d6dc06568f566a404dcd6bbbe9172f4f3425ef8543baeb669a40c93"
    ),
    "torch/lib/libc10.so": (
        "6d7c560348f342ed6704ee26883e1c9fe9f5dca30db3bea41678f34b85374a13"
    ),
    "torch/lib/libc10_cuda.so": (
        "f6e98ec41998c434ad5a9a459e2253bb427bdfe956f51cfa824bcfa1f1880ff0"
    ),
    "nvidia/cu13/lib/libcudart.so.13": (
        "7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63"
    ),
    "nvidia/cu13/lib/libnvJitLink.so.13": (
        "2ad249d2d60c87ea4941f5ce7ce1b06d4096032e95381c237f929cdaa5c7d5cc"
    ),
    "vllm/third_party/deep_gemm/_C.cpython-313-aarch64-linux-gnu.so": (
        "cc1b104f24d4e90e3fd1007f51bfa2e60c168efe59d92669ba4193f12c34a51c"
    ),
}
VLLM_RUNTIME_BUILD_IDS = {
    "torch/_C.cpython-313-aarch64-linux-gnu.so": (
        "6384f5a3865a934ecd6cfc49a7dcce1aab7421da"
    ),
    "torch/lib/libtorch.so": "74a64d3785e8c3ab9cb51ec389ed9798923c815a",
    "torch/lib/libtorch_cpu.so": "1434d2c1b44c70583659347a2774d959d9e58321",
    "torch/lib/libtorch_cuda.so": "5eb8db468b8f3ba9a5698a4821bd70c568a6be30",
    "torch/lib/libc10.so": "808f6b663a7d57301cbf5659efe6682174f7435b",
    "torch/lib/libc10_cuda.so": "67359b9fbfbd743d8094de6c966509a134fafaa9",
    "nvidia/cu13/lib/libnvJitLink.so.13": (
        "80c2ce4f5c7fa22c8818122581c503d58ccff183"
    ),
    "vllm/third_party/deep_gemm/_C.cpython-313-aarch64-linux-gnu.so": (
        "b8c610d1ab3c2fbc32983869d3922bf0e20daf4f"
    ),
}
LIBCUDA_PATH = pathlib.Path("/opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1.1")
LIBCUDA_SHA256 = "108623be4166cea6b75c4b959e3d6d3747f8bc48cc1d6d63c9f4e9effe2422c4"
LIBCUDA_BUILD_ID = "87eafd0170cb0821a561662038e724ec41b1cbb0"
VLLM_STABLE_EXTENSION_BUILD_ID = "67c60d7e22235d3c26ca906d0d2229c29d1c907e"

DEVICE_TREE_MODEL = pathlib.Path("/proc/device-tree/model")
L4T_RELEASE_FILE = pathlib.Path("/etc/nv_tegra_release")
NVIDIA_DRIVER_FILE = pathlib.Path("/proc/driver/nvidia/version")
CUDA_VERSION_FILE = pathlib.Path("/usr/local/cuda-13.3/version.json")
NVP_MODEL_TOOL = pathlib.Path("/usr/sbin/nvpmodel")
JETSON_CLOCKS_TOOL = pathlib.Path("/usr/bin/jetson_clocks")
GPU_DEVFREQ_ROOT = pathlib.Path(
    "/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu"
)
EMC_DEVFREQ_ROOT = pathlib.Path("/sys/devices/platform/bwmgr/devfreq/bwmgr")
CPUFREQ_ROOT = pathlib.Path("/sys/devices/system/cpu/cpufreq")
THERMAL_ROOT = pathlib.Path("/sys/class/thermal")
TEGRASTATS_PATH = pathlib.Path("/usr/bin/tegrastats")
TEGRASTATS_SHA256 = (
    "873419eb4fed218fd565a311905d58b192f372eec3defe6e77e1ffb38de558c0"
)
TEGRASTATS_INTERVAL_MS = 250
TEGRASTATS_MINIMUM_MEASURED_SAMPLES = 3
TEGRASTATS_POST_REQUEST_SAMPLES = 2
TEGRASTATS_RAM_TOTAL_MB = 62_878
TEGRASTATS_GPU_MHZ = 1_300
TEGRASTATS_EMC_MHZ = 3_200
TEGRASTATS_CPU_MHZ = 2_201
ZMQ_IPC_PATH_MAX_BYTES = 107
ZMQ_UUID_TEXT_BYTES = 36
VLLM_RPC_SOCKET_NAME = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
DELETED_POSIX_SEMAPHORE = re.compile(
    r"/dev/shm/sem\.[A-Za-z0-9_-]+ \(deleted\)"
)

PROMPT_TOKENS = 40_000
OUTPUT_TOKENS = 1
MACROCHUNK_BUDGETS = (40_000,)
DEFERRED_EXPLANATORY_BUDGETS = (8_192, 4_096, 2_048)
MEASURED_CORPUS = (
    WORK_ROOT
    / "evalscope/corpora/q3x-repository-agent-context-p40000-one-token.jsonl"
)
MEASURED_CORPUS_SHA256 = (
    "8970ac50693f49d1b27d35a0610ecbe5072594330d69b301f4dab731789b6844"
)
WARMUP_CORPUS = (
    WORK_ROOT
    / "evalscope/corpora/q3x-sharegpt-ordinal4-cycled-p40000-one-token.jsonl"
)
WARMUP_CORPUS_SHA256 = (
    "42e15a4ec36070a25f53d0237b20800d83213de47cf3aee2dca3be1686748732"
)

DIRECTORY_ENVIRONMENT_KEYS = (
    "HOME",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "UV_CACHE_DIR",
    "PIP_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "MODELSCOPE_CACHE",
    "MS_CACHE_HOME",
    "EVALSCOPE_CACHE",
    "PYTHONPYCACHEPREFIX",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "MPLCONFIGDIR",
    "CUPY_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "VLLM_CACHE_ROOT",
    "VLLM_CONFIG_ROOT",
    "VLLM_RPC_BASE_PATH",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "CUTE_DSL_CACHE_DIR",
    "HUMMING_CACHE_DIR",
    "HUMMING_TMP_DIR",
)

# Preserve only user identity, locale, and the per-user runtime directory.
# HOME, dynamic-loader/tool discovery, route, compiler, Python, and tuning
# knobs are controlled explicitly rather than inherited from the launcher.
BASE_ENVIRONMENT_KEYS = (
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "XDG_RUNTIME_DIR",
    "XDG_DATA_DIRS",
)

COMPILATION_CACHE_ENVIRONMENT_KEYS = (
    "HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "CUPY_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "VLLM_CACHE_ROOT",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "CUTE_DSL_CACHE_DIR",
    "HUMMING_CACHE_DIR",
    "HUMMING_TMP_DIR",
    "PYTHONPYCACHEPREFIX",
)

PROMETHEUS_SAMPLE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+"
    r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|"
    r"NaN|[+-]?Inf)(?:\s+[-+]?[0-9]+)?$"
)
GR3D_SAMPLE = re.compile(r"(?:^|\s)GR3D_FREQ\s+([0-9]+(?:\.[0-9]+)?)%")
GR3D_CLOCK_SAMPLE = re.compile(
    r"(?:^|\s)GR3D_FREQ\s+[0-9]+(?:\.[0-9]+)?%"
    r"@\[([0-9]+),([0-9]+)\]"
)
EMC_CLOCK_SAMPLE = re.compile(
    r"(?:^|\s)EMC_FREQ\s+[0-9]+(?:\.[0-9]+)?%@([0-9]+)"
)
CPU_SAMPLE = re.compile(r"(?:^|\s)CPU\s+\[([^\]]+)\]")
CPU_ENTRY = re.compile(r"[0-9]+(?:\.[0-9]+)?%@([0-9]+)")
RAM_SAMPLE = re.compile(r"(?:^|\s)RAM\s+[0-9]+/([0-9]+)MB")
THERMAL_SAMPLE = re.compile(
    r"(?:^|\s)([A-Za-z0-9_-]+)@([0-9]+(?:\.[0-9]+)?)C/"
    r"([0-9]+(?:\.[0-9]+)?)C"
)


class GeometryError(RuntimeError):
    """The witness cannot produce admissible evidence."""


@dataclasses.dataclass(frozen=True)
class GeometryConfig:
    output_dir: pathlib.Path
    model_dir: pathlib.Path
    vllm_bin: pathlib.Path
    uvx_bin: pathlib.Path
    port: int
    allow_pids: tuple[int, ...]
    readiness_timeout_seconds: float
    request_timeout_seconds: float
    preflight_samples: int
    preflight_interval_ms: int
    dry_run: bool


@dataclasses.dataclass(frozen=True)
class EvalScopeRuntime:
    prefix: pathlib.Path
    python: pathlib.Path
    executable: pathlib.Path
    receipt: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class PrometheusSnapshot:
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_is_within(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def validate_output_dir(path: pathlib.Path, *, may_exist: bool = False) -> pathlib.Path:
    output = path.expanduser().resolve()
    work = WORK_ROOT.resolve()
    if not path_is_within(output, work) or output == work:
        raise GeometryError("--output-dir must name a child of repository .q3x-work")
    if output.exists() and not output.is_dir():
        raise GeometryError("--output-dir exists and is not a directory")
    if output.exists() and any(output.iterdir()) and not may_exist:
        raise GeometryError("--output-dir must not already contain artifacts")
    return output


def validate_corpus(path: pathlib.Path, expected_sha256: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise GeometryError(f"cannot read corpus {path}: {error}") from error
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise GeometryError(
            f"corpus hash mismatch for {path}: {observed_sha256}"
        )
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GeometryError(f"corpus must contain exactly one request: {path}")
    try:
        request = json.loads(lines[0])
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GeometryError(f"invalid corpus JSON {path}: {error}") from error
    prompt = request.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != PROMPT_TOKENS
        or any(not isinstance(token, int) or isinstance(token, bool) for token in prompt)
    ):
        raise GeometryError(f"corpus is not an exact P{PROMPT_TOKENS} token-ID request")
    expected_fields = {
        "model": MODEL_NAME,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
    }
    for field, expected in expected_fields.items():
        if request.get(field) != expected:
            raise GeometryError(
                f"corpus field {field!r} is {request.get(field)!r}, expected {expected!r}"
            )
    return {
        "path": str(path.resolve()),
        "sha256": observed_sha256,
        "prompt_tokens": len(prompt),
        "max_tokens": request["max_tokens"],
    }


def _parse_prometheus_labels(raw: str) -> tuple[tuple[str, str], ...]:
    labels: list[tuple[str, str]] = []
    cursor = 0
    while cursor < len(raw):
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        key_match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", raw[cursor:])
        if key_match is None:
            raise GeometryError(f"invalid Prometheus label set: {raw!r}")
        key = key_match.group(0)
        cursor += len(key)
        if cursor >= len(raw) or raw[cursor] != "=":
            raise GeometryError(f"invalid Prometheus label assignment: {raw!r}")
        cursor += 1
        if cursor >= len(raw) or raw[cursor] != '"':
            raise GeometryError(f"Prometheus label is not quoted: {raw!r}")
        cursor += 1
        encoded: list[str] = []
        escaped = False
        while cursor < len(raw):
            character = raw[cursor]
            cursor += 1
            if escaped:
                encoded.append("\\" + character)
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                break
            else:
                encoded.append(character)
        else:
            raise GeometryError(f"unterminated Prometheus label: {raw!r}")
        try:
            value = json.loads('"' + "".join(encoded) + '"')
        except json.JSONDecodeError as error:
            raise GeometryError(f"invalid Prometheus label escape: {raw!r}") from error
        labels.append((key, value))
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor == len(raw):
            break
        if raw[cursor] != ",":
            raise GeometryError(f"invalid Prometheus label separator: {raw!r}")
        cursor += 1
    if len({key for key, _ in labels}) != len(labels):
        raise GeometryError(f"duplicate Prometheus label: {raw!r}")
    return tuple(sorted(labels))


def parse_prometheus(text: str) -> PrometheusSnapshot:
    samples: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE.fullmatch(line)
        if match is None:
            raise GeometryError(f"invalid Prometheus sample at line {line_number}")
        name = match.group(1)
        labels = _parse_prometheus_labels(match.group(2) or "")
        value = float(match.group(3))
        key = (name, labels)
        if key in samples:
            raise GeometryError(f"duplicate Prometheus sample at line {line_number}")
        samples[key] = value
    if not samples:
        raise GeometryError("Prometheus snapshot contains no samples")
    return PrometheusSnapshot(samples)


def metric_total(
    snapshot: PrometheusSnapshot,
    name: str,
    required_labels: Mapping[str, str] | None = None,
) -> float:
    required = dict(required_labels or {})
    values: list[float] = []
    for (sample_name, labels), value in snapshot.samples.items():
        if sample_name != name:
            continue
        label_map = dict(labels)
        if any(label_map.get(key) != expected for key, expected in required.items()):
            continue
        if not math.isfinite(value):
            raise GeometryError(f"metric {name} contains a non-finite value")
        values.append(value)
    if not values:
        suffix = f" labels={required}" if required else ""
        raise GeometryError(f"required metric is absent: {name}{suffix}")
    return sum(values)


def _metric_delta(
    before: PrometheusSnapshot,
    after: PrometheusSnapshot,
    name: str,
    required_labels: Mapping[str, str] | None = None,
) -> float:
    return metric_total(after, name, required_labels) - metric_total(
        before, name, required_labels
    )


def _require_delta(name: str, observed: float, expected: float) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6):
        raise GeometryError(
            f"metric delta {name} is {observed}, expected exactly {expected}"
        )


def validate_metric_delta(
    before: PrometheusSnapshot, after: PrometheusSnapshot
) -> dict[str, float]:
    deltas = {
        "prefill_phase_seconds": _metric_delta(
            before, after, "vllm:request_prefill_time_seconds_sum"
        ),
        "prefill_request_count": _metric_delta(
            before, after, "vllm:request_prefill_time_seconds_count"
        ),
        "prefill_kv_tokens": _metric_delta(
            before, after, "vllm:request_prefill_kv_computed_tokens_sum"
        ),
        "prefill_kv_request_count": _metric_delta(
            before, after, "vllm:request_prefill_kv_computed_tokens_count"
        ),
        "request_prompt_tokens": _metric_delta(
            before, after, "vllm:request_prompt_tokens_sum"
        ),
        "request_prompt_count": _metric_delta(
            before, after, "vllm:request_prompt_tokens_count"
        ),
        "local_compute_tokens": _metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            {"source": "local_compute"},
        ),
        "local_cache_hit_tokens": _metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            {"source": "local_cache_hit"},
        ),
        "external_kv_transfer_tokens": _metric_delta(
            before,
            after,
            "vllm:prompt_tokens_by_source_total",
            {"source": "external_kv_transfer"},
        ),
        "cached_prompt_tokens": _metric_delta(
            before, after, "vllm:prompt_tokens_cached_total"
        ),
        "successful_requests": _metric_delta(
            before, after, "vllm:request_success_total"
        ),
        "generation_tokens": _metric_delta(
            before, after, "vllm:generation_tokens_total"
        ),
        "preemptions": _metric_delta(before, after, "vllm:num_preemptions_total"),
        "queue_seconds": _metric_delta(
            before, after, "vllm:request_queue_time_seconds_sum"
        ),
    }
    for name in (
        "prefill_request_count",
        "prefill_kv_request_count",
        "request_prompt_count",
        "successful_requests",
        "generation_tokens",
    ):
        _require_delta(name, deltas[name], 1.0)
    for name in ("prefill_kv_tokens", "request_prompt_tokens", "local_compute_tokens"):
        _require_delta(name, deltas[name], float(PROMPT_TOKENS))
    for name in (
        "local_cache_hit_tokens",
        "external_kv_transfer_tokens",
        "cached_prompt_tokens",
        "preemptions",
    ):
        _require_delta(name, deltas[name], 0.0)
    if deltas["prefill_phase_seconds"] <= 0.0:
        raise GeometryError("server Prefill-phase span is not positive")
    if deltas["queue_seconds"] < 0.0:
        raise GeometryError("server queue-time delta is negative")
    deltas["server_prefill_phase_tokens_per_second"] = (
        PROMPT_TOKENS / deltas["prefill_phase_seconds"]
    )
    return deltas


def validate_evalscope_summary(path: pathlib.Path) -> dict[str, float]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GeometryError(f"cannot read EvalScope summary {path}: {error}") from error
    expected = {
        "Total Requests": 1.0,
        "Success Requests": 1.0,
        "Failed Requests": 0.0,
        "Avg Input Tokens": float(PROMPT_TOKENS),
        "Avg Output Tokens": float(OUTPUT_TOKENS),
    }
    normalized: dict[str, float] = {}
    for field, wanted in expected.items():
        value = summary.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GeometryError(f"EvalScope summary field is missing: {field}")
        observed = float(value)
        _require_delta(f"EvalScope {field}", observed, wanted)
        normalized[field] = observed
    ttft = summary.get("TTFT (ms)")
    latency = summary.get("Avg Latency (s)")
    if (
        not isinstance(ttft, (int, float))
        or isinstance(ttft, bool)
        or not math.isfinite(float(ttft))
        or ttft <= 0
    ):
        raise GeometryError("EvalScope TTFT is absent or non-positive")
    if (
        not isinstance(latency, (int, float))
        or isinstance(latency, bool)
        or not math.isfinite(float(latency))
        or latency <= 0
    ):
        raise GeometryError("EvalScope latency is absent or non-positive")
    normalized["TTFT (ms)"] = float(ttft)
    normalized["Avg Latency (s)"] = float(latency)
    normalized["prompt_tokens_per_ttft_second"] = (
        PROMPT_TOKENS / (float(ttft) / 1_000.0)
    )
    return normalized


def build_prefill_metric_surfaces(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the three Prefill observations separate and explicitly scoped."""
    evalscope = result.get("evalscope")
    metric_delta = result.get("server_metric_delta")
    log_observations = result.get("server", {}).get("final_log_observations")
    return {
        "logger_window": {
            "authority": "supporting_telemetry_only",
            "samples": (
                log_observations.get("logger_interval_samples", [])
                if isinstance(log_observations, Mapping)
                else []
            ),
            "maximum_prompt_tokens_per_second": (
                log_observations.get(
                    "logger_interval_prompt_throughput_maximum"
                )
                if isinstance(log_observations, Mapping)
                else None
            ),
        },
        "request_prefill": {
            "authority": "request_bound_server_evidence",
            "tokens_per_second": (
                metric_delta.get("server_prefill_phase_tokens_per_second")
                if isinstance(metric_delta, Mapping)
                else None
            ),
            "prefill_phase_seconds": (
                metric_delta.get("prefill_phase_seconds")
                if isinstance(metric_delta, Mapping)
                else None
            ),
        },
        "external_ttft": {
            "authority": "product_visible_external_result",
            "ttft_milliseconds": (
                evalscope.get("TTFT (ms)")
                if isinstance(evalscope, Mapping)
                else None
            ),
            "prompt_tokens_per_ttft_second": (
                evalscope.get("prompt_tokens_per_ttft_second")
                if isinstance(evalscope, Mapping)
                else None
            ),
        },
        "comparison_contract": (
            "logger_window, request_prefill, and external_ttft are distinct "
            "metric surfaces and are not expected to be numerically equal"
        ),
    }


def build_server_command(config: GeometryConfig, budget: int) -> list[str]:
    command = [
        str(config.vllm_bin),
        "serve",
        str(config.model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--served-model-name",
        MODEL_NAME,
        "--max-model-len",
        "40001",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(budget),
        "--gpu-memory-utilization",
        "0.78",
        "--kv-cache-dtype",
        "bfloat16",
        "--mamba-cache-dtype",
        "bfloat16",
        "--mamba-ssm-cache-dtype",
        "bfloat16",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--attention-backend",
        "FLASHINFER",
        "--mamba-backend",
        "TRITON",
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    ]
    forbidden = {"--linear-backend", "--speculative-config", "--spec-config"}
    if any(argument in forbidden for argument in command):
        raise GeometryError("stock reference command contains a forbidden override")
    return command


def build_evalscope_command(
    config: GeometryConfig,
    corpus: pathlib.Path,
    outputs_dir: pathlib.Path,
    name: str,
    evalscope_bin: pathlib.Path | None = None,
) -> list[str]:
    command = [
        str(evalscope_bin or config.uvx_bin),
    ]
    if evalscope_bin is None:
        command.extend(
            (
                "--offline",
                "--no-config",
                "--no-python-downloads",
                "--isolated",
                "--python",
                str(EVALSCOPE_BASE_PYTHON),
                "--from",
                EVALSCOPE_REQUIREMENT,
                "evalscope",
            )
        )
    command.extend(
        (
            "perf",
            "--model",
            MODEL_NAME,
            "--api",
            "openai",
            "--url",
            f"http://127.0.0.1:{config.port}/v1/completions",
            "--tokenizer-path",
            str(config.model_dir),
            "--dataset",
            "line_by_line",
            "--data-source",
            "local",
            "--dataset-path",
            str(corpus),
            "--number",
            "1",
            "--parallel",
            "1",
            "--warmup-num",
            "0",
            "--num-workers",
            "1",
            "--max-prompt-length",
            "131072",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--seed",
            "42",
            "--stream",
            "--tokenize-prompt",
            "--no-test-connection",
            "--total-timeout",
            str(int(config.request_timeout_seconds)),
            "--outputs-dir",
            str(outputs_dir),
            "--name",
            name,
            "--no-timestamp",
        )
    )
    return command


def environment_overrides(budget: int) -> dict[str, str]:
    cache = WORK_ROOT / "cache"
    configuration = WORK_ROOT / "config"
    data = WORK_ROOT / "data"
    home = WORK_ROOT / "home"
    temporary = WORK_ROOT / "tmp"
    rpc_base = WORK_ROOT.resolve()
    rpc_probe = rpc_base / ("0" * ZMQ_UUID_TEXT_BYTES)
    if len(os.fsencode(str(rpc_probe))) > ZMQ_IPC_PATH_MAX_BYTES:
        raise GeometryError(
            "repository .q3x-work path is too long for vLLM ZeroMQ IPC: "
            f"{len(os.fsencode(str(rpc_probe)))} > {ZMQ_IPC_PATH_MAX_BYTES}"
        )
    return {
        "HOME": str(home),
        "PATH": CONTROLLED_PATH,
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.3/lib64",
        "CUDA_HOME": "/usr/local/cuda-13.3",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "FLASHINFER_NO_DOWNLOAD": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_DO_NOT_TRACK": "1",
        "TMPDIR": str(temporary),
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(configuration),
        "XDG_DATA_HOME": str(data),
        "UV_CACHE_DIR": str(cache / "uv"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "HF_HOME": str(cache / "hf"),
        "HF_HUB_CACHE": str(cache / "hf/hub"),
        "HF_DATASETS_CACHE": str(cache / "hf/datasets"),
        "TRANSFORMERS_CACHE": str(cache / "hf/transformers"),
        "SENTENCE_TRANSFORMERS_HOME": str(cache / "hf/sentence-transformers"),
        "MODELSCOPE_CACHE": str(cache / "modelscope/hub"),
        "MS_CACHE_HOME": str(cache / "modelscope"),
        "EVALSCOPE_CACHE": str(cache / "evalscope"),
        "PYTHONPYCACHEPREFIX": str(cache / "python-pycache-disabled"),
        "TORCH_HOME": str(cache / "torch"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "CUPY_CACHE_DIR": str(cache / "cupy"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "VLLM_CONFIG_ROOT": str(cache / "vllm/config"),
        # vLLM appends a 36-byte UUID.  The nested Codex worktree leaves only
        # enough sockaddr_un space for the .q3x-work root itself; using it
        # keeps every socket in the workspace without falling back to /tmp.
        "VLLM_RPC_BASE_PATH": str(rpc_base),
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": str(cache / "flashinfer-autotune"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "CUTE_DSL_CACHE_DIR": str(cache / "cute-dsl"),
        "HUMMING_CACHE_DIR": str(cache / "humming"),
        "HUMMING_TMP_DIR": str(temporary / "humming"),
    }


def build_environment(budget: int, *, create: bool) -> dict[str, str]:
    overrides = environment_overrides(budget)
    if create:
        for key in DIRECTORY_ENVIRONMENT_KEYS:
            pathlib.Path(overrides[key]).mkdir(parents=True, exist_ok=True)
    environment = {
        key: os.environ[key] for key in BASE_ENVIRONMENT_KEYS if key in os.environ
    }
    environment.update(overrides)
    return environment


def environment_receipt(budget: int) -> dict[str, Any]:
    effective = build_environment(budget, create=False)
    return {
        "base_allowlist": list(BASE_ENVIRONMENT_KEYS),
        "preserved_base": {
            key: effective[key]
            for key in BASE_ENVIRONMENT_KEYS
            if key in effective
        },
        "controlled_overrides": environment_overrides(budget),
        "pythonpath_inherited": False,
        "virtual_env_inherited": False,
        "home_inherited": False,
    }


def validate_empty_workspace_directory(
    path: pathlib.Path, identity_name: str
) -> dict[str, Any]:
    """Require one dedicated workspace directory to contain no entries."""

    if (
        not path_is_within(path, WORK_ROOT)
        or path.is_symlink()
        or path.absolute() != path.resolve()
        or not path.is_dir()
    ):
        raise GeometryError(f"{identity_name} is not a sealed workspace directory: {path}")
    try:
        entries = sorted(entry.name for entry in path.iterdir())
    except OSError as error:
        raise GeometryError(f"cannot inspect {identity_name} {path}: {error}") from error
    if entries:
        preview = entries[:20]
        raise GeometryError(
            f"{identity_name} must be empty before and after execution: {preview}"
        )
    return {"path": str(path.resolve()), "empty": True}


def validate_empty_pycache_prefix(budget: int) -> dict[str, Any]:
    path = pathlib.Path(environment_overrides(budget)["PYTHONPYCACHEPREFIX"])
    return validate_empty_workspace_directory(path, "alternate Python bytecode prefix")


def snapshot_compilation_caches(budget: int) -> dict[str, Any]:
    overrides = environment_overrides(budget)
    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for key in COMPILATION_CACHE_ENVIRONMENT_KEYS:
        root = pathlib.Path(overrides[key])
        if not path_is_within(root, WORK_ROOT) or root.is_symlink() or not root.is_dir():
            raise GeometryError(f"compilation-cache root is not sealed: {key}={root}")
        try:
            entries = sorted(
                root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
            )
            for path in entries:
                path_stat = path.lstat()
                if stat.S_ISLNK(path_stat.st_mode):
                    raise GeometryError(f"compilation cache contains a symlink: {path}")
                if stat.S_ISDIR(path_stat.st_mode):
                    continue
                if not stat.S_ISREG(path_stat.st_mode):
                    raise GeometryError(
                        f"compilation cache contains a non-regular file: {path}"
                    )
                relative = path.relative_to(root).as_posix()
                files[f"{key}/{relative}"] = {
                    "bytes": path_stat.st_size,
                    "mtime_ns": path_stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                }
                total_bytes += path_stat.st_size
        except OSError as error:
            raise GeometryError(f"cannot attest compilation cache {root}: {error}") from error
    return {
        "keys": list(COMPILATION_CACHE_ENVIRONMENT_KEYS),
        "file_count": len(files),
        "bytes": total_bytes,
        "files": files,
    }


def write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_identity() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise GeometryError("formal witness requires a clean tracked worktree")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": status,
    }


def _run_logged(
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: pathlib.Path,
    timeout_seconds: float,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(environment),
            start_new_session=True,
            close_fds=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            stop_process_group(process)
            raise GeometryError(f"command timed out; see {log_path}") from error
        except BaseException:
            stop_process_group(process)
            raise
        cleanup = stop_process_group(process)
        if cleanup["signals"]:
            raise GeometryError(
                f"command left live process-group members; see {log_path}"
            )
        return returncode


def _process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise GeometryError(
            f"cannot prove ownership of process group {group}"
        ) from error
    return True


def _wait_for_process_group_empty(
    process: subprocess.Popen[bytes], group: int, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        # Reap the session leader as soon as it exits. A zombie remains a
        # process-group member, so killpg(group, 0) alone would falsely keep
        # the group alive until the full escalation timeout expires.
        process.poll()
        if not _process_group_exists(group):
            return True
        time.sleep(0.1)
    process.poll()
    return not _process_group_exists(group)


def stop_process_group(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    group = process.pid
    result: dict[str, Any] = {
        "pid": process.pid,
        "process_group": group,
        "signals": [],
        "group_empty": False,
    }
    if process.poll() is None:
        try:
            observed_group = os.getpgid(process.pid)
            observed_session = os.getsid(process.pid)
        except ProcessLookupError:
            process.poll()
        else:
            if observed_group != group or observed_session != group:
                raise GeometryError(
                    "refusing to signal a process that does not own its new session"
                )
    for sent_signal, timeout in (
        (signal.SIGINT, 60.0),
        (signal.SIGTERM, 20.0),
        (signal.SIGKILL, 10.0),
    ):
        if not _process_group_exists(group):
            break
        result["signals"].append(sent_signal.name)
        try:
            os.killpg(group, sent_signal)
        except ProcessLookupError:
            break
        if _wait_for_process_group_empty(process, group, timeout):
            break
    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    result["group_empty"] = not _process_group_exists(group)
    if not result["group_empty"]:
        raise GeometryError("process group did not terminate after SIGKILL")
    process.poll()
    result["returncode"] = process.returncode
    return result


def collect_vllm_rpc_sockets() -> list[dict[str, Any]]:
    """Return only UUID-named Unix sockets directly below .q3x-work."""

    root = WORK_ROOT.resolve()
    if WORK_ROOT.is_symlink() or not root.is_dir():
        raise GeometryError(f"vLLM RPC root is not a sealed directory: {WORK_ROOT}")
    records: list[dict[str, Any]] = []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise GeometryError(f"cannot inspect vLLM RPC root {root}: {error}") from error
    for path in entries:
        if VLLM_RPC_SOCKET_NAME.fullmatch(path.name) is None:
            continue
        try:
            observed = path.lstat()
        except OSError as error:
            raise GeometryError(f"cannot attest vLLM RPC entry {path}: {error}") from error
        if not stat.S_ISSOCK(observed.st_mode):
            raise GeometryError(f"UUID-named vLLM RPC entry is not a socket: {path}")
        records.append(
            {
                "path": str(path),
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "mode": observed.st_mode,
            }
        )
    return records


def cleanup_vllm_rpc_sockets() -> dict[str, Any]:
    """Remove attested vLLM UUID sockets after its process group is empty."""

    observed = collect_vllm_rpc_sockets()
    removed: list[dict[str, Any]] = []
    for record in observed:
        path = pathlib.Path(record["path"])
        try:
            current = path.lstat()
        except OSError as error:
            raise GeometryError(f"vLLM RPC socket changed before cleanup: {path}") from error
        if (
            not stat.S_ISSOCK(current.st_mode)
            or current.st_dev != record["device"]
            or current.st_ino != record["inode"]
        ):
            raise GeometryError(f"vLLM RPC socket identity changed: {path}")
        try:
            path.unlink()
        except OSError as error:
            raise GeometryError(f"cannot remove vLLM RPC socket {path}: {error}") from error
        removed.append(record)
    remaining = collect_vllm_rpc_sockets()
    if remaining:
        raise GeometryError(f"vLLM RPC sockets remain after cleanup: {remaining}")
    return {
        "root": str(WORK_ROOT.resolve()),
        "observed": observed,
        "removed": removed,
        "remaining": remaining,
    }


def fetch_text(url: str, timeout_seconds: float = 10.0) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise GeometryError(f"HTTP {response.status} from {url}")
        return response.read().decode("utf-8")


def wait_for_server(
    process: subprocess.Popen[bytes], port: int, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not attempted"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise GeometryError(f"vLLM exited before readiness: status={returncode}")
        try:
            fetch_text(f"http://127.0.0.1:{port}/health", 2.0)
            return
        except (GeometryError, OSError, UnicodeError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(2.0)
    raise GeometryError(f"vLLM readiness timed out: {last_error}")


def _parse_proc_process_identity(path: pathlib.Path) -> dict[str, int]:
    try:
        line = (path / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise GeometryError(f"cannot read process identity {path}: {error}") from error
    open_paren = line.find("(")
    close_paren = line.rfind(")")
    fields = line[close_paren + 1 :].strip().split()
    if open_paren <= 0 or close_paren <= open_paren or len(fields) < 20:
        raise GeometryError(f"malformed process identity: {path / 'stat'}")
    try:
        return {
            "pid": int(line[:open_paren].strip()),
            "ppid": int(fields[1]),
            "process_group": int(fields[2]),
            "session": int(fields[3]),
            "start_time_ticks": int(fields[19]),
        }
    except ValueError as error:
        raise GeometryError(f"non-integer process identity: {path / 'stat'}") from error


def expected_loaded_runtime_identities(
    config: GeometryConfig,
) -> dict[str, dict[str, str | None]]:
    vllm_root = resolve_vllm_package_root(config)
    site_packages = vllm_root.parent
    identities = {
        str((site_packages / relative).resolve()): {
            "sha256": sha256,
            "build_id": VLLM_RUNTIME_BUILD_IDS.get(relative),
        }
        for relative, sha256 in VLLM_RUNTIME_FILE_SHA256.items()
    }
    identities[str((vllm_root / "_C_stable_libtorch.abi3.so").resolve())] = {
        "sha256": VLLM_SOURCE_SHA256["_C_stable_libtorch.abi3.so"],
        "build_id": VLLM_STABLE_EXTENSION_BUILD_ID,
    }
    identities[str(LIBCUDA_PATH)] = {
        "sha256": LIBCUDA_SHA256,
        "build_id": LIBCUDA_BUILD_ID,
    }
    return identities


def collect_server_process_tree_runtime(
    config: GeometryConfig, root_pid: int
) -> dict[str, Any]:
    """Attest the live vLLM process tree immediately before measurement."""

    identities: dict[int, dict[str, int]] = {}
    try:
        proc_entries = sorted(
            (entry for entry in pathlib.Path("/proc").iterdir() if entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        )
    except OSError as error:
        raise GeometryError(f"cannot enumerate /proc for vLLM: {error}") from error
    for entry in proc_entries:
        try:
            identity = _parse_proc_process_identity(entry)
        except GeometryError:
            if not entry.exists():
                continue
            raise
        identities[identity["pid"]] = identity
    root = identities.get(root_pid)
    if root is None or root["process_group"] != root_pid or root["session"] != root_pid:
        raise GeometryError(f"vLLM session leader is absent or changed: {root_pid}")

    session_members = {
        pid for pid, identity in identities.items() if identity["session"] == root_pid
    }
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, identity in identities.items():
            if pid not in descendants and identity["ppid"] in descendants:
                descendants.add(pid)
                changed = True
    escaped_descendants = descendants - session_members
    if escaped_descendants:
        raise GeometryError(
            "vLLM descendants escaped the owned session: "
            f"{sorted(escaped_descendants)}"
        )
    expected_runtime_identities = expected_loaded_runtime_identities(config)
    expected_runtime_paths = set(expected_runtime_identities)
    expected_runtime_names = {
        pathlib.Path(path).name for path in expected_runtime_paths
    }
    observed_runtime_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    for pid in sorted(session_members):
        process_root = pathlib.Path("/proc") / str(pid)
        identity = identities[pid]
        if (
            identity["process_group"] != root_pid
            or identity["session"] != root_pid
        ):
            raise GeometryError(
                f"vLLM descendant escaped the owned session: pid={pid}, "
                f"pgrp={identity['process_group']}, session={identity['session']}"
            )
        try:
            affinity = tuple(sorted(os.sched_getaffinity(pid)))
            status = (process_root / "status").read_text(encoding="utf-8")
            executable = os.readlink(process_root / "exe")
            cmdline = (process_root / "cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode("utf-8", errors="replace").strip()
            maps = (process_root / "maps").read_text(encoding="utf-8")
            identity_after = _parse_proc_process_identity(process_root)
        except (OSError, UnicodeError) as error:
            raise GeometryError(
                f"cannot attest live vLLM descendant {pid}: {error}"
            ) from error
        if identity_after != identity:
            raise GeometryError(f"vLLM descendant changed during attestation: {pid}")
        allowed_match = re.search(r"(?m)^Cpus_allowed_list:\s*(\S+)\s*$", status)
        uid_match = re.search(r"(?m)^Uid:\s*(\d+)\s+", status)
        if allowed_match is None or allowed_match.group(1) != "0-11":
            raise GeometryError(f"vLLM descendant has unexpected CPU mask: pid={pid}")
        if uid_match is None or int(uid_match.group(1)) != os.getuid():
            raise GeometryError(f"vLLM descendant has unexpected UID: pid={pid}")
        if affinity != JETSON_ONLINE_CPUS:
            raise GeometryError(
                f"vLLM descendant affinity is not 0-11: pid={pid}, mask={affinity}"
            )
        thread_records: list[dict[str, Any]] = []
        for task in sorted(
            (entry for entry in (process_root / "task").iterdir() if entry.name.isdigit()),
            key=lambda entry: int(entry.name),
        ):
            tid = int(task.name)
            task_status = (task / "status").read_text(encoding="utf-8")
            task_match = re.search(
                r"(?m)^Cpus_allowed_list:\s*(\S+)\s*$", task_status
            )
            task_affinity = tuple(sorted(os.sched_getaffinity(tid)))
            if (
                task_match is None
                or task_match.group(1) != "0-11"
                or task_affinity != JETSON_ONLINE_CPUS
            ):
                raise GeometryError(
                    f"vLLM thread affinity is not 0-11: pid={pid}, tid={tid}"
                )
            thread_records.append(
                {
                    "tid": tid,
                    "cpu_affinity": list(task_affinity),
                    "cpus_allowed_list": task_match.group(1),
                }
            )
        loaded_records: list[dict[str, Any]] = []
        executable_cache_records: list[dict[str, Any]] = []
        deleted_posix_semaphore_records: list[dict[str, Any]] = []
        for line in maps.splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) < 5 or len(parts) == 5:
                continue
            permissions = parts[1]
            device_text = parts[3]
            try:
                mapped_inode = int(parts[4])
            except ValueError as error:
                raise GeometryError(f"invalid mapped inode for pid {pid}") from error
            mapped_path = parts[5]
            if mapped_path.endswith(" (deleted)"):
                if (
                    permissions != "rw-s"
                    or mapped_inode <= 0
                    or DELETED_POSIX_SEMAPHORE.fullmatch(mapped_path) is None
                ):
                    raise GeometryError(
                        f"vLLM maps a forbidden deleted file: pid={pid}, "
                        f"permissions={permissions}, path={mapped_path}"
                    )
                deleted_posix_semaphore_records.append(
                    {
                        "path": mapped_path,
                        "device": device_text,
                        "inode": mapped_inode,
                        "permissions": permissions,
                        "executable": False,
                        "kind": "unlinked_posix_semaphore",
                    }
                )
                continue
            path = pathlib.Path(mapped_path)
            is_core_runtime = path.name in expected_runtime_names
            is_executable_cache = "x" in permissions and path_is_within(
                path, WORK_ROOT / "cache"
            )
            if not is_core_runtime and not is_executable_cache:
                continue
            try:
                mapped_stat = path.stat()
                major_text, minor_text = device_text.split(":", 1)
                mapped_device = (int(major_text, 16), int(minor_text, 16))
            except (OSError, ValueError) as error:
                raise GeometryError(
                    f"cannot reconcile mapped file for pid {pid}: {mapped_path}"
                ) from error
            if (
                mapped_inode != mapped_stat.st_ino
                or mapped_device
                != (os.major(mapped_stat.st_dev), os.minor(mapped_stat.st_dev))
            ):
                raise GeometryError(
                    f"mapped file identity changed for pid {pid}: {mapped_path}"
                )
            record = {
                "path": str(path.resolve()),
                "device": device_text,
                "inode": mapped_inode,
                "permissions": permissions,
                "bytes": mapped_stat.st_size,
                "mtime_ns": mapped_stat.st_mtime_ns,
            }
            if is_core_runtime:
                expected_identity = expected_runtime_identities.get(record["path"])
                if expected_identity is None:
                    raise GeometryError(
                        f"mapped core runtime path is not pinned: {record['path']}"
                    )
                record["attested_sha256"] = expected_identity["sha256"]
                record["attested_build_id"] = expected_identity["build_id"]
                loaded_records.append(record)
            else:
                record["sha256"] = sha256_file(path)
                executable_cache_records.append(record)
        loaded = sorted({record["path"] for record in loaded_records})
        observed_runtime_paths.update(loaded)
        records.append(
            {
                **identity,
                "cpu_affinity": list(affinity),
                "cpus_allowed_list": allowed_match.group(1),
                "uid": int(uid_match.group(1)),
                "executable": executable,
                "command": cmdline,
                "loaded_core_runtime_paths": loaded,
                "loaded_core_runtime_mappings": loaded_records,
                "executable_cache_mappings": executable_cache_records,
                "deleted_posix_semaphore_mappings": (
                    deleted_posix_semaphore_records
                ),
                "thread_count": len(thread_records),
                "threads": thread_records,
            }
        )
    if observed_runtime_paths != expected_runtime_paths:
        raise GeometryError(
            "vLLM loaded core runtime paths do not match the pinned environment: "
            f"observed={sorted(observed_runtime_paths)}"
        )
    return {
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root_pid": root_pid,
        "process_count": len(records),
        "all_in_owned_process_group": True,
        "all_cpu_affinity_0_11": True,
        "expected_loaded_core_runtime_paths": sorted(expected_runtime_paths),
        "observed_loaded_core_runtime_paths": sorted(observed_runtime_paths),
        "processes": records,
    }


def server_process_tree_stability_key(receipt: Mapping[str, Any]) -> list[Any]:
    stable: list[Any] = []
    for process in receipt.get("processes", []):
        if not isinstance(process, Mapping):
            raise GeometryError("vLLM process-tree receipt is malformed")
        stable.append(
            {
                key: process[key]
                for key in (
                    "pid",
                    "ppid",
                    "process_group",
                    "session",
                    "start_time_ticks",
                    "cpu_affinity",
                    "cpus_allowed_list",
                    "uid",
                    "executable",
                    "loaded_core_runtime_paths",
                    "loaded_core_runtime_mappings",
                    "executable_cache_mappings",
                    "deleted_posix_semaphore_mappings",
                    "thread_count",
                    "threads",
                )
            }
        )
    return stable


def _telemetry_complete_line_count(path: pathlib.Path) -> int:
    try:
        return path.read_bytes().count(b"\n")
    except OSError as error:
        raise GeometryError(f"cannot read tegrastats monitor log {path}: {error}") from error


def wait_for_telemetry_lines(
    process: subprocess.Popen[bytes],
    path: pathlib.Path,
    minimum_lines: int,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _telemetry_complete_line_count(path) >= minimum_lines:
            return
        returncode = process.poll()
        if returncode is not None:
            raise GeometryError(
                f"tegrastats exited before the measurement envelope: {returncode}"
            )
        time.sleep(0.05)
    raise GeometryError(
        f"tegrastats did not produce {minimum_lines} complete measurement samples"
    )


def start_telemetry_monitor(
    path: pathlib.Path,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    resolved = TEGRASTATS_PATH.resolve()
    if (
        TEGRASTATS_PATH.is_symlink()
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
        or sha256_file(resolved) != TEGRASTATS_SHA256
    ):
        raise GeometryError(f"tegrastats identity mismatch: {TEGRASTATS_PATH}")
    command = [
        "sudo",
        "-n",
        str(resolved),
        "--interval",
        str(TEGRASTATS_INTERVAL_MS),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return process, {
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": process.pid,
        "process_group": process.pid,
        "command": command,
        "binary_sha256": TEGRASTATS_SHA256,
        "interval_ms": TEGRASTATS_INTERVAL_MS,
    }


def _process_group_member_pids(group: int) -> list[int]:
    members: list[int] = []
    try:
        entries = pathlib.Path("/proc").iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                identity = _parse_proc_process_identity(entry)
            except GeometryError:
                if not entry.exists():
                    continue
                raise
            if identity["process_group"] == group:
                members.append(identity["pid"])
    except OSError as error:
        raise GeometryError(f"cannot audit process group {group}: {error}") from error
    return sorted(members)


def _wait_for_privileged_process_group_empty(
    process: subprocess.Popen[bytes], group: int, timeout_seconds: float
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_member_pids(group):
            return True
        time.sleep(0.05)
    process.poll()
    return not _process_group_member_pids(group)


def stop_telemetry_monitor(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    group = process.pid
    receipt: dict[str, Any] = {"pid": process.pid, "signals": []}
    if process.poll() is None:
        try:
            if os.getpgid(process.pid) != group or os.getsid(process.pid) != group:
                raise GeometryError("tegrastats monitor escaped its owned session")
        except ProcessLookupError:
            process.poll()
    for sent_signal, timeout in (
        (signal.SIGTERM, 3.0),
        (signal.SIGKILL, 3.0),
    ):
        members = _process_group_member_pids(group)
        if not members:
            break
        receipt["signals"].append(sent_signal.name)
        receipt.setdefault("members_before_signal", {})[
            sent_signal.name
        ] = members
        completed = subprocess.run(
            [
                "sudo",
                "-n",
                "/usr/bin/kill",
                f"-{sent_signal.name}",
                "--",
                f"-{group}",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10.0,
        )
        if completed.returncode != 0 and _process_group_member_pids(group):
            raise GeometryError(
                "cannot stop privileged tegrastats monitor group: "
                f"status={completed.returncode}, stderr={completed.stderr!r}"
            )
        if _wait_for_privileged_process_group_empty(process, group, timeout):
            break
    receipt["group_empty"] = not _process_group_member_pids(group)
    if not receipt["group_empty"]:
        raise GeometryError("tegrastats monitor did not terminate after SIGKILL")
    process.poll()
    receipt["returncode"] = process.returncode
    receipt["stopped_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return receipt


def validate_measurement_telemetry(
    path: pathlib.Path,
    pre_request_samples: int,
    request_end_samples: int,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise GeometryError(f"cannot read measured tegrastats log {path}: {error}") from error
    samples: list[dict[str, Any]] = []
    maximum_temperature: dict[str, float] = {}
    required_sensors = ("cpu", "gpu", "tj")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        gr3d_matches = GR3D_SAMPLE.findall(line)
        gpu_clock_matches = GR3D_CLOCK_SAMPLE.findall(line)
        emc_clock_matches = EMC_CLOCK_SAMPLE.findall(line)
        cpu_matches = CPU_SAMPLE.findall(line)
        ram_matches = RAM_SAMPLE.findall(line)
        thermal_matches = THERMAL_SAMPLE.findall(line)
        if (
            len(gr3d_matches) != 1
            or len(gpu_clock_matches) != 1
            or len(emc_clock_matches) != 1
            or len(cpu_matches) != 1
            or len(ram_matches) != 1
        ):
            raise GeometryError(
                f"tegrastats line {line_number} lacks a unique resource/clock sample"
            )
        gr3d = float(gr3d_matches[0])
        if not math.isfinite(gr3d) or gr3d < 0.0 or gr3d > 100.0:
            raise GeometryError(f"invalid GR3D utilization at line {line_number}: {gr3d}")
        gpu_clocks = tuple(int(value) for value in gpu_clock_matches[0])
        emc_clock = int(emc_clock_matches[0])
        ram_total = int(ram_matches[0])
        cpu_entries = cpu_matches[0].split(",")
        cpu_clocks: list[int] = []
        for entry in cpu_entries:
            match = CPU_ENTRY.fullmatch(entry.strip())
            if match is None:
                raise GeometryError(
                    f"invalid/offline CPU sample at tegrastats line {line_number}"
                )
            cpu_clocks.append(int(match.group(1)))
        if (
            gpu_clocks != (TEGRASTATS_GPU_MHZ, TEGRASTATS_GPU_MHZ)
            or emc_clock != TEGRASTATS_EMC_MHZ
            or len(cpu_clocks) != len(JETSON_ONLINE_CPUS)
            or set(cpu_clocks) != {TEGRASTATS_CPU_MHZ}
            or ram_total != TEGRASTATS_RAM_TOTAL_MB
        ):
            raise GeometryError(
                f"clock/RAM drift at tegrastats line {line_number}: "
                f"gpu={gpu_clocks}, emc={emc_clock}, cpu={cpu_clocks}, "
                f"ram_total={ram_total}"
            )
        temperatures: dict[str, dict[str, float]] = {}
        for name, current_text, maximum_text in thermal_matches:
            if name in temperatures:
                raise GeometryError(
                    f"duplicate thermal sensor {name} at line {line_number}"
                )
            current = float(current_text)
            maximum = float(maximum_text)
            if (
                not math.isfinite(current)
                or not math.isfinite(maximum)
                or current > maximum
            ):
                raise GeometryError(
                    f"invalid thermal sample {name} at line {line_number}"
                )
            temperatures[name] = {"current_c": current, "maximum_c": maximum}
            maximum_temperature[name] = max(
                maximum_temperature.get(name, maximum), maximum
            )
        missing = [name for name in required_sensors if name not in temperatures]
        if missing:
            raise GeometryError(
                f"tegrastats line {line_number} lacks required sensors: {missing}"
            )
        samples.append(
            {
                "line": line_number,
                "gr3d_percent": gr3d,
                "gpu_clock_mhz": list(gpu_clocks),
                "emc_clock_mhz": emc_clock,
                "cpu_clock_mhz": cpu_clocks,
                "ram_total_mb": ram_total,
                "temperatures": temperatures,
                "raw": line,
            }
        )
    if len(samples) < TEGRASTATS_MINIMUM_MEASURED_SAMPLES:
        raise GeometryError(
            "measured tegrastats envelope has too few samples: "
            f"{len(samples)}"
        )
    if pre_request_samples < 1:
        raise GeometryError("measured tegrastats envelope has no pre-request sample")
    if request_end_samples > len(samples) or pre_request_samples > request_end_samples:
        raise GeometryError("measured tegrastats request window is inconsistent")
    pre_window = samples[:pre_request_samples]
    in_window = samples[pre_request_samples:request_end_samples]
    post_window = samples[request_end_samples:]
    if any(sample["gr3d_percent"] != 0.0 for sample in pre_window):
        raise GeometryError("GPU was active before the measured request began")
    if len(in_window) < TEGRASTATS_MINIMUM_MEASURED_SAMPLES:
        raise GeometryError(
            "measured tegrastats envelope has too few in-request samples: "
            f"{len(in_window)}"
        )
    if len(post_window) < TEGRASTATS_POST_REQUEST_SAMPLES:
        raise GeometryError("measured tegrastats envelope has too few post-request samples")
    if post_window[-1]["gr3d_percent"] != 0.0:
        raise GeometryError("GPU remained active after the measured request ended")
    maximum_gr3d = max(sample["gr3d_percent"] for sample in in_window)
    if maximum_gr3d <= 0.0:
        raise GeometryError("measured tegrastats envelope proves no GPU activity")
    over_temperature = {
        name: value
        for name, value in maximum_temperature.items()
        if name in required_sensors
        and value * 1000.0 >= JETSON_MAX_TEMPERATURE_MILLIC
    }
    if over_temperature:
        raise GeometryError(
            f"measured request crossed the thermal envelope: {over_temperature}"
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "sample_count": len(samples),
        "pre_request_sample_count": pre_request_samples,
        "in_request_sample_count": len(in_window),
        "post_request_sample_count": len(post_window),
        "maximum_gr3d_percent": maximum_gr3d,
        "clock_contract": {
            "gpu_mhz": TEGRASTATS_GPU_MHZ,
            "emc_mhz": TEGRASTATS_EMC_MHZ,
            "cpu_mhz": TEGRASTATS_CPU_MHZ,
            "cpu_count": len(JETSON_ONLINE_CPUS),
            "ram_total_mb": TEGRASTATS_RAM_TOTAL_MB,
            "stable_for_every_sample": True,
        },
        "required_sensors": list(required_sensors),
        "maximum_temperature_c": maximum_temperature,
        "maximum_admitted_temperature_c": JETSON_MAX_TEMPERATURE_MILLIC / 1000.0,
        "samples": samples,
    }


def run_preflight(
    config: GeometryConfig,
    output: pathlib.Path,
    allowed_pids: Sequence[int],
) -> dict[str, Any]:
    command = [
        "sudo",
        "-n",
        sys.executable,
        "-B",
        str(PREFLIGHT_TOOL),
        "--output",
        str(output),
        "--samples",
        str(config.preflight_samples),
        "--interval-ms",
        str(config.preflight_interval_ms),
        "--max-gr3d-percent",
        "0",
        "--max-unexpected-cpu-percent",
        "5",
    ]
    for pid in sorted(set(config.allow_pids) | set(allowed_pids)):
        command.extend(("--allow-pid", str(pid)))
    environment = build_environment(MACROCHUNK_BUDGETS[0], create=True)
    returncode = _run_logged(command, environment, output.with_suffix(".log"), 180.0)
    raw_preflight_sha256: str | None = None
    if output.exists():
        owner = f"{os.getuid()}:{os.getgid()}"
        try:
            subprocess.run(
                ["sudo", "-n", "chown", owner, str(output)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["sudo", "-n", "chmod", "0644", str(output)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as error:
            raise GeometryError(
                f"cannot transfer ownership of preflight evidence: {output}"
            ) from error
        raw_preflight_sha256 = sha256_file(output)
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GeometryError(f"preflight produced no readable report: {output}") from error
    decision = report.get("decision", {})
    if returncode != 0 or decision.get("accepted") is not True:
        raise GeometryError(
            f"clean-host preflight rejected the lane: {decision.get('result')}"
        )
    report["raw_preflight_sha256"] = raw_preflight_sha256
    report["lane_state_after_acceptance"] = collect_performance_lane_state()
    write_json(output, report)
    return report


def run_evalscope(
    config: GeometryConfig,
    runtime: EvalScopeRuntime,
    budget: int,
    corpus: pathlib.Path,
    outputs_dir: pathlib.Path,
    name: str,
    log_path: pathlib.Path,
) -> dict[str, float]:
    command = build_evalscope_command(
        config, corpus, outputs_dir, name, runtime.executable
    )
    returncode = _run_logged(
        command,
        build_environment(budget, create=True),
        log_path,
        config.request_timeout_seconds + 180.0,
    )
    if returncode != 0:
        raise GeometryError(f"EvalScope failed with status {returncode}: {log_path}")
    summary = outputs_dir / name / "parallel_1_number_1/benchmark_summary.json"
    return validate_evalscope_summary(summary)


def run_measured_evalscope(
    config: GeometryConfig,
    runtime: EvalScopeRuntime,
    budget: int,
    outputs_dir: pathlib.Path,
    name: str,
    log_path: pathlib.Path,
    telemetry_path: pathlib.Path,
) -> tuple[
    dict[str, float] | None,
    dict[str, Any],
    BaseException | None,
]:
    monitor: subprocess.Popen[bytes] | None = None
    monitor_start: dict[str, Any] = {}
    monitor_stop: dict[str, Any] = {}
    result: dict[str, float] | None = None
    failure: BaseException | None = None
    control_failure: BaseException | None = None
    pre_request_samples = 0
    request_end_samples = 0
    request_window: dict[str, Any] = {}
    try:
        monitor, monitor_start = start_telemetry_monitor(telemetry_path)
        wait_for_telemetry_lines(monitor, telemetry_path, 1, 5.0)
        pre_request_samples = _telemetry_complete_line_count(telemetry_path)
        request_window["started_at_utc"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()
        request_window["started_monotonic_ns"] = time.monotonic_ns()
        request_window["started_unix_ns"] = time.time_ns()
        result = run_evalscope(
            config,
            runtime,
            budget,
            MEASURED_CORPUS,
            outputs_dir,
            name,
            log_path,
        )
        request_window["ended_monotonic_ns"] = time.monotonic_ns()
        request_window["ended_unix_ns"] = time.time_ns()
        request_window["ended_at_utc"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()
        request_window["elapsed_seconds"] = (
            request_window["ended_monotonic_ns"]
            - request_window["started_monotonic_ns"]
        ) / 1_000_000_000.0
        request_end_samples = _telemetry_complete_line_count(telemetry_path)
        wait_for_telemetry_lines(
            monitor,
            telemetry_path,
            request_end_samples + TEGRASTATS_POST_REQUEST_SAMPLES,
            3.0,
        )
    except BaseException as error:
        if isinstance(error, Exception):
            failure = error
        else:
            control_failure = error
    finally:
        if monitor is not None:
            try:
                monitor_stop = stop_telemetry_monitor(monitor)
            except BaseException as error:
                monitor_stop = {"cleanup_error": repr(error)}
                if not isinstance(error, Exception):
                    if control_failure is None:
                        control_failure = error
                elif failure is None:
                    failure = error

    telemetry: dict[str, Any] = {
        "monitor_start": monitor_start,
        "monitor_stop": monitor_stop,
        "request_window": request_window,
    }
    if telemetry_path.exists():
        try:
            telemetry["measurement"] = validate_measurement_telemetry(
                telemetry_path, pre_request_samples, request_end_samples
            )
        except BaseException as error:
            telemetry["validation_error"] = repr(error)
            if not isinstance(error, Exception):
                if control_failure is None:
                    control_failure = error
            elif failure is None:
                failure = error
    try:
        write_json(telemetry_path.with_suffix(".json"), telemetry)
    except BaseException as error:
        if not isinstance(error, Exception):
            if control_failure is None:
                control_failure = error
        elif failure is None:
            failure = error
    if control_failure is not None:
        raise control_failure
    if result is None and failure is None:
        failure = GeometryError("measured EvalScope request produced no result")
    # Do not raise here.  The server is still alive, and a thermal, frequency,
    # EvalScope, or telemetry validation failure must not prevent the caller
    # from retaining the post-request runtime, metric, and cache state.  The
    # exact exception object is returned so run_budget can re-raise the primary
    # failure after those best-effort collectors finish.
    return result, telemetry, failure


def observe_server_log(path: pathlib.Path, budget: int) -> dict[str, Any]:
    try:
        raw_log = path.read_bytes()
    except OSError as error:
        raise GeometryError(f"cannot read vLLM log {path}: {error}") from error
    # vLLM and its progress renderers can emit bare carriage returns while
    # updating one terminal line.  Universal-newline text IO and
    # str.splitlines() both turn those updates into fictitious physical lines.
    # Decode only after retaining the original bytes, and recognize LF alone
    # as the physical-record delimiter.  CR remains available for byte-order
    # event attribution inside its actual LF-delimited record.
    text = raw_log.decode("utf-8", errors="replace")
    physical_lines = text.split("\n")
    forbidden = (
        "linear_backend='humming'",
        '"linear_backend": "humming"',
        "Using Humming",
        "speculative_config={'",
        "speculative_config=SpeculativeConfig",
    )
    matched_forbidden = [value for value in forbidden if value in text]
    required_probes = {
        "chunked_prefill": "enable_chunked_prefill=True",
        "exact_macrochunk_budget": f"'max_num_batched_tokens': {budget}",
        "prefix_cache_off": "enable_prefix_caching=False",
        "speculative_none": "speculative_config=None",
        "modelopt_quantization": "quantization=modelopt",
        "flashinfer_attention": "Using AttentionBackendEnum.FLASHINFER backend.",
        "triton_fla_gdn": "Using Triton/FLA GDN prefill kernel",
    }
    probe_hits = {name: value in text for name, value in required_probes.items()}
    missing = [name for name, hit in probe_hits.items() if not hit]
    validation_reasons: list[str] = []
    if matched_forbidden:
        validation_reasons.append(
            f"vLLM log shows a forbidden route: {matched_forbidden}"
        )
    if missing:
        validation_reasons.append(
            f"vLLM log is missing required route evidence: {missing}"
        )
    logger_pattern = re.compile(
        r"Avg prompt throughput:\s*(?P<prompt>[0-9]+(?:\.[0-9]+)?)\s+"
        r"tokens/s,\s*Avg generation throughput:\s*"
        r"(?P<generation>[0-9]+(?:\.[0-9]+)?)\s+tokens/s,\s*"
        r"Running:\s*(?P<running>[0-9]+)\s+reqs,\s*"
        r"Waiting:\s*(?P<waiting>[0-9]+)\s+reqs,.*?"
        r"GPU KV cache usage:\s*(?P<kv>[0-9]+(?:\.[0-9]+)?)%,\s*"
        r"Prefix cache hit rate:\s*(?P<prefix>[0-9]+(?:\.[0-9]+)?)%"
    )
    timestamp_pattern = re.compile(
        r"\b(?P<stamp>[0-9]{2}-[0-9]{2} "
        r"[0-9]{2}:[0-9]{2}:[0-9]{2})\s+\["
    )
    response_start_access_pattern = re.compile(
        r'"POST /v1/completions HTTP/1\.1"\s+(?P<status>[0-9]{3})\s+OK'
    )
    jit_pattern = re.compile(
        r"(?:Triton )?[Kk]ernel JIT compilation during inference:\s*"
        r"(?P<kernel>[^.]+)\."
    )

    logger_interval_samples: list[dict[str, Any]] = []
    http_response_start_access_log_observations: list[dict[str, Any]] = []
    runtime_jit_events: list[dict[str, Any]] = []
    http_response_start_access_log_observations_seen = 0

    def event_timestamp(line: str, offset: int) -> str | None:
        preceding = None
        carriage_return_segment_start = line.rfind("\r", 0, offset) + 1
        for candidate in timestamp_pattern.finditer(line):
            if candidate.start() > offset:
                break
            if candidate.start() >= carriage_return_segment_start:
                preceding = candidate
        return preceding.group("stamp") if preceding is not None else None

    for line_number, line in enumerate(physical_lines, start=1):
        # A progress renderer can leave several observations in one physical
        # LF record separated by CR.  Process every match in original byte
        # order instead of imposing parser-type order on that record.
        events: list[tuple[int, str, Any]] = []
        events.extend(
            (match.start(), "http_response_start_access_log", match)
            for match in response_start_access_pattern.finditer(line)
        )
        events.extend(
            (match.start(), "logger_interval", match)
            for match in logger_pattern.finditer(line)
        )
        events.extend(
            (match.start(), "runtime_jit", match)
            for match in jit_pattern.finditer(line)
        )
        events.sort(key=lambda event: event[0])

        for offset, kind, match in events:
            if kind == "http_response_start_access_log":
                http_response_start_access_log_observations_seen += 1
                http_response_start_access_log_observations.append(
                    {
                        "ordinal": (
                            http_response_start_access_log_observations_seen
                        ),
                        "line": line_number,
                        "character_offset_within_decoded_physical_line": (
                            offset
                        ),
                        "physical_line_contains_carriage_return": "\r" in line,
                        "status": int(match.group("status")),
                        "semantics": (
                            "access_log_emitted_at_asgi_http_response_start"
                        ),
                    }
                )
                continue

            if http_response_start_access_log_observations_seen == 0:
                response_start_position = (
                    "before_first_http_response_start_access_log_observation"
                )
            elif http_response_start_access_log_observations_seen == 1:
                response_start_position = (
                    "after_first_before_second_http_response_start_access_log_"
                    "observation"
                )
            else:
                response_start_position = (
                    "after_second_http_response_start_access_log_observation"
                )

            if kind == "logger_interval":
                logger_interval_samples.append(
                    {
                        "line": line_number,
                        "character_offset_within_decoded_physical_line": (
                            offset
                        ),
                        "physical_line_contains_carriage_return": "\r" in line,
                        "month_day_clock": event_timestamp(line, offset),
                        "prompt_tokens_per_second": float(match.group("prompt")),
                        "generation_tokens_per_second": float(
                            match.group("generation")
                        ),
                        "running_requests": int(match.group("running")),
                        "waiting_requests": int(match.group("waiting")),
                        "gpu_kv_cache_usage_percent": float(match.group("kv")),
                        "prefix_cache_hit_rate_percent": float(
                            match.group("prefix")
                        ),
                        "http_response_start_access_log_observations_seen": (
                            http_response_start_access_log_observations_seen
                        ),
                        "position_by_fixed_harness_http_response_start_"
                        "access_log_order": response_start_position,
                    }
                )
                continue

            runtime_jit_events.append(
                {
                    "line": line_number,
                    "character_offset_within_decoded_physical_line": offset,
                    "physical_line_contains_carriage_return": "\r" in line,
                    "month_day_clock": event_timestamp(line, offset),
                    "kernel": match.group("kernel").strip(),
                    "http_response_start_access_log_observations_seen": (
                        http_response_start_access_log_observations_seen
                    ),
                    "request_phase_from_http_response_start_access_log": (
                        "indeterminate_including_streaming_warmup"
                    ),
                }
            )

    logger_interval_prompt_throughput = [
        sample["prompt_tokens_per_second"] for sample in logger_interval_samples
    ]

    def first_group(pattern: str) -> str | None:
        match = re.search(pattern, text)
        return match.group(1) if match is not None else None

    def first_float(pattern: str) -> float | None:
        value = first_group(pattern)
        return float(value) if value is not None else None

    compile_ranges = [
        {"start": start, "end": end}
        for start, end in sorted(
            {
                (int(start), int(end))
                for start, end in re.findall(
                    r"compile range \(([0-9]+),\s*([0-9]+)\)", text
                )
            }
        )
    ]
    selected_attention = re.findall(
        r"Using (AttentionBackendEnum\.[A-Z0-9_]+) backend\.", text
    )
    fp8_linear = re.findall(
        r"Selected ([A-Za-z0-9_]+) for ModelOptFp8LinearMethod", text
    )
    return {
        "schema_version": 2,
        "parser_identity": (
            "q3x.vllm_server_log_observation.lf_physical_lines.v2"
        ),
        "physical_line_contract": {
            "delimiter": "LF_byte_only",
            "decoded_physical_line_count": len(physical_lines),
            "carriage_return_byte_count": raw_log.count(b"\r"),
            "bare_carriage_return_byte_count": sum(
                1
                for offset, value in enumerate(raw_log)
                if value == 0x0D
                and (offset + 1 == len(raw_log) or raw_log[offset + 1] != 0x0A)
            ),
            "carriage_returns_preserved_within_physical_lines": True,
        },
        "budget": budget,
        "forbidden_route_matches": matched_forbidden,
        "required_route_probe_hits": probe_hits,
        "route_validation": {
            "passed": not validation_reasons,
            "missing_required_route_probes": missing,
            "forbidden_route_matches": matched_forbidden,
            "failure_reasons": validation_reasons,
        },
        "backend_evidence": {
            "quantization": first_group(r"\bquantization=([^,\s]+)"),
            "requested_linear_backend": first_group(
                r"\blinear_backend='([^']+)'"
            ),
            "selected_fp8_linear_kernels": fp8_linear,
            "selected_attention_backends": selected_attention,
            "gdn_prefill": first_group(
                r"Using ([^\n]+ GDN prefill kernel \(requested=[^)]+\))\."
            ),
            "flashinfer_resolved_dtypes": first_group(
                r"FlashInfer resolved query dtypes:\s*([^\n]+)"
            ),
            "nvfp4_linear_selection": {
                "kernel": "MarlinNvFp4LinearKernel",
                "evidence_status": "source-resolved/runtime-hit-unproven",
                "runtime_selection_log_observed": False,
            },
            "humming_runtime_selected": bool(
                re.search(r"Using Humming|linear_backend=['\"]humming", text)
            ),
        },
        "startup_and_compilation": {
            "compile_ranges": compile_ranges,
            "torch_compile_total_seconds": first_float(
                r"torch\.compile took ([0-9]+(?:\.[0-9]+)?) s in total"
            ),
            "initial_profile_warmup_seconds": first_float(
                r"Initial profiling/warmup run took "
                r"([0-9]+(?:\.[0-9]+)?) s"
            ),
            "aot_function_saved": "saved AOT compiled function to" in text,
            "flashinfer_autotune_enabled": "enable_flashinfer_autotune=True"
            in text,
            "runtime_jit_events": runtime_jit_events,
        },
        "http_response_start_access_log_observations": (
            http_response_start_access_log_observations
        ),
        "logger_interval_samples": logger_interval_samples,
        "logger_interval_prompt_throughput_tokens_per_second": (
            logger_interval_prompt_throughput
        ),
        "logger_interval_prompt_throughput_maximum": (
            max(logger_interval_prompt_throughput)
            if logger_interval_prompt_throughput
            else None
        ),
        "log_sha256": hashlib.sha256(raw_log).hexdigest(),
        "authority": {
            "backend_and_startup_route": "supporting_route_evidence",
            "logger_interval_throughput": "supporting_telemetry_only",
            "logger_interval_duration": "not_established_by_server_log",
            "nvfp4_linear_runtime_hit": "not_established_by_server_log",
            "request_latency": "not_established_by_server_log",
            "request_phase_from_http_response_start_access_log": (
                "not_established"
            ),
            "scheduler_geometry": "must_be_established_by_metric_delta",
            "performance_promotion": False,
        },
        "note": (
            "W4A16_NVFP4 resolves to Marlin in the attested vLLM 0.26.0 "
            "ModelOpt source, but this server log has no NVFP4 runtime-selection "
            "or kernel-hit observation; its status is therefore source-resolved/"
            "runtime-hit-unproven. Avg prompt throughput is "
            "computed from tokens recorded during vLLM's local logger interval; "
            "the actual interval duration is not present in this server log and "
            "must not be assumed to be exactly ten seconds from nominal cadence. "
            "A long iteration is recorded atomically after it completes, so this "
            "window rate is not request elapsed time or pure-Prefill latency. "
            "For streaming requests the retained POST line is an access-log "
            "observation emitted at ASGI http.response.start, not proof of client "
            "header receipt, first body byte, or request completion; its order "
            "alone cannot assign a logger or JIT event to "
            "warmup completion, the measured request, or another request phase."
        ),
    }


def install_termination_handlers() -> tuple[dict[str, Any], dict[int, Any]]:
    state: dict[str, Any] = {"cleanup_started": False, "received": []}
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        state["received"].append(name)
        if not state["cleanup_started"]:
            raise GeometryError(f"received {name} while vLLM was active")

    try:
        for watched in (signal.SIGTERM, signal.SIGHUP):
            previous[watched] = signal.getsignal(watched)
            signal.signal(watched, handler)
    except (OSError, ValueError):
        for watched, old_handler in previous.items():
            signal.signal(watched, old_handler)
        raise GeometryError("cannot install scoped vLLM termination handlers")
    return state, previous


def restore_termination_handlers(previous: Mapping[int, Any]) -> None:
    for watched, old_handler in previous.items():
        signal.signal(watched, old_handler)


def run_budget(
    config: GeometryConfig, runtime: EvalScopeRuntime, budget: int
) -> dict[str, Any]:
    run_dir = config.output_dir / f"b{budget}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "command.json",
        {
            "budget": budget,
            "server": build_server_command(config, budget),
            "warmup": build_evalscope_command(
                config,
                WARMUP_CORPUS,
                run_dir / "warmup",
                f"warmup-b{budget}",
                runtime.executable,
            ),
            "measured": build_evalscope_command(
                config,
                MEASURED_CORPUS,
                run_dir / "evalscope",
                f"p40-b{budget}",
                runtime.executable,
            ),
            "environment": environment_receipt(budget),
        },
    )
    rpc_before_server = collect_vllm_rpc_sockets()
    if rpc_before_server:
        raise GeometryError(
            f"stale vLLM RPC sockets exist before server start: {rpc_before_server}"
        )
    cooldown_before_server = wait_for_thermal_cooldown(
        f"b{budget}-before-server"
    )
    result_preflight = run_preflight(
        config, run_dir / "preflight-before-server.json", ()
    )

    server_log_path = run_dir / "server.log"
    server_log: Any = None
    server: subprocess.Popen[bytes] | None = None
    failure: BaseException | None = None

    def retain_failure(error: BaseException) -> None:
        """Keep the first ordinary failure, but never hide control flow."""
        nonlocal failure
        is_control = not isinstance(error, Exception)
        current_is_control = (
            failure is not None and not isinstance(failure, Exception)
        )
        if failure is None or (is_control and not current_is_control):
            failure = error

    result: dict[str, Any] = {
        "preflight_before_server": result_preflight["decision"],
        "rpc_socket_admission": {"before_server": rpc_before_server},
        "thermal_cooldown_before_server": cooldown_before_server,
    }
    cleanup: dict[str, Any] = {}
    termination_state, previous_handlers = install_termination_handlers()
    try:
        server_log = server_log_path.open("wb")
        watched_signals = {signal.SIGTERM, signal.SIGHUP}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
        try:
            server = subprocess.Popen(
                build_server_command(config, budget),
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                env=build_environment(budget, create=True),
                start_new_session=True,
                close_fds=True,
            )
            process_group = os.getpgid(server.pid)
            process_session = os.getsid(server.pid)
            if process_group != server.pid or process_session != server.pid:
                raise GeometryError("vLLM did not become its own process-group leader")
            process_receipt = {
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pid": server.pid,
                "process_group": process_group,
                "session": process_session,
                "command": build_server_command(config, budget),
            }
            write_json(run_dir / "server-process.json", process_receipt)
            result["server"] = process_receipt
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        wait_for_server(server, config.port, config.readiness_timeout_seconds)
        result["warmup"] = run_evalscope(
            config,
            runtime,
            budget,
            WARMUP_CORPUS,
            run_dir / "warmup",
            f"warmup-b{budget}",
            run_dir / "warmup-evalscope.log",
        )
        cache_after_warmup = snapshot_compilation_caches(budget)
        write_json(run_dir / "cache-after-warmup.json", cache_after_warmup)
        runtime_after_warmup = collect_server_process_tree_runtime(
            config, server.pid
        )
        write_json(
            run_dir / "server-runtime-after-warmup.json", runtime_after_warmup
        )
        result["thermal_cooldown_before_request"] = wait_for_thermal_cooldown(
            f"b{budget}-before-request"
        )
        request_preflight = run_preflight(
            config,
            run_dir / "preflight-before-request.json",
            (server.pid,),
        )
        result["preflight_before_request"] = request_preflight["decision"]
        runtime_before_request = collect_server_process_tree_runtime(
            config, server.pid
        )
        write_json(
            run_dir / "server-runtime-before-request.json",
            runtime_before_request,
        )
        if server_process_tree_stability_key(
            runtime_before_request
        ) != server_process_tree_stability_key(runtime_after_warmup):
            raise GeometryError(
                "vLLM process/runtime identity changed between warmup and request"
            )

        # Fetch the before snapshot only after the accepted request preflight.
        metrics_url = f"http://127.0.0.1:{config.port}/metrics"
        before_text = fetch_text(metrics_url)
        before_path = run_dir / "metrics-before.prom"
        before_path.write_text(before_text, encoding="utf-8")
        before = parse_prometheus(before_text)

        measured_result, measured_telemetry, request_failure = (
            run_measured_evalscope(
                config,
                runtime,
                budget,
                run_dir / "evalscope",
                f"p40-b{budget}",
                run_dir / "measured-evalscope.log",
                run_dir / "measured-tegrastats.log",
            )
        )
        result["measurement_telemetry"] = measured_telemetry
        if measured_result is not None:
            result["evalscope"] = measured_result

        def error_receipt(error: BaseException) -> dict[str, str]:
            return {
                "type": type(error).__name__,
                "repr": repr(error),
            }

        post_request_evidence: dict[str, Any] = {
            "schema_version": 1,
            "artifact": "q3x_vllm_p40_post_request_evidence",
            "primary_request_error": (
                error_receipt(request_failure)
                if request_failure is not None
                else None
            ),
            "collectors": {},
            "secondary_errors": [],
        }
        secondary_failures: list[Exception] = []

        def record_secondary(name: str, error: Exception) -> None:
            receipt = {"collector": name, **error_receipt(error)}
            post_request_evidence["secondary_errors"].append(receipt)
            secondary_failures.append(error)

        runtime_after_path = run_dir / "server-runtime-after-request.json"
        try:
            runtime_after_request = collect_server_process_tree_runtime(
                config, server.pid
            )
            write_json(runtime_after_path, runtime_after_request)
            if server_process_tree_stability_key(
                runtime_after_request
            ) != server_process_tree_stability_key(runtime_before_request):
                raise GeometryError(
                    "vLLM process/runtime identity changed during the measured "
                    "request"
                )
            result["server_runtime_stable"] = True
            post_request_evidence["collectors"]["server_runtime"] = {
                "status": "pass",
                "path": runtime_after_path.name,
                "sha256": sha256_file(runtime_after_path),
                "stable": True,
            }
        except Exception as error:
            post_request_evidence["collectors"]["server_runtime"] = {
                "status": "fail",
                "path": (
                    runtime_after_path.name if runtime_after_path.exists() else None
                ),
                "error": error_receipt(error),
            }
            record_secondary("server_runtime", error)

        lane_after_path = run_dir / "lane-state-after-request.json"
        try:
            lane_after_request = collect_performance_lane_state()
            write_json(lane_after_path, lane_after_request)
            result["lane_state_after_request"] = lane_after_request
            post_request_evidence["collectors"]["performance_lane"] = {
                "status": "pass",
                "path": lane_after_path.name,
                "sha256": sha256_file(lane_after_path),
            }
        except Exception as error:
            post_request_evidence["collectors"]["performance_lane"] = {
                "status": "fail",
                "path": lane_after_path.name if lane_after_path.exists() else None,
                "error": error_receipt(error),
            }
            record_secondary("performance_lane", error)

        metrics_attempts: list[dict[str, Any]] = []
        metrics_collector: dict[str, Any] = {
            "status": "fail",
            "attempts": metrics_attempts,
        }
        post_request_evidence["collectors"]["prometheus_after"] = metrics_collector
        try:
            deadline = time.monotonic() + 20.0
            last_error: Exception | None = None
            attempt = 0
            while time.monotonic() < deadline:
                attempt += 1
                attempt_record: dict[str, Any] = {"ordinal": attempt}
                metrics_attempts.append(attempt_record)
                try:
                    after_text = fetch_text(metrics_url)
                except Exception as error:
                    attempt_record["fetch_error"] = error_receipt(error)
                    last_error = error
                    time.sleep(1.0)
                    continue

                attempt_path = run_dir / f"metrics-after-attempt-{attempt:03d}.prom"
                # Retain every raw successful scrape before parsing or applying
                # the exact one-request delta gate.  An invalid delta remains
                # useful diagnostic evidence and must not erase its source.
                attempt_path.write_text(after_text, encoding="utf-8")
                attempt_record.update(
                    {
                        "path": attempt_path.name,
                        "sha256": sha256_file(attempt_path),
                        "bytes": len(after_text.encode("utf-8")),
                    }
                )
                try:
                    after = parse_prometheus(after_text)
                    metric_delta = validate_metric_delta(before, after)
                except Exception as error:
                    attempt_record["validation_error"] = error_receipt(error)
                    last_error = error
                    time.sleep(1.0)
                    continue

                final_after_path = run_dir / "metrics-after.prom"
                final_after_path.write_text(after_text, encoding="utf-8")
                result["server_metric_delta"] = metric_delta
                metrics_collector.update(
                    {
                        "status": "pass",
                        "settled_attempt": attempt,
                        "path": final_after_path.name,
                        "sha256": sha256_file(final_after_path),
                    }
                )
                break
            else:
                raise GeometryError(
                    f"Prometheus delta did not settle: {last_error}"
                )
        except Exception as error:
            metrics_collector["error"] = error_receipt(error)
            record_secondary("prometheus_after", error)

        cache_after_path = run_dir / "cache-after-measured.json"
        try:
            cache_after_measured = snapshot_compilation_caches(budget)
            write_json(cache_after_path, cache_after_measured)
            if cache_after_measured != cache_after_warmup:
                raise GeometryError(
                    "compilation/JIT cache changed during the measured request"
                )
            result["compilation_cache_stable"] = {
                "stable": True,
                "file_count": cache_after_measured["file_count"],
                "bytes": cache_after_measured["bytes"],
            }
            post_request_evidence["collectors"]["compilation_cache"] = {
                "status": "pass",
                "path": cache_after_path.name,
                "sha256": sha256_file(cache_after_path),
                "stable": True,
            }
        except Exception as error:
            post_request_evidence["collectors"]["compilation_cache"] = {
                "status": "fail",
                "path": cache_after_path.name if cache_after_path.exists() else None,
                "error": error_receipt(error),
            }
            record_secondary("compilation_cache", error)

        post_request_evidence["completed_at_utc"] = dt.datetime.now(
            dt.timezone.utc
        ).isoformat()
        try:
            write_json(
                run_dir / "post-request-evidence.json", post_request_evidence
            )
        except Exception as error:
            # The aggregate receipt is itself a best-effort collector.  Its
            # write failure must never replace the request/telemetry failure
            # that made this run invalid in the first place.
            record_secondary("post_request_evidence", error)
        if request_failure is not None:
            raise request_failure
        if secondary_failures:
            raise secondary_failures[0]
    except BaseException as error:  # preserve cleanup even on KeyboardInterrupt
        retain_failure(error)
    finally:
        termination_state["cleanup_started"] = True
        cleanup["termination_signals"] = termination_state["received"]
        try:
            if server is not None:
                try:
                    cleanup["process_group"] = stop_process_group(server)
                except BaseException as error:
                    cleanup["process_group_error"] = repr(error)
                    retain_failure(error)
            process_group_released = server is None or bool(
                cleanup.get("process_group", {}).get("group_empty")
            )
            if process_group_released:
                try:
                    cleanup["rpc_sockets"] = cleanup_vllm_rpc_sockets()
                except BaseException as error:
                    cleanup["rpc_socket_error"] = repr(error)
                    retain_failure(error)
            if server_log is not None:
                try:
                    server_log.close()
                except BaseException as error:
                    cleanup["server_log_close_error"] = repr(error)
                    retain_failure(error)
            if server is not None:
                try:
                    final_log_observations = observe_server_log(
                        server_log_path, budget
                    )
                    write_json(
                        run_dir / "server-log-observations.json",
                        final_log_observations,
                    )
                    result.setdefault("server", {})["final_log_observations"] = (
                        final_log_observations
                    )
                    result["prefill_metric_surfaces"] = (
                        build_prefill_metric_surfaces(result)
                    )
                    cleanup["server_log_observations"] = {
                        "path": "server-log-observations.json",
                        "sha256": sha256_file(
                            run_dir / "server-log-observations.json"
                        ),
                        "retained_when_performance_invalid": True,
                    }
                    route_validation = final_log_observations.get(
                        "route_validation", {}
                    )
                    if route_validation.get("passed") is not True:
                        route_error = GeometryError(
                            "vLLM server-log route validation failed: "
                            f"{route_validation.get('failure_reasons')}"
                        )
                        cleanup["server_log_route_validation_error"] = {
                            "type": type(route_error).__name__,
                            "repr": repr(route_error),
                            "structured_observation_retained": True,
                        }
                        retain_failure(route_error)
                except BaseException as error:
                    cleanup["server_log_audit_error"] = repr(error)
                    retain_failure(error)
            if server is not None and process_group_released:
                try:
                    cleanup["thermal_cooldown_before_post_release"] = (
                        wait_for_thermal_cooldown(
                            f"b{budget}-before-post-release"
                        )
                    )
                except BaseException as error:
                    cleanup["thermal_cooldown_error"] = repr(error)
                    retain_failure(error)
            try:
                cleanup["post_release_preflight"] = run_preflight(
                    config, run_dir / "post-release-preflight.json", ()
                )["decision"]
            except BaseException as error:
                cleanup["post_release_error"] = repr(error)
                retain_failure(error)
        except BaseException as error:
            cleanup["unexpected_cleanup_error"] = repr(error)
            retain_failure(error)
        finally:
            try:
                restore_termination_handlers(previous_handlers)
            except BaseException as error:
                cleanup["termination_handler_restore_error"] = repr(error)
                retain_failure(error)
            try:
                write_json(run_dir / "cleanup.json", cleanup)
            except BaseException as error:
                retain_failure(error)
    if failure is not None:
        result["prefill_metric_surfaces"] = build_prefill_metric_surfaces(result)
        partial_result = {
            "schema_version": 1,
            "artifact": "q3x_vllm_p40_partial_result",
            "valid": False,
            "budget": budget,
            "failure": {
                "type": type(failure).__name__,
                "repr": repr(failure),
            },
            "prefill_metric_surfaces": result["prefill_metric_surfaces"],
            "available_result": result,
        }
        try:
            write_json(run_dir / "partial-result.json", partial_result)
        except BaseException as receipt_error:
            try:
                failure.add_note(
                    "partial result receipt could not be written: "
                    f"{receipt_error!r}"
                )
            except (AttributeError, TypeError):
                pass
        raise failure
    if termination_state["received"]:
        raise GeometryError(
            "termination signal was received during the budget lifecycle: "
            f"{termination_state['received']}"
        )
    result["budget"] = budget
    result["planned_scheduled_forwards"] = math.ceil(PROMPT_TOKENS / budget)
    result["prefill_metric_surfaces"] = build_prefill_metric_surfaces(result)
    result["valid"] = True
    write_json(run_dir / "result.json", result)
    return result


def build_plan(
    config: GeometryConfig, runtime: EvalScopeRuntime | None = None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "q3x_vllm_p40_macrochunk_geometry_plan",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": {"name": MODEL_NAME, "directory": str(config.model_dir)},
        "vllm": {
            "binary": str(config.vllm_bin),
            "required_version": VLLM_VERSION,
            "linear_backend": "stock-auto (no CLI override; W4A16_NVFP4 is Marlin)",
            "attention_backend": "FLASHINFER",
            "mamba_backend": "TRITON",
        },
        "evalscope": {
            "uvx": str(config.uvx_bin),
            "requirement": EVALSCOPE_REQUIREMENT,
            "offline": True,
            "resolved_prefix": str(runtime.prefix) if runtime else None,
            "measured_executable": str(runtime.executable) if runtime else None,
            "request_time_resolution": False if runtime else None,
        },
        "budgets": [
            {
                "max_num_batched_tokens": budget,
                "planned_scheduled_forwards": math.ceil(PROMPT_TOKENS / budget),
                "server_command": build_server_command(config, budget),
            }
            for budget in MACROCHUNK_BUDGETS
        ],
        "deferred_explanatory_budgets": list(DEFERRED_EXPLANATORY_BUDGETS),
        "environment": environment_receipt(MACROCHUNK_BUDGETS[0]),
        "output_dir": str(config.output_dir),
        "clean_host": {
            "preflight": str(PREFLIGHT_TOOL),
            "gpu_utilization_source": "tegrastats-only",
            "measured_thermal_source": "tegrastats cpu/gpu/tj envelope",
            "nvidia_smi_used": False,
            "allow_pids": list(config.allow_pids),
        },
    }


def validate_hashed_files(
    root: pathlib.Path,
    expected: Mapping[str, str],
    identity_name: str,
) -> dict[str, Any]:
    if root.is_symlink() or root.absolute() != root.resolve():
        raise GeometryError(f"{identity_name} root is not canonical: {root}")
    resolved_root = root.resolve()
    observed: dict[str, dict[str, Any]] = {}
    for relative, wanted_sha256 in sorted(expected.items()):
        relative_path = pathlib.PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise GeometryError(
                f"invalid {identity_name} identity path: {relative}"
            )
        path = root
        try:
            for component in relative_path.parts:
                path /= component
                component_stat = path.lstat()
                if stat.S_ISLNK(component_stat.st_mode):
                    raise GeometryError(
                        f"{identity_name} identity path contains a symlink: {path}"
                    )
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
            file_stat = path.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise GeometryError(
                    f"{identity_name} identity path is not a regular file: {path}"
                )
            digest = sha256_file(path)
        except (OSError, ValueError) as error:
            raise GeometryError(
                f"cannot read {identity_name} identity file {path}: {error}"
            ) from error
        if digest != wanted_sha256:
            raise GeometryError(
                f"{identity_name} identity mismatch for {path}: {digest}"
            )
        observed[relative] = {
            "path": str(resolved),
            "sha256": digest,
            "bytes": file_stat.st_size,
        }
    return {"root": str(resolved_root), "files": observed}


def validate_directory_tree(
    root: pathlib.Path, expected_sha256: str, identity_name: str
) -> dict[str, Any]:
    if root.is_symlink() or root.absolute() != root.resolve():
        raise GeometryError(f"{identity_name} root is not canonical: {root}")
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    try:
        entries = sorted(
            root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
        )
        for path in entries:
            relative = path.relative_to(root)
            path_stat = path.lstat()
            if stat.S_ISLNK(path_stat.st_mode):
                raise GeometryError(
                    f"{identity_name} tree contains a symlink: {path}"
                )
            if stat.S_ISDIR(path_stat.st_mode):
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                raise GeometryError(
                    f"{identity_name} tree contains a non-regular file: {path}"
                )
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            name = relative.as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(path_stat.st_size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            file_count += 1
            byte_count += path_stat.st_size
    except OSError as error:
        raise GeometryError(f"cannot attest {identity_name} tree: {error}") from error
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise GeometryError(
            f"{identity_name} tree identity mismatch: {observed}"
        )
    return {
        "root": str(root.resolve()),
        "sha256": observed,
        "files": file_count,
        "bytes": byte_count,
    }


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_evalscope_environment(prefix: pathlib.Path) -> dict[str, Any]:
    """Verify the complete uvx-resolved environment outside that environment."""

    archive_root = (WORK_ROOT / "cache/uv/archive-v0").resolve()
    if prefix.is_symlink() or prefix.absolute() != prefix.resolve():
        raise GeometryError(f"EvalScope prefix is not canonical: {prefix}")
    try:
        prefix.resolve().relative_to(archive_root)
    except ValueError as error:
        raise GeometryError(
            f"EvalScope prefix is outside the repository uv cache: {prefix}"
        ) from error

    python = prefix / "bin/python"
    executable = prefix / "bin/evalscope"
    if not python.is_symlink() or python.resolve() != EVALSCOPE_BASE_PYTHON.resolve():
        raise GeometryError(f"EvalScope Python link is not the pinned interpreter: {python}")
    if executable.is_symlink() or not executable.is_file():
        raise GeometryError(f"EvalScope executable is not a sealed file: {executable}")
    executable_sha256 = sha256_file(executable)
    if executable_sha256 != EVALSCOPE_EXECUTABLE_SHA256:
        raise GeometryError(
            f"EvalScope executable identity mismatch: {executable_sha256}"
        )

    site_packages = prefix / "lib/python3.13/site-packages"
    tree = validate_directory_tree(
        site_packages,
        EVALSCOPE_SITE_PACKAGES_SHA256,
        "EvalScope site-packages",
    )
    if (
        tree["files"] != EVALSCOPE_SITE_PACKAGES_FILES
        or tree["bytes"] != EVALSCOPE_SITE_PACKAGES_BYTES
    ):
        raise GeometryError(
            "EvalScope site-packages count/size mismatch: "
            f"files={tree['files']}, bytes={tree['bytes']}"
        )

    entries: list[dict[str, Any]] = []
    normalized_names: set[str] = set()
    total_rows = 0
    total_hashed_rows = 0
    records = sorted(
        site_packages.glob("*.dist-info/RECORD"), key=lambda path: path.parent.name
    )
    for record in records:
        metadata = record.parent / "METADATA"
        if record.is_symlink() or metadata.is_symlink():
            raise GeometryError(f"EvalScope distribution metadata is symlinked: {record.parent}")
        try:
            parsed = email.parser.Parser().parsestr(
                metadata.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError) as error:
            raise GeometryError(
                f"cannot parse EvalScope distribution metadata {metadata}: {error}"
            ) from error
        name = parsed.get("Name")
        version = parsed.get("Version")
        if not name or not version:
            raise GeometryError(f"distribution metadata lacks Name/Version: {metadata}")
        normalized = _canonical_distribution_name(name)
        if normalized in normalized_names:
            raise GeometryError(f"duplicate normalized distribution name: {name}")
        normalized_names.add(normalized)

        rows = 0
        hashed_rows = 0
        try:
            with record.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.reader(stream):
                    rows += 1
                    values = (row + ["", ""])[:3]
                    relative, encoded_digest, encoded_size = values
                    if not relative or pathlib.PurePosixPath(relative).is_absolute():
                        raise GeometryError(f"invalid RECORD path in {record}: {relative!r}")
                    candidate = site_packages / pathlib.PurePosixPath(relative)
                    try:
                        resolved = candidate.resolve(strict=True)
                        resolved.relative_to(prefix)
                    except (OSError, ValueError) as error:
                        raise GeometryError(
                            f"RECORD path escapes or is unavailable: {candidate}"
                        ) from error
                    if not resolved.is_file():
                        raise GeometryError(f"RECORD path is not a file: {resolved}")
                    if encoded_digest:
                        try:
                            algorithm, expected = encoded_digest.split("=", 1)
                            digest = hashlib.new(algorithm, resolved.read_bytes()).digest()
                        except (OSError, ValueError) as error:
                            raise GeometryError(
                                f"cannot verify RECORD digest for {resolved}: {error}"
                            ) from error
                        observed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                        if observed != expected:
                            raise GeometryError(f"RECORD digest mismatch for {resolved}")
                        if encoded_size and resolved.stat().st_size != int(encoded_size):
                            raise GeometryError(f"RECORD size mismatch for {resolved}")
                        hashed_rows += 1
                    elif resolved != record.resolve():
                        raise GeometryError(
                            f"only a RECORD self-row may omit a digest: {resolved}"
                        )
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            raise GeometryError(f"cannot verify distribution RECORD {record}: {error}") from error

        total_rows += rows
        total_hashed_rows += hashed_rows
        entries.append(
            {
                "name": name,
                "version": version,
                "dist_info": record.parent.name,
                "record_sha256": sha256_file(record),
                "metadata_sha256": sha256_file(metadata),
                "record_rows": rows,
                "hashed_rows": hashed_rows,
            }
        )

    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        len(entries) != EVALSCOPE_DISTRIBUTION_COUNT
        or total_rows != EVALSCOPE_RECORD_ROWS
        or total_hashed_rows != EVALSCOPE_HASHED_RECORD_ROWS
        or manifest_sha256 != EVALSCOPE_DISTRIBUTION_MANIFEST_SHA256
    ):
        raise GeometryError(
            "EvalScope distribution manifest mismatch: "
            f"count={len(entries)}, rows={total_rows}, hashed={total_hashed_rows}, "
            f"sha256={manifest_sha256}"
        )
    return {
        "prefix": str(prefix),
        "python": {
            "path": str(python),
            "target": str(python.resolve()),
            "target_sha256": sha256_file(python.resolve()),
            "version": list(EVALSCOPE_PYTHON_VERSION),
        },
        "executable": {
            "path": str(executable),
            "sha256": executable_sha256,
        },
        "site_packages": tree,
        "distribution_manifest": {
            "sha256": manifest_sha256,
            "distribution_count": len(entries),
            "record_rows": total_rows,
            "hashed_record_rows": total_hashed_rows,
            "unhashed_record_rows": total_rows - total_hashed_rows,
            "entries": entries,
        },
    }


def resolve_vllm_package_root(config: GeometryConfig) -> pathlib.Path:
    environment_root = config.vllm_bin.parent.parent
    candidates = sorted(environment_root.glob("lib/python*/site-packages/vllm"))
    candidates = [candidate for candidate in candidates if candidate.is_dir()]
    if len(candidates) != 1:
        raise GeometryError(
            "cannot resolve one vLLM package root below the pinned environment: "
            f"{candidates}"
        )
    return candidates[0]


def _read_int(path: pathlib.Path, identity: str) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError) as error:
        raise GeometryError(f"cannot read {identity} from {path}: {error}") from error


def _run_readonly_host_command(command: Sequence[str], timeout: float = 30.0) -> str:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=build_environment(MACROCHUNK_BUDGETS[0], create=True),
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise GeometryError(
            f"read-only host command failed: {list(command)!r}, "
            f"status={completed.returncode}, stderr={completed.stderr!r}"
        )
    return completed.stdout


def _elf_build_id(path: pathlib.Path) -> str:
    completed = subprocess.run(
        ["/usr/bin/readelf", "-n", str(path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60.0,
    )
    matches = sorted(set(re.findall(r"Build ID:\s*([0-9a-fA-F]+)", completed.stdout)))
    if completed.returncode != 0 or len(matches) != 1:
        raise GeometryError(
            f"cannot resolve one ELF build ID for {path}: "
            f"status={completed.returncode}, ids={matches}"
        )
    return matches[0].lower()


def collect_thermal_state(*, enforce_envelope: bool = True) -> dict[str, Any]:
    zones: dict[str, Any] = {}
    for zone in sorted(THERMAL_ROOT.glob("thermal_zone*"), key=lambda path: path.name):
        try:
            name = (zone / "type").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            raise GeometryError(f"cannot identify thermal zone {zone}: {error}") from error
        if not name or name in zones:
            raise GeometryError(f"thermal zone identity is empty or duplicated: {name!r}")
        record: dict[str, Any] = {"path": str(zone)}
        try:
            temperature = _read_int(zone / "temp", f"{name} temperature")
        except GeometryError as error:
            record["temperature_millic"] = None
            record["unavailable"] = str(error)
        else:
            record["temperature_millic"] = temperature
        trips: list[dict[str, Any]] = []
        for trip_type in sorted(zone.glob("trip_point_*_type")):
            match = re.fullmatch(r"trip_point_(\d+)_type", trip_type.name)
            if match is None:
                continue
            trip_temp = zone / f"trip_point_{match.group(1)}_temp"
            trips.append(
                {
                    "index": int(match.group(1)),
                    "type": trip_type.read_text(encoding="utf-8").strip(),
                    "temperature_millic": _read_int(
                        trip_temp, f"{name} trip {match.group(1)}"
                    ),
                }
            )
        record["trip_points"] = trips
        zones[name] = record

    required = ("cpu-thermal", "gpu-thermal", "tj-thermal")
    for name in required:
        temperature = zones.get(name, {}).get("temperature_millic")
        if not isinstance(temperature, int):
            raise GeometryError(f"required thermal sensor is unavailable: {name}")
        if enforce_envelope and temperature >= JETSON_MAX_TEMPERATURE_MILLIC:
            raise GeometryError(
                f"{name} is outside the admitted thermal envelope: {temperature}"
            )
    return {
        "maximum_admitted_millic": JETSON_MAX_TEMPERATURE_MILLIC,
        "required_sensors": list(required),
        "zones": zones,
    }


def wait_for_thermal_cooldown(
    stage: str,
    *,
    target_millic: int = THERMAL_COOLDOWN_TARGET_MILLIC,
    stable_samples: int = THERMAL_COOLDOWN_STABLE_SAMPLES,
    interval_seconds: float = THERMAL_COOLDOWN_INTERVAL_SECONDS,
    timeout_seconds: float = THERMAL_COOLDOWN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Passively establish thermal hysteresis before an admission boundary."""

    if (
        target_millic <= 0
        or target_millic >= JETSON_MAX_TEMPERATURE_MILLIC
        or stable_samples < 1
        or interval_seconds <= 0.0
        or timeout_seconds <= 0.0
    ):
        raise GeometryError("invalid thermal cooldown controller configuration")
    started = time.monotonic()
    deadline = started + timeout_seconds
    consecutive = 0
    samples: list[dict[str, Any]] = []
    required = ("cpu-thermal", "gpu-thermal", "tj-thermal")
    while True:
        thermal = collect_thermal_state(enforce_envelope=False)
        temperatures = {
            name: thermal["zones"][name]["temperature_millic"]
            for name in required
        }
        if any(not isinstance(value, int) for value in temperatures.values()):
            raise GeometryError(
                f"thermal cooldown sensor is unavailable at {stage}: {temperatures}"
            )
        maximum = max(temperatures.values())
        consecutive = consecutive + 1 if maximum <= target_millic else 0
        samples.append(
            {
                "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "temperatures_millic": temperatures,
                "maximum_millic": maximum,
                "consecutive_at_or_below_target": consecutive,
            }
        )
        if consecutive >= stable_samples:
            return {
                "stage": stage,
                "target_millic": target_millic,
                "rejection_millic": JETSON_MAX_TEMPERATURE_MILLIC,
                "stable_samples_required": stable_samples,
                "interval_seconds": interval_seconds,
                "elapsed_seconds": time.monotonic() - started,
                "samples": samples,
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise GeometryError(
                f"thermal cooldown timed out at {stage}: last={temperatures}"
            )
        time.sleep(min(interval_seconds, remaining))


def collect_performance_lane_state() -> dict[str, Any]:
    try:
        machine = os.uname().machine
        model = DEVICE_TREE_MODEL.read_bytes().rstrip(b"\0").decode("utf-8")
        compatible = [
            value.decode("utf-8")
            for value in pathlib.Path("/proc/device-tree/compatible")
            .read_bytes()
            .rstrip(b"\0")
            .split(b"\0")
        ]
        online_cpus = pathlib.Path("/sys/devices/system/cpu/online").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, UnicodeError) as error:
        raise GeometryError(f"cannot identify the Jetson performance lane: {error}") from error
    if machine != JETSON_ARCHITECTURE or model != JETSON_MODEL:
        raise GeometryError(f"unexpected performance host: machine={machine}, model={model}")
    if "nvidia,tegra234" not in compatible or online_cpus != "0-11":
        raise GeometryError(
            f"unexpected Orin topology: compatible={compatible}, online={online_cpus}"
        )
    affinity = tuple(sorted(os.sched_getaffinity(0)))
    if affinity != JETSON_ONLINE_CPUS:
        raise GeometryError(f"harness CPU affinity is not pinned to 0-11: {affinity}")

    clocks = {
        "gpu": {
            name: _read_int(GPU_DEVFREQ_ROOT / name, f"GPU {name}")
            for name in ("min_freq", "max_freq", "cur_freq")
        },
        "emc": {
            name: _read_int(EMC_DEVFREQ_ROOT / name, f"EMC {name}")
            for name in ("min_freq", "max_freq", "cur_freq")
        },
        "cpu_policies": {},
    }
    if set(clocks["gpu"].values()) != {JETSON_GPU_HZ}:
        raise GeometryError(f"GPU clocks are not fixed at {JETSON_GPU_HZ}: {clocks['gpu']}")
    if set(clocks["emc"].values()) != {JETSON_EMC_HZ}:
        raise GeometryError(f"EMC clocks are not fixed at {JETSON_EMC_HZ}: {clocks['emc']}")
    for policy in (0, 4, 8):
        root = CPUFREQ_ROOT / f"policy{policy}"
        values = {
            name: _read_int(root / name, f"CPU policy{policy} {name}")
            for name in ("scaling_min_freq", "scaling_max_freq", "scaling_cur_freq")
        }
        if set(values.values()) != {JETSON_CPU_KHZ}:
            raise GeometryError(f"CPU policy{policy} clocks are not fixed: {values}")
        clocks["cpu_policies"][str(policy)] = values

    nvpmodel = _run_readonly_host_command(
        ("sudo", "-n", str(NVP_MODEL_TOOL), "-q")
    )
    if "NV Power Mode: MAXN" not in nvpmodel or not re.search(r"(?m)^0$", nvpmodel):
        raise GeometryError(f"nvpmodel is not MAXN mode 0: {nvpmodel!r}")
    jetson_clocks = _run_readonly_host_command(
        ("sudo", "-n", str(JETSON_CLOCKS_TOOL), "--show"), timeout=60.0
    )
    required_clock_markers = (
        "Active GPU TPCs: 8",
        f"GPU MinFreq={JETSON_GPU_HZ} MaxFreq={JETSON_GPU_HZ} CurrentFreq={JETSON_GPU_HZ}",
        f"EMC MinFreq={JETSON_EMC_HZ} MaxFreq={JETSON_EMC_HZ} CurrentFreq={JETSON_EMC_HZ}",
        "FAN Dynamic Speed Control=nvfancontrol hwmon0_pwm1=255",
        "NV Power Mode: MAXN",
    )
    missing = [marker for marker in required_clock_markers if marker not in jetson_clocks]
    if missing:
        raise GeometryError(f"jetson_clocks readback is incomplete: {missing}")
    return {
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "machine": machine,
        "model": model,
        "compatible": compatible,
        "online_cpus": online_cpus,
        "process_affinity": list(affinity),
        "clocks": clocks,
        "nvpmodel": {"query": nvpmodel, "mode": "MAXN", "id": 0},
        "jetson_clocks": {
            "query": jetson_clocks,
            "required_markers": list(required_clock_markers),
        },
        "thermal": collect_thermal_state(),
    }


def validate_host_runtime(config: GeometryConfig) -> dict[str, Any]:
    system_files = {
        str(L4T_RELEASE_FILE): JETSON_L4T_SHA256,
        str(NVIDIA_DRIVER_FILE): JETSON_DRIVER_SHA256,
        str(CUDA_VERSION_FILE): CUDA_VERSION_JSON_SHA256,
    }
    observed_system: dict[str, Any] = {}
    for raw_path, expected in system_files.items():
        path = pathlib.Path(raw_path)
        digest = sha256_file(path)
        if digest != expected:
            raise GeometryError(f"host runtime identity mismatch for {path}: {digest}")
        observed_system[raw_path] = {"sha256": digest, "text": path.read_text()}

    vllm_root = resolve_vllm_package_root(config)
    site_packages = vllm_root.parent
    runtime_files = validate_hashed_files(
        site_packages,
        VLLM_RUNTIME_FILE_SHA256,
        "vLLM Torch/CUDA runtime",
    )
    build_ids: dict[str, str] = {}
    for relative, expected in VLLM_RUNTIME_BUILD_IDS.items():
        observed = _elf_build_id(site_packages / relative)
        if observed != expected:
            raise GeometryError(f"runtime ELF build ID mismatch for {relative}: {observed}")
        build_ids[relative] = observed
    if LIBCUDA_PATH.is_symlink() or not LIBCUDA_PATH.is_file():
        raise GeometryError(f"actual Jetson libcuda is unavailable: {LIBCUDA_PATH}")
    libcuda_sha256 = sha256_file(LIBCUDA_PATH)
    if libcuda_sha256 != LIBCUDA_SHA256:
        raise GeometryError(f"Jetson libcuda identity mismatch: {libcuda_sha256}")
    libcuda_build_id = _elf_build_id(LIBCUDA_PATH)
    if libcuda_build_id != LIBCUDA_BUILD_ID:
        raise GeometryError(f"Jetson libcuda build ID mismatch: {libcuda_build_id}")
    stable_extension = vllm_root / "_C_stable_libtorch.abi3.so"
    stable_extension_build_id = _elf_build_id(stable_extension)
    if stable_extension_build_id != VLLM_STABLE_EXTENSION_BUILD_ID:
        raise GeometryError(
            f"vLLM stable extension build ID mismatch: {stable_extension_build_id}"
        )

    expected_loaded = sorted(
        str((site_packages / relative).resolve())
        for relative in VLLM_RUNTIME_FILE_SHA256
        if pathlib.Path(relative).name != "_C.cpython-313-aarch64-linux-gnu.so"
    ) + [str(LIBCUDA_PATH)]
    probe_code = """
import ctypes,json,pathlib,torch
p=torch.cuda.get_device_properties(0)
driver=ctypes.c_int(); runtime=ctypes.c_int()
driver_status=ctypes.CDLL('libcuda.so.1').cuDriverGetVersion(ctypes.byref(driver))
runtime_status=ctypes.CDLL('libcudart.so.13').cudaRuntimeGetVersion(ctypes.byref(runtime))
wanted={'libtorch.so','libtorch_cpu.so','libtorch_cuda.so','libc10.so','libc10_cuda.so','libcudart.so.13','libnvJitLink.so.13','libcuda.so.1.1'}
loaded=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if line.split() and pathlib.Path(line.split()[-1]).name in wanted})
print(json.dumps({'torch_version':torch.__version__,'torch_git':torch.version.git_version,'torch_cuda':torch.version.cuda,'name':p.name,'capability':[p.major,p.minor],'multiprocessors':p.multi_processor_count,'total_memory':p.total_memory,'device_count':torch.cuda.device_count(),'driver_status':driver_status,'driver_api':driver.value,'runtime_status':runtime_status,'runtime_api':runtime.value,'loaded':loaded},sort_keys=True))
""".strip()
    probe = subprocess.run(
        [str(config.vllm_bin.parent / "python"), "-B", "-c", probe_code],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=build_environment(MACROCHUNK_BUDGETS[0], create=True),
        timeout=120.0,
    )
    try:
        torch = json.loads(probe.stdout)
    except json.JSONDecodeError as error:
        raise GeometryError(f"Torch runtime probe is not parseable: {probe.stderr!r}") from error
    expected_torch = {
        "torch_version": TORCH_VERSION,
        "torch_git": TORCH_GIT_VERSION,
        "torch_cuda": TORCH_CUDA_VERSION,
        "name": "Orin",
        "capability": list(JETSON_COMPUTE_CAPABILITY),
        "multiprocessors": JETSON_MULTIPROCESSOR_COUNT,
        "total_memory": JETSON_TOTAL_DEVICE_MEMORY,
        "device_count": 1,
        "driver_status": 0,
        "driver_api": TORCH_DRIVER_API_VERSION,
        "runtime_status": 0,
        "runtime_api": TORCH_RUNTIME_API_VERSION,
        "loaded": sorted(expected_loaded),
    }
    if probe.returncode != 0 or torch != expected_torch:
        raise GeometryError(
            f"Torch/CUDA runtime probe mismatch: status={probe.returncode}, receipt={torch!r}"
        )
    return {
        "system_files": observed_system,
        "torch_cuda_files": runtime_files,
        "elf_build_ids": build_ids,
        "libcuda": {
            "path": str(LIBCUDA_PATH),
            "sha256": libcuda_sha256,
            "build_id": libcuda_build_id,
        },
        "vllm_stable_extension_build_id": stable_extension_build_id,
        "torch_cuda_probe": torch,
        "torch_probe_stderr": probe.stderr,
        "lane_state": collect_performance_lane_state(),
    }


def validate_static_inputs(config: GeometryConfig) -> dict[str, Any]:
    if not config.model_dir.is_dir():
        raise GeometryError(f"model directory does not exist: {config.model_dir}")
    if not config.vllm_bin.is_file() or not os.access(config.vllm_bin, os.X_OK):
        raise GeometryError(f"vLLM executable is unavailable: {config.vllm_bin}")
    if not config.uvx_bin.is_file() or not os.access(config.uvx_bin, os.X_OK):
        raise GeometryError(f"uvx executable is unavailable: {config.uvx_bin}")
    measured = validate_corpus(MEASURED_CORPUS, MEASURED_CORPUS_SHA256)
    warmup = validate_corpus(WARMUP_CORPUS, WARMUP_CORPUS_SHA256)
    if measured["sha256"] == warmup["sha256"]:
        raise GeometryError("warmup and measured corpora must be distinct")
    binary_sha256 = sha256_file(config.vllm_bin)
    if binary_sha256 != VLLM_BINARY_SHA256:
        raise GeometryError(f"vLLM launcher identity mismatch: {binary_sha256}")
    vllm_python = config.vllm_bin.parent / "python"
    if (
        not vllm_python.is_symlink()
        or vllm_python.resolve() != EVALSCOPE_BASE_PYTHON.resolve()
        or sha256_file(vllm_python.resolve()) != EVALSCOPE_BASE_PYTHON_SHA256
    ):
        raise GeometryError(f"vLLM Python identity mismatch: {vllm_python}")
    vllm_root = resolve_vllm_package_root(config)
    site_packages = vllm_root.parent
    return {
        "corpora": {"measured": measured, "warmup": warmup},
        "model": validate_hashed_files(
            config.model_dir, MODEL_FILE_SHA256, "model"
        ),
        "vllm": {
            "launcher": {
                "path": str(config.vllm_bin),
                "sha256": binary_sha256,
            },
            "python": {
                "path": str(vllm_python),
                "target": str(vllm_python.resolve()),
                "target_sha256": EVALSCOPE_BASE_PYTHON_SHA256,
                "version": list(EVALSCOPE_PYTHON_VERSION),
            },
            "source": validate_hashed_files(
                vllm_root,
                VLLM_SOURCE_SHA256,
                "vLLM source",
            ),
        },
        "backend_packages": {
            "versions": {"flashinfer-python": "0.6.14", "triton": "3.6.0"},
            "metadata": validate_hashed_files(
                site_packages,
                PACKAGE_METADATA_SHA256,
                "backend package metadata",
            ),
            "trees": {
                name: validate_directory_tree(
                    site_packages / name,
                    expected_sha256,
                    f"backend package {name}",
                )
                for name, expected_sha256 in sorted(PACKAGE_TREE_SHA256.items())
            },
        },
    }


def resolve_evalscope_runtime(config: GeometryConfig) -> EvalScopeRuntime:
    if config.uvx_bin.is_symlink() or config.uvx_bin.absolute() != config.uvx_bin.resolve():
        raise GeometryError(f"uvx path is not canonical: {config.uvx_bin}")
    uvx_sha256 = sha256_file(config.uvx_bin)
    if uvx_sha256 != UVX_SHA256:
        raise GeometryError(f"uvx identity mismatch: {uvx_sha256}")
    if (
        EVALSCOPE_BASE_PYTHON.is_symlink()
        or not EVALSCOPE_BASE_PYTHON.is_file()
        or sha256_file(EVALSCOPE_BASE_PYTHON) != EVALSCOPE_BASE_PYTHON_SHA256
    ):
        raise GeometryError(
            f"EvalScope base Python identity mismatch: {EVALSCOPE_BASE_PYTHON}"
        )

    environment = build_environment(MACROCHUNK_BUDGETS[0], create=True)
    pycache_before = validate_empty_pycache_prefix(MACROCHUNK_BUDGETS[0])
    version_process = subprocess.run(
        [str(config.uvx_bin), "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=30.0,
    )
    observed_uvx_version = version_process.stdout.strip()
    if version_process.returncode != 0 or observed_uvx_version != UVX_VERSION:
        raise GeometryError(
            "uvx version mismatch: "
            f"status={version_process.returncode}, value={observed_uvx_version!r}"
        )

    bootstrap_code = (
        "import json,sys; "
        "print(json.dumps({'prefix':sys.prefix,'executable':sys.executable,"
        "'version':list(sys.version_info[:3])},sort_keys=True))"
    )
    bootstrap_command = [
        str(config.uvx_bin),
        "--offline",
        "--no-config",
        "--no-python-downloads",
        "--isolated",
        "--python",
        str(EVALSCOPE_BASE_PYTHON),
        "--from",
        EVALSCOPE_REQUIREMENT,
        "python",
        "-c",
        bootstrap_code,
    ]
    bootstrap = subprocess.run(
        bootstrap_command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=180.0,
    )
    try:
        resolved = json.loads(bootstrap.stdout)
    except json.JSONDecodeError as error:
        raise GeometryError(
            "uvx did not return one parseable EvalScope environment receipt: "
            f"status={bootstrap.returncode}, stderr={bootstrap.stderr!r}"
        ) from error
    if bootstrap.returncode != 0 or resolved.get("version") != list(
        EVALSCOPE_PYTHON_VERSION
    ):
        raise GeometryError(
            "EvalScope Python resolution mismatch: "
            f"status={bootstrap.returncode}, receipt={resolved!r}"
        )
    prefix = pathlib.Path(str(resolved.get("prefix", "")))
    python = pathlib.Path(str(resolved.get("executable", "")))
    receipt = validate_evalscope_environment(prefix)
    if python != prefix / "bin/python":
        raise GeometryError(f"EvalScope resolved an unexpected Python: {python}")
    executable = prefix / "bin/evalscope"

    version = subprocess.run(
        [str(executable), "--version"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=180.0,
    )
    expected = f"evalscope {EVALSCOPE_VERSION}"
    if version.returncode != 0 or version.stdout.strip() != expected:
        raise GeometryError(
            "EvalScope version mismatch: "
            f"status={version.returncode}, value={version.stdout.strip()!r}"
        )
    receipt["bootstrap"] = {
        "command": bootstrap_command,
        "uvx": {
            "path": str(config.uvx_bin),
            "sha256": uvx_sha256,
            "version": observed_uvx_version,
        },
        "requirement": EVALSCOPE_REQUIREMENT,
        "offline": True,
        "isolated": True,
        "no_config": True,
        "no_python_downloads": True,
    }
    receipt["version"] = EVALSCOPE_VERSION
    receipt["pycache_prefix"] = {
        "before": pycache_before,
        "after": validate_empty_pycache_prefix(MACROCHUNK_BUDGETS[0]),
    }
    return EvalScopeRuntime(prefix, python, executable, receipt)


def run(config: GeometryConfig) -> dict[str, Any]:
    identity = None if config.dry_run else git_identity()
    static_inputs = validate_static_inputs(config)
    evalscope_runtime = resolve_evalscope_runtime(config)
    plan = build_plan(config, evalscope_runtime)
    versions = {
        "vllm": VLLM_VERSION,
        "evalscope": EVALSCOPE_VERSION,
        "uvx": UVX_VERSION,
        "torch": TORCH_VERSION,
        "torch_cuda": TORCH_CUDA_VERSION,
    }
    if config.dry_run:
        return {
            "plan": plan,
            "static_inputs": static_inputs,
            "evalscope_runtime": evalscope_runtime.receipt,
            "host_runtime": {
                "status": "deferred",
                "reason": (
                    "the CUDA device probe is admitted only inside the formal "
                    "clean-host preflight sandwich"
                ),
            },
            "versions": versions,
            "dry_run": True,
        }

    assert identity is not None
    config.output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "q3x_vllm_p40_macrochunk_geometry",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "valid": False,
        "repository": identity,
        "versions": versions,
        "static_inputs": static_inputs,
        "evalscope_runtime": evalscope_runtime.receipt,
        "host_runtime": None,
        "host_runtime_admission": None,
        "plan": plan,
        "results": [],
    }
    write_json(config.output_dir / "manifest.json", manifest)
    try:
        before_host_probe = run_preflight(
            config,
            config.output_dir / "preflight-before-host-runtime.json",
            (),
        )
        host_runtime = validate_host_runtime(config)
        after_host_probe = run_preflight(
            config,
            config.output_dir / "preflight-after-host-runtime.json",
            (),
        )
        manifest["host_runtime"] = host_runtime
        manifest["host_runtime_admission"] = {
            "before_probe": before_host_probe["decision"],
            "after_probe": after_host_probe["decision"],
            "probe_context_released": True,
        }
        write_json(config.output_dir / "manifest.json", manifest)
        for budget in MACROCHUNK_BUDGETS:
            manifest["results"].append(
                run_budget(config, evalscope_runtime, budget)
            )
            write_json(config.output_dir / "manifest.json", manifest)
        final_evalscope = validate_evalscope_environment(evalscope_runtime.prefix)
        initial_evalscope = {
            key: evalscope_runtime.receipt[key]
            for key in (
                "prefix",
                "python",
                "executable",
                "site_packages",
                "distribution_manifest",
            )
        }
        if final_evalscope != initial_evalscope:
            raise GeometryError("EvalScope environment changed during the witness")
        manifest["evalscope_runtime_unchanged"] = True
        manifest["pycache_prefix_empty_after_witness"] = (
            validate_empty_pycache_prefix(MACROCHUNK_BUDGETS[0])
        )
    except BaseException as error:
        manifest["failure"] = repr(error)
        try:
            write_json(config.output_dir / "manifest.json", manifest)
        except BaseException as receipt_error:
            try:
                error.add_note(
                    "manifest failure receipt could not be written: "
                    f"{receipt_error!r}"
                )
            except (AttributeError, TypeError):
                pass
        raise
    manifest["valid"] = True
    manifest["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> GeometryConfig:
    parser = argparse.ArgumentParser(
        description="Run the fixed stock-vLLM P40 macrochunk geometry witness"
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vllm-bin", type=pathlib.Path, default=DEFAULT_VLLM_BIN)
    parser.add_argument("--uvx-bin", type=pathlib.Path, default=None)
    parser.add_argument("--port", type=int, default=18093)
    parser.add_argument("--allow-pid", action="append", type=int, default=[])
    parser.add_argument("--readiness-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=680.0)
    parser.add_argument("--preflight-samples", type=int, default=5)
    parser.add_argument("--preflight-interval-ms", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    uvx = args.uvx_bin or shutil.which("uvx")
    if uvx is None:
        raise GeometryError("uvx is not available on PATH; pass --uvx-bin")
    if args.port < 1024 or args.port > 65535:
        raise GeometryError("--port must be in [1024,65535]")
    if args.readiness_timeout <= 0 or args.request_timeout <= 0:
        raise GeometryError("timeouts must be positive")
    if args.preflight_samples < 3:
        raise GeometryError("--preflight-samples must be at least 3")
    if args.preflight_interval_ms < 100:
        raise GeometryError("--preflight-interval-ms must be at least 100")
    if any(pid <= 0 for pid in args.allow_pid):
        raise GeometryError("--allow-pid values must be positive")
    output = validate_output_dir(args.output_dir, may_exist=args.dry_run)
    return GeometryConfig(
        output_dir=output,
        model_dir=args.model_dir.expanduser().resolve(),
        vllm_bin=args.vllm_bin.expanduser().resolve(),
        uvx_bin=pathlib.Path(uvx).expanduser().resolve(),
        port=args.port,
        allow_pids=tuple(sorted(set(args.allow_pid))),
        readiness_timeout_seconds=args.readiness_timeout,
        request_timeout_seconds=args.request_timeout,
        preflight_samples=args.preflight_samples,
        preflight_interval_ms=args.preflight_interval_ms,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        result = run(config)
    except (GeometryError, OSError, subprocess.SubprocessError) as error:
        print(f"vllm-p40-geometry: FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
