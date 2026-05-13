# Phirst Blood — Deployment Architecture Targets

This folder defines the GPU architectures that Phirst Blood actively targets,
and the infrastructure required to compile and test each backend on each architecture.

## `architectures.yml`

The canonical list of all target architectures. Each entry specifies:

- **id** — machine-readable key used as a matrix value in CI workflows
- **name** — human-readable description
- **family** — `nvidia`, `amd`, or `intel`
- **class** — `hpc`, `desktop`, or `cloud`
- **runner_requirement** — which self-hosted runner label can build for this arch
- **backends** — which Phirst Blood backends can target this architecture, and with
  what CMake flags

## CI Compile-Check Workflows

All compile checks are consolidated in `.github/workflows/compile-check.yml` with
five jobs, each loading modules once and running all relevant architectures sequentially.
sm_70 (Volta) is excluded from all NVIDIA jobs — CUDA 13.0 dropped sm_70 support.

| Job | Backends | Architectures tested |
|-----|----------|---------------------|
| `nvidia-toolchains` | CUDA, Alpaka/CUDA, Kokkos/CUDA | sm_75, sm_80, sm_86, sm_89, sm_90 |
| `hip-toolchains` | HIP, Kokkos/HIP, Alpaka/HIP | gfx908, gfx90a, gfx942, gfx1100 |
| `sycl-cuda` | SYCL/CUDA | sm_75, sm_80, sm_86, sm_89, sm_90 |
| `sycl-intel` | SYCL/Intel (JIT), Kokkos/SYCL | JIT — all Intel GPUs at runtime |
| `sycl-amd` | SYCL/AMD (AdaptiveCpp) | gfx908, gfx90a, gfx942, gfx1100 |

HIP/Alpaka/HIP/Kokkos/HIP cross-compilation runs on the `[self-hosted, gpu]` runner —
ROCm 6.3 provides `amdclang++` which can target any `gfx*` offline. SYCL/AMD uses
AdaptiveCpp (provided by the `rocm/6.3` module).

## Runner Labels

| Labels | Hardware | Available |
|--------|----------|-----------|
| `[self-hosted, gpu]` | NVIDIA GPU (RTX 2000 Ada, sm_89) | ✅ Yes (also runs HIP/SYCL/AMD cross-compilation) |
| `[self-hosted, amd]` | AMD GPU with ROCm | ❌ Not yet (needed for HIP execution tests) |
| `[self-hosted, intel]` | Intel GPU with oneAPI | ❌ Not yet (needed for SYCL/Intel execution tests) |

## Adding a New Architecture

1. Add an entry to `architectures.yml` with all required fields
2. For NVIDIA: add the arch value to the relevant steps in `compile-check.yml`
3. For AMD/Intel: add the entry with `runner_requirement: self-hosted-amd/intel` and
   a `note` explaining any CMakeLists.txt work needed before it can be CI-tested
4. Update the table in this README

## Adding a New Runner

When a new runner becomes available (AMD or Intel):
1. Register the runner with labels `self-hosted` + `amd` (or `intel`)
2. Add a build-and-test job to `.github/workflows/build.yml` for that hardware
3. Update runner availability in this README
