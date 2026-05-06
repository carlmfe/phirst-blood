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

One workflow per backend tests that the library compiles across all supported
architectures. Find them in `.github/workflows/compile-check-*.yml`.

| Workflow | Backend | Architectures tested |
|----------|---------|---------------------|
| `compile-check-cuda.yml` | CUDA | All NVIDIA archs (sm_70 – sm_90) |
| `compile-check-alpaka.yml` | Alpaka/CUDA | All NVIDIA archs (sm_70 – sm_90) |
| `compile-check-sycl.yml` | SYCL/CUDA | All NVIDIA archs (sm_70 – sm_90) |
| `compile-check-kokkos.yml` | Kokkos | Single build (arch baked into module) |
| `compile-check-hip.yml` | HIP | All AMD archs (disabled — no runner yet) |

AMD and Intel architectures are defined in `architectures.yml` but their CI steps
are not yet active because:
- **AMD**: requires a self-hosted runner with ROCm + Kokkos/HIP or SYCL/AMD toolchain,
  and CMakeLists.txt needs AMD AOT flags added (C++ Development task)
- **Intel**: requires a self-hosted runner with Intel oneAPI DPC++ and AOT support
  in CMakeLists.txt (C++ Development task)

## Runner Labels

| Label | Hardware | Available |
|-------|----------|-----------|
| `self-hosted-gpu` | NVIDIA GPU (RTX 2000 Ada, sm_89) | ✅ Yes |
| `self-hosted-amd` | AMD GPU with ROCm | ❌ Not yet |
| `self-hosted-intel` | Intel GPU with oneAPI | ❌ Not yet |

## Adding a New Architecture

1. Add an entry to `architectures.yml` with all required fields
2. For NVIDIA: add the arch value to the matrix in the relevant `compile-check-*.yml`
3. For AMD/Intel: add the entry with `runner_requirement: self-hosted-amd/intel` and
   a `note` explaining any CMakeLists.txt work needed before it can be CI-tested
4. Update the table in this README

## Adding a New Runner

When a new runner becomes available (AMD or Intel):
1. Register the runner with the appropriate label (`self-hosted-amd` or `self-hosted-intel`)
2. Complete any pending C++ Development tasks (AMD/Intel AOT flags in CMakeLists.txt)
3. Uncomment the relevant `if: false` steps in `compile-check-sycl.yml` and/or add
   a new `compile-check-kokkos-amd.yml` workflow
