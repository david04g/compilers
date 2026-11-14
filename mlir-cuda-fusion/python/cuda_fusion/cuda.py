"""Minimal NVRTC + CUDA Driver API bindings; no PyTorch, CUDA SDK, or nvcc needed."""
import ctypes as C
from importlib import metadata
import os
from pathlib import Path
import time


class CUDAUnavailable(RuntimeError):
    pass


def _bind(lib, name, args):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, C.c_int
    return fn


P = C.c_void_p
I = C.c_int
U = C.c_uint
Z = C.c_size_t
D = C.c_uint64


class NVRTC:
    def __init__(self):
        candidates = []
        if os.environ.get("NVRTC_LIBRARY"):
            candidates.append(Path(os.environ["NVRTC_LIBRARY"]))
        try:
            dist = metadata.distribution("nvidia-cuda-nvrtc-cu12")
            candidates += [Path(dist.locate_file(f)) for f in dist.files
                           if (f.name.startswith("nvrtc64_") and f.suffix == ".dll" and ".alt." not in f.name)
                           or f.name.startswith("libnvrtc.so")]
        except metadata.PackageNotFoundError:
            pass
        if os.environ.get("CUDA_PATH"):
            root = Path(os.environ["CUDA_PATH"])
            candidates += [p for p in (root / "bin").glob("nvrtc64_*.dll") if ".alt." not in p.name]
            candidates += list((root / "lib64").glob("libnvrtc.so*"))
        self._dll_dirs = []
        self._builtins = []
        self.lib = None
        errors = []
        for path in candidates:
            try:
                if os.name == "nt":
                    self._dll_dirs.append(os.add_dll_directory(str(path.parent.resolve())))
                    # NVRTC uses its own DLL lookup; preload its companion by
                    # absolute path for Python 3.8+'s restricted DLL search.
                    for builtin in path.parent.glob("nvrtc-builtins64_*.dll"):
                        self._builtins.append(C.CDLL(str(builtin.resolve())))
                self.lib = C.CDLL(str(path.resolve()))
                break
            except OSError as e:
                errors.append(str(e))
        if self.lib is None:
            raise CUDAUnavailable("NVRTC unavailable: install the [cuda] extra or set NVRTC_LIBRARY. " + "; ".join(errors))
        self.version_fn = _bind(self.lib, "nvrtcVersion", [C.POINTER(I), C.POINTER(I)])
        self.create = _bind(self.lib, "nvrtcCreateProgram", [C.POINTER(P), C.c_char_p, C.c_char_p, I, P, P])
        self.compile_fn = _bind(self.lib, "nvrtcCompileProgram", [P, I, C.POINTER(C.c_char_p)])
        self.log_size = _bind(self.lib, "nvrtcGetProgramLogSize", [P, C.POINTER(Z)])
        self.log_fn = _bind(self.lib, "nvrtcGetProgramLog", [P, P])
        self.size_fn = _bind(self.lib, "nvrtcGetCUBINSize", [P, C.POINTER(Z)])
        self.cubin_fn = _bind(self.lib, "nvrtcGetCUBIN", [P, P])
        self.destroy = _bind(self.lib, "nvrtcDestroyProgram", [C.POINTER(P)])
        major, minor = I(), I()
        self._check(self.version_fn(C.byref(major), C.byref(minor)))
        self.version = (major.value, minor.value)

    @staticmethod
    def _check(result):
        if result:
            raise RuntimeError(f"NVRTC error {result}")

    def compile(self, source, arch):
        if arch < 75 and self.version[0] >= 13:
            raise CUDAUnavailable("Pascal requires NVRTC/CUDA 12.9 or earlier")
        program = P()
        self._check(self.create(C.byref(program), source.encode(), b"fusion.cu", 0, None, None))
        started = time.perf_counter()
        try:
            options = [f"--gpu-architecture=sm_{arch}", "--std=c++14", "--fmad=false"]
            argv = (C.c_char_p * len(options))(*(s.encode() for s in options))
            result = self.compile_fn(program, len(options), argv)
            size = Z()
            self._check(self.log_size(program, C.byref(size)))
            log = C.create_string_buffer(max(size.value, 1))
            self._check(self.log_fn(program, log))
            if result:
                raise RuntimeError(f"NVRTC compilation failed ({result}):\n{log.value.decode()}")
            self._check(self.size_fn(program, C.byref(size)))
            cubin = C.create_string_buffer(size.value)
            self._check(self.cubin_fn(program, cubin))
            return cubin.raw, log.value.decode(), (time.perf_counter() - started) * 1000, options
        finally:
            self.destroy(C.byref(program))


class CUDA:
    def __init__(self):
        try:
            self.lib = C.WinDLL("nvcuda.dll") if os.name == "nt" else C.CDLL("libcuda.so.1")
        except OSError as e:
            raise CUDAUnavailable(f"CUDA driver unavailable: {e}") from e
        signatures = {
            "cuInit": [U], "cuDeviceGet": [C.POINTER(I), I],
            "cuDeviceGetName": [P, I, I], "cuDeviceGetAttribute": [C.POINTER(I), I, I],
            "cuDriverGetVersion": [C.POINTER(I)],
            "cuDevicePrimaryCtxRetain": [C.POINTER(P), I],
            "cuDevicePrimaryCtxRelease_v2": [I], "cuCtxSetCurrent": [P],
            "cuCtxGetCurrent": [C.POINTER(P)], "cuCtxSynchronize": [],
            "cuStreamCreate": [C.POINTER(P), U], "cuStreamDestroy_v2": [P],
            "cuMemAlloc_v2": [C.POINTER(D), Z], "cuMemFree_v2": [D],
            "cuMemcpyHtoD_v2": [D, P, Z], "cuMemcpyDtoH_v2": [P, D, Z],
            "cuModuleLoadData": [C.POINTER(P), P], "cuModuleUnload": [P],
            "cuModuleGetFunction": [C.POINTER(P), P, C.c_char_p],
            "cuFuncGetAttribute": [C.POINTER(I), I, P],
            "cuLaunchKernel": [P, U, U, U, U, U, U, U, P, C.POINTER(P), P],
            "cuOccupancyMaxActiveBlocksPerMultiprocessor": [C.POINTER(I), P, I, Z],
            "cuStreamBeginCapture": [P, I], "cuStreamEndCapture": [P, C.POINTER(P)],
            "cuGraphInstantiateWithFlags": [C.POINTER(P), P, C.c_ulonglong],
            "cuGraphLaunch": [P, P], "cuGraphExecDestroy": [P], "cuGraphDestroy": [P],
            "cuEventCreate": [C.POINTER(P), U], "cuEventRecord": [P, P],
            "cuEventSynchronize": [P], "cuEventElapsedTime": [C.POINTER(C.c_float), P, P],
            "cuEventDestroy_v2": [P], "cuGetErrorName": [I, C.POINTER(C.c_char_p)]}
        for name, args in signatures.items():
            setattr(self, name, _bind(self.lib, name, args))
        try:
            self.check(self.cuInit(0))
            self.device = I()
            self.check(self.cuDeviceGet(C.byref(self.device), 0))
        except RuntimeError as e:
            raise CUDAUnavailable(str(e)) from e
        self.previous = P()
        self.check(self.cuCtxGetCurrent(C.byref(self.previous)))
        self.context, self.stream = P(), P()
        self.check(self.cuDevicePrimaryCtxRetain(C.byref(self.context), self.device))
        self.check(self.cuCtxSetCurrent(self.context))
        self.check(self.cuStreamCreate(C.byref(self.stream), 1))
        self.allocations, self.modules, self.graphs, self.executables, self.events = [], [], [], [], []
        name = C.create_string_buffer(256)
        self.check(self.cuDeviceGetName(name, 256, self.device))
        driver = I()
        self.check(self.cuDriverGetVersion(C.byref(driver)))
        self.arch = 10 * self.device_attr(75) + self.device_attr(76)
        self.info = {"name": name.value.decode(), "compute_capability": self.arch,
                     "cuda_driver_api_version": driver.value, "sm_count": self.device_attr(16),
                     "max_threads_per_sm": self.device_attr(39)}
        self.closed = False

    def check(self, result):
        if result:
            name = C.c_char_p()
            self.cuGetErrorName(result, C.byref(name))
            raise RuntimeError(f"CUDA error {result}: {(name.value or b'unknown').decode()}")

    def device_attr(self, attr):
        value = I()
        self.check(self.cuDeviceGetAttribute(C.byref(value), attr, self.device))
        return value.value

    def malloc(self, size):
        ptr = D()
        self.check(self.cuMemAlloc_v2(C.byref(ptr), max(4, size)))
        self.allocations.append(ptr)
        return ptr

    def upload(self, array):
        if not array.flags.c_contiguous:
            raise ValueError("expected contiguous array")
        ptr = self.malloc(array.nbytes)
        self.check(self.cuMemcpyHtoD_v2(ptr, array.ctypes.data, array.nbytes))
        # Pageable HtoD may return after staging. Our nonblocking compute stream
        # does not inherit default-stream dependencies, so finish the transfer.
        self.check(self.cuCtxSynchronize())
        return ptr

    def download(self, ptr, array):
        self.check(self.cuCtxSynchronize())
        self.check(self.cuMemcpyDtoH_v2(array.ctypes.data, ptr, array.nbytes))

    def load(self, cubin, launches):
        module = P()
        buffer = C.create_string_buffer(cubin)
        self.check(self.cuModuleLoadData(C.byref(module), buffer))
        self.modules.append(module)
        loaded, resources = [], []
        for launch in launches:
            fn = P()
            self.check(self.cuModuleGetFunction(C.byref(fn), module, launch["name"].encode()))
            attrs = {}
            for name, attr in [("registers_per_thread", 4), ("local_bytes_per_thread", 3),
                               ("static_shared_bytes", 1)]:
                val = I()
                self.check(self.cuFuncGetAttribute(C.byref(val), attr, fn))
                attrs[name] = val.value
            blocks = I()
            self.check(self.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                C.byref(blocks), fn, launch["threads"], launch["shared"]))
            attrs.update(kernel=launch["name"], active_blocks_per_sm=blocks.value,
                         theoretical_occupancy=blocks.value * launch["threads"] / self.info["max_threads_per_sm"],
                         dynamic_shared_bytes=launch["shared"])
            resources.append(attrs)
            loaded.append((fn, launch))
        return loaded, resources

    def launch(self, loaded, pointers):
        args = (P * len(pointers))(*(C.cast(C.byref(p), P) for p in pointers))
        for fn, spec in loaded:
            self.check(self.cuLaunchKernel(fn, spec["grid"], 1, 1, spec["threads"], 1, 1,
                                          spec["shared"], self.stream, args, None))

    def capture(self, loaded, pointers, repetitions):
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        self.check(self.cuStreamBeginCapture(self.stream, 0))
        try:
            for _ in range(repetitions):
                self.launch(loaded, pointers)
        except BaseException:
            abandoned = P()
            self.cuStreamEndCapture(self.stream, C.byref(abandoned))
            if abandoned:
                self.cuGraphDestroy(abandoned)
            raise
        graph, executable = P(), P()
        self.check(self.cuStreamEndCapture(self.stream, C.byref(graph)))
        self.graphs.append(graph)
        self.check(self.cuGraphInstantiateWithFlags(C.byref(executable), graph, 0))
        self.executables.append(executable)
        return executable

    def time_graph_us(self, executable, repetitions):
        start, stop = P(), P()
        self.check(self.cuEventCreate(C.byref(start), 0))
        self.events.append(start)
        self.check(self.cuEventCreate(C.byref(stop), 0))
        self.events.append(stop)
        self.check(self.cuEventRecord(start, self.stream))
        self.check(self.cuGraphLaunch(executable, self.stream))
        self.check(self.cuEventRecord(stop, self.stream))
        self.check(self.cuEventSynchronize(stop))
        ms = C.c_float()
        self.check(self.cuEventElapsedTime(C.byref(ms), start, stop))
        return ms.value * 1000 / repetitions

    def close(self):
        if self.closed:
            return
        self.cuCtxSetCurrent(self.context)
        self.cuCtxSynchronize()
        for obj in reversed(self.events): self.cuEventDestroy_v2(obj)
        for obj in reversed(self.executables): self.cuGraphExecDestroy(obj)
        for obj in reversed(self.graphs): self.cuGraphDestroy(obj)
        for obj in reversed(self.modules): self.cuModuleUnload(obj)
        for obj in reversed(self.allocations): self.cuMemFree_v2(obj)
        self.cuStreamDestroy_v2(self.stream)
        self.cuDevicePrimaryCtxRelease_v2(self.device)
        self.cuCtxSetCurrent(self.previous)
        self.closed = True

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
