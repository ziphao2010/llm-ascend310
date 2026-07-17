"""
Low-level Ascend 310 ACL bindings.
Wraps CANN C API (libascendcl.so) into Python classes.
"""
import ctypes, numpy as np
from ctypes import c_void_p, c_size_t, c_int, c_uint32, c_char_p, byref, CDLL, POINTER
from typing import Dict, List, Optional, Tuple


class Device:
    """
    Ascend 310 NPU device wrapper.
    Each chip is a separate device (0-3 for Atlas 300I 3010).

    Usage:
        dev = Device(0)
        ptr = dev.malloc(4096)
        dev.free(ptr)
    """
    _initialized = False

    def __init__(self, device_id: int):
        self.dev = device_id
        self.L = CDLL("libascendcl.so")
        self._setup_funcs()
        if not Device._initialized:
            self.L.aclInit(None)
            Device._initialized = True
        self.L.aclrtSetDevice(device_id)
        self._oc: Dict[str, dict] = {}  # cached .om models

    def _setup_funcs(self):
        L, P = self.L, POINTER
        for name, args, ret in [
            ("aclrtMalloc",           [P(c_void_p), c_size_t, c_int], c_int),
            ("aclrtFree",             [c_void_p], c_int),
            ("aclrtMemcpy",           [c_void_p, c_size_t, c_void_p, c_size_t, c_int], c_int),
            ("aclrtMemset",           [c_void_p, c_size_t, c_int, c_size_t], c_int),
            ("aclrtSetDevice",        [c_int], c_int),
            ("aclrtGetDevice",        [P(c_int)], c_int),
            ("aclrtGetMemInfo",       [P(c_size_t), P(c_size_t)], c_int),
            ("aclmdlLoadFromFile",    [c_char_p, P(c_uint32)], c_int),
            ("aclmdlCreateDesc",      [], c_void_p),
            ("aclmdlGetDesc",         [c_void_p, c_uint32], c_int),
            ("aclmdlGetNumInputs",    [c_void_p], c_size_t),
            ("aclmdlGetNumOutputs",   [c_void_p], c_size_t),
            ("aclmdlGetInputSizeByIndex",  [c_void_p, c_size_t], c_size_t),
            ("aclmdlGetOutputSizeByIndex", [c_void_p, c_size_t], c_size_t),
            ("aclmdlCreateDataset",   [], c_void_p),
            ("aclmdlDestroyDataset",  [c_void_p], None),
            ("aclmdlAddDatasetBuffer",[c_void_p, c_void_p], c_int),
            ("aclmdlExecute",         [c_uint32, c_void_p, c_void_p], c_int),
            ("aclCreateDataBuffer",   [c_void_p, c_size_t], c_void_p),
        ]:
            f = getattr(L, name)
            f.argtypes = args
            f.restype = ret

    def set_device(self):
        """Set this device as the active context."""
        self.L.aclrtSetDevice(self.dev)

    def malloc(self, size_bytes: int) -> int:
        """Allocate NPU memory. Returns device pointer (int)."""
        p = c_void_p(0)
        ret = self.L.aclrtMalloc(byref(p), size_bytes, 1)
        if ret != 0:
            raise RuntimeError(f"aclrtMalloc({size_bytes}) failed: {ret}")
        return p.value

    def free(self, ptr: int):
        """Free NPU memory."""
        if ptr:
            self.L.aclrtFree(c_void_p(ptr))

    def h2d(self, dst: int, src_np: np.ndarray):
        """Host → Device copy."""
        self.L.aclrtMemcpy(c_void_p(dst), src_np.nbytes,
                          src_np.ctypes.data_as(c_void_p), src_np.nbytes, 1)

    def d2h(self, dst_np: np.ndarray, src: int, size: int = 0):
        """Device → Host copy. dst_np must be pre-allocated."""
        n = size or dst_np.nbytes
        self.L.aclrtMemcpy(dst_np.ctypes.data_as(c_void_p), n,
                          c_void_p(src), n, 2)

    def d2d(self, dst: int, src: int, size: int):
        """Device → Device copy (same chip or cross-chip)."""
        self.L.aclrtMemcpy(c_void_p(dst), size, c_void_p(src), size, 3)

    def memset(self, ptr: int, size: int, value: int = 0):
        """Set device memory to a constant value."""
        self.L.aclrtMemset(c_void_p(ptr), size, value, size)

    def mem_info(self) -> Tuple[int, int]:
        """Get (free_bytes, total_bytes) for this device."""
        free, total = c_size_t(0), c_size_t(0)
        self.L.aclrtGetMemInfo(byref(free), byref(total))
        return free.value, total.value

    def exec(self, model_name: str, inputs: List[Tuple[int, int]]) -> List[int]:
        """
        Execute a compiled .om model on NPU.

        Args:
            model_name: name without .om (e.g. 'mm_1_1536_4608')
            inputs: list of (device_ptr, size_bytes) tuples

        Returns:
            list of output device pointers
        """
        om_dir = None  # Will be set per-model or globally

        if model_name not in self._oc:
            # Search for .om file
            path = None
            for d in ["/root/llm-ascend310/om_models",
                       "/root/qwythos_engine/om_models"]:
                import os
                p = f"{d}/{model_name}.om"
                if os.path.exists(p):
                    path = p
                    break
            if path is None:
                raise FileNotFoundError(f"om model not found: {model_name}.om")

            mid = c_uint32(0)
            ret = self.L.aclmdlLoadFromFile(path.encode(), byref(mid))
            if ret != 0:
                raise RuntimeError(f"aclmdlLoadFromFile({path}) failed: {ret}")
            desc = self.L.aclmdlCreateDesc()
            self.L.aclmdlGetDesc(desc, mid.value)
            self._oc[model_name] = {"id": mid.value, "desc": desc}

        om = self._oc[model_name]
        desc = om["desc"]
        in_ds = self.L.aclmdlCreateDataset()
        out_ds = self.L.aclmdlCreateDataset()

        for ptr, sz in inputs:
            buf = self.L.aclCreateDataBuffer(c_void_p(ptr), sz)
            self.L.aclmdlAddDatasetBuffer(in_ds, buf)

        num_out = self.L.aclmdlGetNumOutputs(desc)
        out_ptrs = []
        for i in range(num_out):
            sz = self.L.aclmdlGetOutputSizeByIndex(desc, i)
            p = c_void_p(0)
            self.L.aclrtMalloc(byref(p), sz, 1)
            out_ptrs.append(p.value)
            buf = self.L.aclCreateDataBuffer(c_void_p(p.value), sz)
            self.L.aclmdlAddDatasetBuffer(out_ds, buf)

        ret = self.L.aclmdlExecute(om["id"], in_ds, out_ds)
        self.L.aclmdlDestroyDataset(in_ds)
        self.L.aclmdlDestroyDataset(out_ds)

        if ret != 0:
            for p in out_ptrs:
                self.L.aclrtFree(c_void_p(p))
            raise RuntimeError(f"{model_name} exec failed: {ret}")

        return out_ptrs
