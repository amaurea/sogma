import numpy as np
import cupy
import contextlib
import nvidia_smi
from pixell import bunch

nvidia_smi.nvmlInit()
mempool = cupy.get_default_memory_pool()
scratch = bunch.Bunch()
plan_cache = gpu_mm.cufft.PlanCache()

def get_gpu_memuse():
	handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
	info   = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
	return info.used

@contextlib.contextmanager
def leakcheck(name):
	try:
		mem1 = get_gpu_memuse()
		yield
	finally:
		mem2 = get_gpu_memuse()
		print("%s %8.5f GB" % (name, (mem2-mem1)/1e9))

def gpu_garbage_collect():
	mempool.free_all_blocks()

class CuBuffer:
	def __init__(self, size, align=512, name="[unnamed]"):
		self.size   = int(size)
		self.memptr = mempool.malloc(self.size)
		self.offset = 0
		self.align  = align
		self.name   = name
	def free(self): return self.size - self.offset
	def view(self, shape, dtype=np.float32):
		return cupy.ndarray(shape, dtype, memptr=cupy.cuda.MemoryPointer(self.memptr.mem, self.offset))
	def copy(self, arr):
		if self.free() < arr.nbytes: raise MemoryError("CuBuffer too small to copy array with size %d" % arr.nbytes)
		res = self.view(arr.shape, arr.dtype)
		if isinstance(arr, cupy.ndarray):
			cupy.cuda.runtime.memcpy(res.data.ptr, arr.data.ptr, arr.nbytes, cupy.cuda.runtime.memcpyDefault)
		else:
			cupy.cuda.runtime.memcpy(res.data.ptr, arr.ctypes.data, arr.nbytes, cupy.cuda.runtime.memcpyDefault)
		return res
	def alloc(self, size):
		#print("CuBuffer %s alloc %d offset %d" % (self.name, size, self.offset))
		if self.free() < size:
			raise MemoryError("CuBuffer not enough memory to allocate %d bytes" % size)
		res = cupy.cuda.MemoryPointer(self.memptr.mem, self.offset)
		self.offset = round_up(self.offset+size, self.align)
		return res
	def reset(self):
		self.offset = 0
	@contextlib.contextmanager
	def as_allocator(self):
		offset = self.offset
		try:
			with cupy.cuda.using_allocator(self.alloc):
				yield
		finally:
			self.offset = offset
