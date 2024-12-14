import numpy as np
import cupy
import contextlib
import nvidia_smi
import gpu_mm
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

# The current CuBuffer needs to be allocated with a fixed size. This
# avoids all garbage collection, but it's inconvenient to need to know
# the size beforehand. An alternative is to allow the buffer to grow if
# it's too small (but not shrink automatically). This will cause garbage
# overhead when it needs to reallocate, but once the buffer reaches its
# final size, then this overhead will disappear. Let's generalize it to
# allow this, and make this the default

class CuBuffer:
	def __init__(self, name="[unnamed]", size=512, align=512, growth_factor=1.5, allocator=None):
		"""Initialize a Cubuffer, a growable memory region for storing data products
		on the gpu. The purpose of this class is to minimize cupy memory allocations.
		Each buffer is meant to be reused for a given purpose, e.g. a TOD buffer
		would be used to hold time-ordered data any time that's needed in a calculation.
		Once buffer is as big as the biggest array it needs to represent, then no
		further allocations will be done.

		Arguments:
		* size:  Initial size of this buffer, in bytes
		* align: Alignment in bytes. Only relevant for the allocator interface
		* name:  Name of this buffer. Useful for error messages
		* growth_factor: size is multiplied by this factor when growing.
		* allocator: How to allocate memory. If None, uses the cuda allocator active at the
		  time the buffer is initialized"""
		self.size   = int(size)
		self.memptr = mempool.malloc(self.size)
		self.align  = align
		self.name   = name
		# FIXME: We're not using growth_factor when growing
		self.growth_factor = growth_factor
		self.allocator = allocator or cupy.cuda.get_allocator()
	def empty(self, shape, dtype=np.float32):
		"""Return an empty cupy array with the given shape and data type.
		If the buffer is too small to represent such an array, then it will
		grow to accomodate it, but otherwise no allocation is performed.
		Invalidates previous views"""
		self.reserve(np.prod(shape)*np.dtype(dtype).itemsize)
		#from .logging import L
		#L.print("%s buffer overwritten" % (self.name), level=3)
		return cupy.ndarray(shape, dtype, memptr=cupy.cuda.MemoryPointer(self.memptr.mem, 0))
	def full(self, shape, val, dtype=np.float32):
		arr    = self.empty(shape, dtype=dtype)
		arr[:] = val
		return arr
	def zeros(self, shape, dtype=np.float32):
		return self.full(shape, 0, dtype=dtype)
	def array(self, arr):
		"""Like empty(), but based on the data of an existing array, which
		can be either a numpy or cupy array."""
		res = self.empty(arr.shape, arr.dtype)
		copy(arr, res)
		return res
	def as_allocator(self): return BufAllocator(self)
	def reserve(self, size):
		"""Grow buffer to given size, copying over data as necessary"""
		if size <= self.size: return
		from .logging import L
		# This allocator stuff is to avoid an infinite loop when
		# used with BufAllocator
		L.print("%s buffer increased to %.4f GB" % (self.name, size/1024**3), level=2)
		memptr = self.allocator(size)
		cupy.cuda.runtime.memcpy(memptr.ptr, self.memptr.ptr, self.size, cupy.cuda.runtime.memcpyDefault)
		self.memptr = memptr
		self.size   = size
	def aligned_size(self, *sizes): return aligned_size(*sizes, align=self.align)
	def register(self):
		scratch[self.name] = self
		return self

def copy(afrom,ato):
	cupy.cuda.runtime.memcpy(getptr(ato), getptr(afrom), ato.nbytes, cupy.cuda.runtime.memcpyDefault)

def getptr(arr):
	try: return arr.data.ptr
	except AttributeError: return arr.ctypes.data

def aligned_size(*sizes, align=512):
	return sum([round_up(size,align) for size in sizes])

class BufAllocator:
	"""BufAllocator allows one to use the memory in a CuBuffer as a
	memory allocator arena. Used to implement CuBuffer.as_allocator.
	Invalidated by CuBuffer.empty() or CuBuffer.array()"""
	def __init__(self, cubuf):
		self.buf    = cubuf
		self.offset = 0
	@property
	def size(self):  return self.buf.size
	@property
	def align(self): return self.buf.align
	def free(self):  return self.size-self.offset
	def alloc(self, size):
		#from .logging import L
		#L.print("%s buffer overwritten" % (self.buf.name), level=3)
		if self.free() < size:
			raise MemoryError("BufAllocator on CuBuffer %s: not enough memory to allocate %d bytes" % (self.buf.name, size))
		res = cupy.cuda.MemoryPointer(self.buf.memptr.mem, self.offset)
		self.offset = round_up(self.offset+size, self.align)
		return res
	def __enter__(self):
		self._old_allocator = cupy.cuda.get_allocator()
		cupy.cuda.set_allocator(self.alloc)
		return self
	def __exit__(self, exc_type, exc_value, traceback):
		cupy.cuda.set_allocator(self._old_allocator)

def round_up(a,b): return (a+b-1)//b*b

# Old CuBuffer. Can probably be deleted

class OldCuBuffer:
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
