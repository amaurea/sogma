import numpy as np
from pixell import device, bunch, fft
from pixell.device import anypy

def get_device(type="auto", align=None, alloc_factory=None, priority=["gpu","cpu"]):
	if type == "auto":
		for name in priority:
			if name in Devices:
				type = name
				break
	cname  = Devices[type]
	device = Devices[type](align=align, alloc_factory=alloc_factory)
	return device

Devices = bunch.Bunch()

try:
	import cupy, gpu_mm
	from cupy.cuda import cublas

	class MMDeviceGpu(device.DeviceGpu):
		def __init__(self, align=None, alloc_factory=None):
			super().__init__(align=align, alloc_factory=alloc_factory)
			# pointing
			self.lib.PointingPrePlan = gpu_mm.PointingPrePlan
			self.lib.PointingPlan    = gpu_mm.PointingPlan
			self.lib.tod2map         = gpu_mm.tod2map
			self.lib.map2tod         = gpu_mm.map2tod
			# Cuts
			self.lib.insert_ranges   = gpu_mm.insert_ranges
			self.lib.extract_ranges  = gpu_mm.extract_ranges
			self.lib.clear_ranges    = gpu_mm.clear_ranges
			# Low-level fft plans
			self.lib.get_plan_size   = gpu_mm.cufft.get_plan_size
			self.lib.get_plan_r2c    = gpu_mm.cufft.get_plan_r2c
			self.lib.get_plan_c2r    = gpu_mm.cufft.get_plan_c2r
			self.lib.set_plan_scratch= gpu_mm.cufft.set_plan_scratch
			# Plan caching, including a reusable scratch array
			self.pools.want("fft_scratch")
			self.plan_cache          = PlanCacheGpu(self.pools.fft_scratch, self.lib)
			# The actual ffts, using this plan cache
			def rfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
				if plan_cache is None: plan_cache = self.plan_cache
				return gpu_mm.cufft.rfft(dat, out=out, axis=axis, plan=plan, plan_cache=plan_cache)
			self.lib.rfft = rfft
			def irfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
				if plan_cache is None: plan_cache = self.plan_cache
				return gpu_mm.cufft.irfft(dat, out=out, axis=axis, plan=plan, plan_cache=plan_cache)
			self.lib.irfft = irfft
			# Tiling
			self.lib.DynamicMap     = gpu_mm.DynamicMap
			self.lib.LocalMap       = gpu_mm.LocalMap
			# BLAS. May need to find a way to make this more compact if we need
			# more of these functions
			self.cublas_handle = cupy.cuda.Device().cublas_handle
			def sgemm(opA, opB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, handle=None):
				assert A.dtype == np.float32, "sgemm needs single precision"
				assert B.dtype == np.float32, "sgemm needs single precision"
				assert C.dtype == np.float32, "sgemm needs single precision"
				arr_alpha, arr_beta = [np.array(x, np.float32) for x in [alpha, beta]]
				if handle is None: handle = self.cublas_handle
				cublas.sgemm(handle, get_cublas_op(opA), get_cublas_op(opB),
					m, n, k, self.ptr(arr_alpha), self.ptr(A), ldA, self.ptr(B), ldB,
					self.ptr(arr_beta), self.ptr(C), ldC)
			self.lib.sgemm = sgemm
			def sdgmm(side, m, n, A, ldA, X, incX, C, ldC, handle=None):
				assert A.dtype == np.float32, "sdgmm needs single precision"
				assert X.dtype == np.float32, "sdgmm needs single precision"
				assert C.dtype == np.float32, "sdgmm needs single precision"
				if handle is None: handle = self.cublas_handle
				cublas.sdgmm(handle, get_cublas_side(side),
					m, n, self.ptr(A), ldA, self.ptr(X), incX, self.ptr(C), ldC)
			self.lib.sdgmm = sdgmm

	def get_cublas_op(op):
		# TODO add more
		return {"n": cublas.CUBLAS_OP_N, "t": cublas.CUBLAS_OP_T}[op.lower()]
	def get_cublas_side(side):
		return {"l": cublas.CUBLAS_SIDE_LEFT, "r": cublas.CUBLAS_SIDE_RIGHT}[side.lower()]

	# Adapted from gpu_mm's version, to use our memory pools
	class PlanCacheGpu:
		def __init__(self, pool, lib):
			self.plans = {}
			self.pool  = pool
			self.lib   = lib
		def get(self, kind, shape, axis=1):
			# Get a plan from the cache, or set it up if not present.
			# Reallocates the scratch space if necessary. We assume that
			# get_plan_size and set_plan_scratch are very fast compared to an fft
			tag = "%s_%s_ax%d" % (str(kind), str(shape), axis)
			if tag in self.plans:
				plan = self.plans[tag]
			else:
				if kind == "r2c": fun = self.lib.get_plan_r2c
				else:             fun = self.lib.get_plan_c2r
				plan = fun(shape[0], shape[1], alloc=False)
			# Make sure scratch is big enough
			size = self.lib.get_plan_size(plan)
			scratch = self.pool.zeros(size, dtype=np.uint8)
			self.lib.set_plan_scratch(plan, scratch)
			# Update cache. We do this even if we had it cached
			# because the scratch buffer may have changed
			self.plans[tag] = plan
			return plan

	Devices.gpu = MMDeviceGpu

except ImportError:
	pass

if False:

	class MMDeviceCpu(device.DeviceCpu):
		def __init__(self, align=None, alloc_factory=None):
			super().__init__(align=align, alloc_factory=alloc_factory)
			# ffts. No plan caching for now
			def rfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
				return fft.rfft(dat, ft=out, axis=axis)
			self.lib.rfft = rfft
			def irfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
				return fft.irfft(dat, tod=out, axis=axis, normalize=False)
			self.lib.irfft = irfft
			# BLAS. May need to find a way to make this more compact if we need
			# more of these functions
			def sgemm(opA, opB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, handle=None):
				assert A.dtype == np.float32, "sgemm needs single precision"
				assert B.dtype == np.float32, "sgemm needs single precision"
				assert C.dtype == np.float32, "sgemm needs single precision"
				# Hack: We ignore m, n, k, ldA, ldB, ldC here and assume that
				# these match the array objects passed! This wouldn't be necessary
				# if scipy made the underlying blas interface directly available.
				# Alternatively one could construct proxy arrays using ldA etc,
				# and pass those in
				scipy.linalg.blas.sgemm(alpha, A.T, B.T, beta=beta, c=C.T, overwrite_c=True,
					trans_a = opA.lower()=="t", trans_b=opB.lower()=="t")
			self.lib.sgemm = sgemm
			def sdgmm(side, m, n, A, ldA, X, incX, C, ldC, handle=None):
				assert A.dtype == np.float32, "sdgmm needs single precision"
				assert X.dtype == np.float32, "sdgmm needs single precision"
				assert C.dtype == np.float32, "sdgmm needs single precision"
				raise NotImplementedError
			self.lib.sdgmm = sdgmm
			raise NotImplementedError

	Devices.cpu = MMDeviceCpu()
