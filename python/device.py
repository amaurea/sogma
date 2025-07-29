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

class MMDeviceMinimal(device.DeviceCpu):
	def __init__(self, align=None, alloc_factory=None):
		super().__init__(align=align, alloc_factory=alloc_factory)
		self.name = "minimal"
		# ffts. No plan caching for now
		def rfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
			return fft.rfft(dat, ft=out, axes=axis)
		self.lib.rfft = rfft
		def irfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
			return fft.irfft(dat, tod=out, axes=axis, normalize=False)
		self.lib.irfft = irfft
		self.lib.fft_factors = [2,4,5,7]
		# BLAS. May need to find a way to make this more compact if we need
		# more of these functions
		def sgemm(opA, opB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, handle=None):
			import scipy
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

Devices.minimal = MMDeviceMinimal

try:
	import cupy, gpu_mm
	from cupy.cuda import cublas

	class MMDeviceGpu(device.DeviceGpu):
		def __init__(self, align=None, alloc_factory=None):
			super().__init__(align=align, alloc_factory=alloc_factory)
			self.name = "gpu"
			# pointing
			self.lib.PointingPrePlan = gpu_mm.PointingPrePlan
			self.lib.PointingPlan    = gpu_mm.PointingPlan
			self.lib.tod2map         = gpu_mm.tod2map
			self.lib.map2tod         = gpu_mm.map2tod
			# Cuts
			self.lib.insert_ranges   = gpu_mm.insert_ranges
			self.lib.extract_ranges  = gpu_mm.extract_ranges
			self.lib.clear_ranges    = gpu_mm.clear_ranges
			# Deglitching
			self.lib.get_border_means= gpu_mm.get_border_means
			self.lib.deglitch        = gpu_mm.deglitch
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
			self.lib.fft_factors = [2,4,5,7]
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

try:
	import cpu_mm

	class MMDeviceCpu(device.DeviceCpu):
		def __init__(self, align=None, alloc_factory=None):
			super().__init__(align=align, alloc_factory=alloc_factory)
			self.name = "cpu"
			# pointing
			self.lib.PointingPrePlan = cpu_mm.PointingPrePlan
			self.lib.PointingPlan    = cpu_mm.PointingPlan
			self.lib.tod2map         = cpu_mm.tod2map
			self.lib.map2tod         = cpu_mm.map2tod
			# Cuts
			self.lib.insert_ranges   = cpu_mm.insert_ranges
			self.lib.extract_ranges  = cpu_mm.extract_ranges
			self.lib.clear_ranges    = cpu_mm.clear_ranges
			# ffts. No plan caching for now
			def rfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
				return fft.rfft(dat, ft=out, axes=axis)
			self.lib.rfft = rfft
			def irfft(dat, out=None, axis=-1, plan=None, plan_cache=None):
				return fft.irfft(dat, tod=out, axes=axis, normalize=False)
			self.lib.irfft = irfft
			self.lib.fft_factors = [2,4,5,7]
			# Tiling
			self.lib.DynamicMap     = cpu_mm.DynamicMap
			self.lib.LocalMap       = cpu_mm.LocalMap
			# BLAS. May need to find a way to make this more compact if we need
			# more of these functions
			def sgemm(opA, opB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, handle=None):
				import scipy
				assert A.dtype == np.float32, "sgemm needs single precision"
				assert B.dtype == np.float32, "sgemm needs single precision"
				assert C.dtype == np.float32, "sgemm needs single precision"
				cpu_mm.sgemm(opA, opB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC)
			self.lib.sgemm = sgemm
			def sdgmm(side, m, n, A, ldA, X, incX, C, ldC, handle=None):
				assert A.dtype == np.float32, "sdgmm needs single precision"
				assert X.dtype == np.float32, "sdgmm needs single precision"
				assert C.dtype == np.float32, "sdgmm needs single precision"
				# side=='r': C = A.dot(diag(X))
				# side=='l': C = diag(X).dot(A)
				# But cublas is column-major, so it thinks our matrices are transposed.
				# In row-major terms, we're instead doing
				# side=='r': C' = A'.dot(diag(X)) => C = diag(X).dot(A)
				# side=='l': C' = diag(X).dot(A') => C = A.dot(diag(X))
				# Sadly, we don't have a blas version of this it seems, which
				# means we don't get any thread speedup if we don't implement it
				# ourself.
				if   side.lower() == 'r': C[:] = X[:,None]*A
				elif side.lower() == 'l': C[:] = A*X[None,:]
				else: raise ValueError("Unrecognized side '%s'" % str(side))
			self.lib.sdgmm = sdgmm

	Devices.cpu = MMDeviceCpu

except ImportError:
	pass

	# What do we need to make a sotodlib device?
	# 1. LocalMap
	#    Must contain .arr, a device array reshapable to (ntile,ncomp,tyshape,txshape)
	#    Must contain .pixelization, a LocalPixelization
	#    Must be constructable from (pixelization, arr)
	#    Must be passable to map2tod, tod2map
	#    (tyshape,txshape) must be (64,64)
	# 2. LocalPixelization
	#    Must contain .cell_offsets_cpu. These are offsets from the start of .arr
	#    of each tile for a (3,64,64) tile.
	# 3. DynamicMap
	#    Must be passable to tod2map
	#    Must be constructable from (shape, dtype)
	#    Must contain .finalize(), which returns a LocalMap
	# 4. map2tod
	#    Must accept (LocalMap, tod, pointing, plan, [response])
	#    pointing is [{y,x,psi},ndet,nsamp]
	#    plan is PointingPlan
	#    If implemented with sotodlib, we have a mismatch. sotodlib wants
	#     (ncomp,tyshape,txshape,ntile), but we have (ntile,ncomp,tyshape,txshape).
	#     Will moveaxis be enough, or must we make it contiguous too?
	#    sotodlib expects to compute the pointing on the fly. Is there an
	#     interface for passing in precomputed pointing? Yes, but it doesn't
	#     support bilinear interpolation, which we use in sogma. It also doesn't
	#     support response. Must either generalize it, or add a new CoordSys
	#     that represents precomputed pointing. The latter is probably simplest.
	#    PointingFit calculates the pointing using .dot(), which will hopefully
	#     use threads, otherwise it will be very slow.
	#    How does sotodlib handle sky wrapping?
	# 5. tod2map
	#    Must accept (tod, LocalMap/DyamicMap, pointing, plan, [response])
	#    sotodlib needs thread_intervals, which must be precomputed using
	#    _get_proj_threads. Store a whole P in DynamicMap/LocalMap/PointingPrePlan/PointingPlan?
	# 6. PointingPlan
	#    Must be constructable from (preplan, pointing)
	# 7. PointingPrePlan
	#    Must be constructable from (pointing, ny, nx, periodic_xcoord=True)
	#    For sotodlib this and PointingPlan can be dummies



