import numpy as np
import cupy
import gpu_mm
from cupy.cuda import cublas
from pixell import utils, bunch
from . import gutils, gmem
from .gutils import anypy
from .gmem import scratch
from .logging import L

class Nmat:
	def __init__(self):
		"""Initialize the noise model. In subclasses this will typically set up parameters, but not
		build the details that depend on the actual time-ordered data"""
		self.ivar  = np.ones(1, dtype=np.float32)
		self.ready = True
	def build(self, tod, **kwargs):
		"""Measure the noise properties of the given time-ordered data tod[ndet,nsamp], and
		return a noise model object tailored for that specific tod. The returned object
		needs to provide the .apply(tod) method, which multiplies the tod by the inverse noise
		covariance matrix. Usually the returned object will be of the same class as the one
		we call .build(tod) on, just with more of the internal state initialized."""
		return self
	def apply(self, tod):
		"""Multiply the time-ordered data tod[ndet,nsamp] by the inverse noise covariance matrix.
		This is done in-pace, but the result is also returned."""
		return tod.copy()
	def white(self, tod):
		"""Like apply, but without detector or time correlations"""
		return tod.copy()
	def write(self, fname):
		bunch.write(fname, bunch.Bunch(type="Nmat"))
	@staticmethod
	def from_bunch(data): return Nmat()

def apply_vecs_gpu(ftod, iD, V, Kh, bins, tmp, vtmp, divtmp, handle=None, out=None):
	"""Jon's core for the noise matrix. Does not allocate any memory itself. Takes the work
	memory as arguments instead.

	ftod: The fourier-transform of the TOD, cast to float32. [ndet,nfreq]
	iD:   The inverse white noise variance. [nbin,ndet]
	V:    The eigenvectors. [ndet,nmode]
	Kh:   The square root of the Woodbury kernel (E"+V'DV)**-0.5. [nbin,nmode,nmode]
	bins: The frequency ranges for each bin. [nbin,{from,to}]
	tmp, vtmp, divtmp: Work arrays
	"""
	if out    is None: out = dat
	if handle is None: handle = cupy.cuda.Device().cublas_handle
	ndet, nmode = V.shape
	nfreq = ftod.shape[1]
	zero      = np.full(1, 0, iD.dtype)
	one       = np.full(1, 1, iD.dtype)
	minus_one = np.full(1,-1, iD.dtype)
	for bi, (i1,i2) in enumerate(2*bins):
		bsize = i2-i1
		# We want to perform out = iD ftod - (iD V Kh)(iD V Kh)' ftod
		# 1. divtmp = iD V      [ndet,nmode]
		# Cublas is column-major though, so to it we're doing divtmp = V iD [nmode,ndet]. OK
		cublas.sdgmm(handle, cublas.CUBLAS_SIDE_RIGHT, nmode, ndet, V.data.ptr, nmode, iD.data.ptr+bi*iD.strides[0], 1, divtmp.data.ptr, nmode)
		# 2. vtmp   = iD V Kh   [ndet,nmode] -> vtmp = Kh divtmp [nmode,ndet]. OK
		cublas.sgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, nmode, ndet, nmode, one.ctypes.data, Kh.data.ptr+bi*Kh.strides[0], nmode, divtmp.data.ptr, nmode, zero.ctypes.data, vtmp.data.ptr, nmode)
		# 3. tmp    = (iD V Kh)' ftod  [nmode,bsize] -> tmp = ftod vtmp.T [bsize,nmode]. OK
		cublas.sgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T, bsize, nmode, ndet, one.ctypes.data, ftod.data.ptr+i1*ftod.itemsize, nfreq, vtmp.data.ptr, nmode, zero.ctypes.data, tmp.data.ptr, tmp.shape[1])
		# 4. out    = iD ftod  [ndet,bsize] -> out = ftod iD [bsize,ndet]. OK
		cublas.sdgmm(handle, cublas.CUBLAS_SIDE_RIGHT, bsize, ndet, ftod.data.ptr+i1*ftod.itemsize, nfreq, iD.data.ptr+bi*iD.strides[0], 1, out.data.ptr+i1*out.itemsize, nfreq)
		# 5. out    = iD ftod - (iD V Kh)(iD V Kh)' ftod [ndet,bsize] -> out = ftod iD - ftod vtmp.T vtmp [bsize,ndet]. OK
		cublas.sgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, bsize, ndet, nmode, minus_one.ctypes.data, tmp.data.ptr, tmp.shape[1], vtmp.data.ptr, nmode, one.ctypes.data, out.data.ptr+i1*out.itemsize, nfreq)

class NmatDetvecsGpu(Nmat):
	def __init__(self, bin_edges=None, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20],
			downweight=[], window=2, nwin=None, verbose=False, bins=None, iD=None, V=None, Kh=None, ivar=None):
		# Variables used for building the noise model
		if bin_edges is None: bin_edges = np.array([
			0.16, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
			1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 3.80,
			4.60, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 11.0,
			12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
			24.0, 26.0, 28.0, 30.0, 32.0, 36.5, 41.0,
			45.0, 50.0, 55.0, 65.0, 70.0, 80.0, 90.0,
			100., 110., 120., 130., 140., 150., 160., 170.,
			180., 190.
		])
		self.bin_edges = np.array(bin_edges)
		self.mode_bins = np.array(mode_bins)
		self.eig_lim   = np.zeros(len(mode_bins))+eig_lim
		self.single_lim= np.zeros(len(mode_bins))+single_lim
		self.verbose   = verbose
		self.downweight= downweight
		# Variables used for applying the noise model
		self.bins      = bins
		self.window    = window
		self.nwin      = nwin
		self.iD, self.V, self.Kh, self.ivar = iD, V, Kh, ivar
		self.ready      = all([a is not None for a in [iD, V, Kh, ivar]])
		if self.ready:
			self.iD, self.V, self.Kh, self.ivar = [cupy.asarray(a) for a in [iD, V, Kh, ivar]]
			# Why aren't these on the GPU?
			self.dev    = cupy.cuda.Device()
			self.handle = self.dev.cublas_handle
			self.maxbin = np.max(self.bins[:,1]-self.bins[:,0])
	def build(self, tod, srate, extra=False, **kwargs):
		# Apply window before measuring noise model
		dtype = tod.dtype
		nwin  = utils.nint(self.window*srate)
		ndet, nsamp = tod.shape
		nfreq = nsamp//2+1
		tod   = cupy.asarray(tod)
		gutils.apply_window(tod, nwin)
		#ft   = cupy.fft.rfft(tod)
		ft    = gpu_mm.cufft.rfft(tod, plan_cache=gmem.plan_cache)
		# Unapply window again
		gutils.apply_window(tod, nwin, -1)
		del tod
		# First build our set of eigenvectors in two bins. The first goes from
		# 0.25 to 4 Hz the second from 4Hz and up
		mode_bins = makebins(self.mode_bins, srate, nfreq, 1000, rfun=np.round)[1:]
		if np.any(np.diff(mode_bins) < 0):
			raise RuntimeError(f"At least one of the frequency bins has a negative range: \n{mode_bins}")
		# Then use these to get our set of basis vectors
		V = find_modes_jon(ft, mode_bins, eig_lim=self.eig_lim, single_lim=self.single_lim, verbose=self.verbose, dtype=dtype)
		nmode= V.shape[1]
		if V.size == 0: raise errors.ModelError("Could not find any noise modes")
		# Cut bins that extend beyond our max frequency
		bin_edges = self.bin_edges[self.bin_edges < srate/2 * 0.99]
		bins      = makebins(bin_edges, srate, nfreq, nmin=2*nmode, rfun=np.round)
		nbin      = len(bins)
		# Now measure the power of each basis vector in each bin. The residual
		# noise will be modeled as uncorrelated
		E  = cupy.zeros([nbin,nmode],dtype)
		D  = cupy.zeros([nbin,ndet],dtype)
		Nd = cupy.zeros([nbin,ndet],dtype)
		for bi, b in enumerate(bins):
			# Skip the DC mode, since it's it's unmeasurable and filtered away
			b = np.maximum(1,b)
			E[bi], D[bi], Nd[bi] = measure_detvecs(ft[:,b[0]:b[1]], V)
		del Nd, ft
		# Optionally downweight the lowest frequency bins
		if self.downweight != None and len(self.downweight) > 0:
			D[:len(self.downweight)] /= cupy.array(self.downweight)[:,None]
		# Also compute a representative white noise level
		bsize = cupy.array(bins[:,1]-bins[:,0])
		ivar  = cupy.sum(1/D*bsize[:,None],0)/cupy.sum(bsize)
		ivar *= nsamp
		# We need D", not D
		iD, iE = 1/D, 1/E
		# Precompute Kh = (E" + V'D"V)**-0.5
		Kh = cupy.zeros([nbin,nmode,nmode],dtype)
		for bi in range(nbin):
			iK = cupy.diag(iE[bi]) + V.T.dot(iD[bi,:,None] * V)
			Kh[bi] = np.linalg.cholesky(cupy.linalg.inv(iK))
		# Construct a fully initialized noise matrix
		nmat = NmatDetvecsGpu(bin_edges=self.bin_edges, eig_lim=self.eig_lim, single_lim=self.single_lim,
				window=self.window, nwin=nwin, downweight=self.downweight, verbose=self.verbose,
				bins=bins, iD=iD, V=V, Kh=Kh, ivar=ivar)
		return nmat
	def apply(self, gtod, inplace=True):
		t1 =gutils.cutime()
		if not inplace: god = gtod.copy()
		gutils.apply_window(gtod, self.nwin)
		t2 =gutils.cutime()
		#with scratch.ft.as_allocator():
		#	ft = cupy.fft.rfft(gtod, axis=1)
		ft = scratch.ft.empty((gtod.shape[0],gtod.shape[1]//2+1),utils.complex_dtype(gtod.dtype))
		gpu_mm.cufft.rfft(gtod, ft, plan_cache=gmem.plan_cache)
		# If we don't cast to real here, we get the same result but much slower
		rft = ft.view(gtod.dtype)
		t3 =gutils.cutime()
		# Work arrays. Safe to overwrite tod array here, since we'll overwrite it with the ifft afterwards anyway
		ndet, nmode = self.V.shape
		nbin        = len(self.bins)
		with scratch.tod.as_allocator():
			# Tmp must be big enough to hold a full bin's worth of data
			tmp    = cupy.empty([nmode,2*self.maxbin],dtype=rft.dtype)
			vtmp   = cupy.empty([ndet,nmode],         dtype=rft.dtype)
			divtmp = cupy.empty([ndet,nmode],         dtype=rft.dtype)
			apply_vecs_gpu(rft, self.iD, self.V, self.Kh, self.bins, tmp, vtmp, divtmp, handle=self.handle, out=rft)
		cupy.cuda.runtime.deviceSynchronize()
		t4 =gutils.cutime()
		gpu_mm.cufft.irfft(ft, gtod, plan_cache=gmem.plan_cache)
		t5 =gutils.cutime()
		gutils.apply_window(gtod, self.nwin)
		t6 =gutils.cutime()
		L.print("iN sub win %6.4f fft %6.4f mats %6.4f ifft %6.4f win %6.4f ndet %3d nsamp %5d nmode %2d nbin %2d" % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5, gtod.shape[0], gtod.shape[1], self.V.shape[1], len(self.bins)), level=3)
		return gtod
	def white(self, gtod, inplace=True):
		if not inplace: gtod.copy()
		gutils.apply_window(gtod, self.nwin)
		gtod *= self.ivar[:,None]
		gutils.apply_window(gtod, self.nwin)
		return gtod
	def write(self, fname):
		data = bunch.Bunch(type="NmatDetvecsGpu")
		for field in ["bin_edges", "eig_lim", "single_lim", "window", "nwin", "downweight",
				"bins", "D", "V", "E", "ivar"]:
			data[field] = getattr(self, field)
		bunch.write(fname, data)
	@staticmethod
	def from_bunch(data):
		return NmatDetvecsGpu(bin_edges=data.bin_edges, eig_lim=data.eig_lim, single_lim=data.single_lim,
				window=data.window, nwin=data.nwin, downweight=data.downweight,
				bins=data.bins, D=data.D, V=data.V, E=data.E, ivar=data.ivar)

def measure_cov(d, nmax=10000):
	ap    = anypy(d)
	d = d[:,::max(1,d.shape[1]//nmax)]
	n,m   = d.shape
	step  = 10000
	res = ap.zeros((n,n),utils.real_dtype(d.dtype))
	for i in range(0,m,step):
		sub = ap.ascontiguousarray(d[:,i:i+step])
		res += sub.dot(ap.conj(sub.T)).real
	return res/m

def project_out(d, modes): return d-modes.T.dot(modes.dot(d))

def project_out_from_matrix(A, V):
	# Used Woodbury to project out the given vectors from the covmat A
	if V.size == 0: return A
	Q = A.dot(V)
	return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))

def freq2ind(freqs, srate, nfreq, rfun=None):
	"""Returns the index of the first fourier mode with greater than freq
	frequency, for each freq in freqs."""
	if freqs is None: return freqs
	if rfun  is None: rfun = np.ceil
	return rfun(np.asarray(freqs)/(srate/2.0)*nfreq).astype(int)

def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None):
	# Translate from frequency to index
	binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
	# Make sure no bins have two few entries
	if nmin > 0:
		binds2 = [binds[0]]
		for b in binds:
			if b-binds2[-1] >= nmin: binds2.append(b)
		binds = binds2
	# Cap at nfreq and eliminate any resulting empty bins
	binds = np.unique(np.minimum(np.concatenate([[0],binds,[nfreq]]),nfreq))
	# Go from edges to [:,{from,to}]
	bins  = np.array([binds[:-1],binds[1:]]).T
	return bins

def mycontiguous(a):
	# I used this in act for some reason, but not sure why. I vaguely remember ascontiguousarray
	# causing weird failures later in lapack
	b = np.zeros(a.shape, a.dtype)
	b[...] = a[...]
	return b

def find_modes_jon(ft, bins, eig_lim=None, single_lim=0, skip_mean=False, verbose=False, dtype=np.float32):
	ap   = anypy(ft)
	ndet = ft.shape[0]
	vecs = ap.zeros([ndet,0],dtype=dtype)
	if not skip_mean:
		# Force the uniform common mode to be included. This
		# assumes all the detectors have accurately measured gain.
		# Forcing this avoids the possibility that we don't find
		# any modes at all.
		vecs = ap.concatenate([vecs,ap.full([ndet,1],ndet**-0.5,dtype=dtype)],1)
	for bi, b in enumerate(bins):
		d    = ft[:,b[0]:b[1]]
		cov  = measure_cov(d)
		cov  = project_out_from_matrix(cov, vecs)
		e, v = ap.linalg.eigh(cov)
		del cov
		#e, v = e.real, v.real
		#e, v = e[::-1], v[:,::-1]
		accept = ap.full(len(e), True, bool)
		if eig_lim is not None:
			# Compute median, exempting modes we don't have enough data to measure
			nsamp    = b[1]-b[0]+1
			median_e = ap.median(ap.sort(e)[::-1][:nsamp])
			accept  &= e/median_e >= eig_lim[bi]
		if verbose: print("bin %d %5d %5d: %4d modes above eig_lim" % (bi, b[0], b[1], ap.sum(accept)))
		if single_lim is not None and e.size:
			# Reject modes too concentrated into a single mode. Since v is normalized,
			# values close to 1 in a single component must mean that all other components are small
			singleness = ap.max(ap.abs(v),0)
			accept    &= singleness < single_lim[bi]
		if verbose: print("bin %d %5d %5d: %4d modes also above single_lim" % (bi, b[0], b[1], ap.sum(accept)))
		e, v = e[accept], v[:,accept]
		vecs = ap.concatenate([vecs,v],1)
	return vecs

def measure_detvecs(ft, vecs):
	# Measure amps when we have non-orthogonal vecs
	ap   = anypy(ft)
	rhs  = vecs.T.dot(ft)
	div  = vecs.T.dot(vecs)
	amps = ap.linalg.solve(div,rhs)
	E    = ap.mean(ap.abs(amps)**2,1)
	# Project out modes for every frequency individually
	dclean = ft - vecs.dot(amps)
	# The rest is assumed to be uncorrelated
	Nu = ap.mean(ap.abs(dclean)**2,1)
	# The total auto-power
	Nd = ap.mean(ap.abs(ft)**2,1)
	return E, Nu, Nd

def sichol(A):
	iA = np.linalg.inv(A)
	try: return np.linalg.cholesky(iA), 1
	except np.linalg.LinAlgError:
		return np.linalg.cholesky(-iA), -1

def woodbury_invert(D, V, s=1):
	"""Given a compressed representation C = D + sVV', compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	V, D = map(np.asarray, [V,D])
	# Flatten everything so we can be dimensionality-agnostic
	D = D.reshape(-1, D.shape[-1])
	V = V.reshape(-1, V.shape[-2], V.shape[-1])
	I = np.eye(V.shape[2])
	# Allocate our output arrays
	iD = gutils.safe_inv(D)
	iV = V*0
	# Invert each
	for i in range(len(D)):
		core = I*s + (V[i].T*iD[i,None,:]).dot(V[i])
		core, sout = sichol(core)
		iV[i] = iD[i,:,None]*V[i].dot(core)
	sout = -sout
	return iD, iV, sout
