import numpy as np
import cupy
import gpu_mm
import so3g
from pixell import utils, enmap
from . import gutils
from .gmem import scratch
from .logging import L

class PmatMapGpu:
	def __init__(self, shape, wcs, ctime, bore, offs, polang, ncomp=3, dtype=np.float32):
		"""shape, wcs should be for a fullsky geometry, since we assume
		x wrapping and no negative y pixels"""
		self.ctime = ctime
		self.bore  = bore
		self.offs  = offs
		self.polang= polang
		self.dtype = dtype
		self.ncomp = ncomp
		self.pfit  = PointingFit(shape, wcs, ctime, bore, offs, polang, dtype=dtype)
		self.preplan  = gpu_mm.PointingPrePlan(self.pfit.eval(), shape[-2], shape[-1], periodic_xcoord=True)
		self.pointing = None
		self.plan     = None
	def forward(self, gtod, glmap):
		"""Argument is a LocalMap or equivalent"""
		t1 = gutils.cutime()
		pointing = self.pointing if self.pointing is not None else self.pfit.eval()
		plan     = self.plan     if self.plan     is not None else self._make_plan(pointing)
		t2 = gutils.cutime()
		gpu_mm.map2tod(gtod, glmap, pointing, plan)
		t3 = gutils.cutime()
		L.print("Pcore pt %6.4f gpu %6.4f" % (t2-t1,t3-t2), level=3)
		return gtod
	def backward(self, gtod, glmap):
		t1 = gutils.cutime()
		pointing = self.pointing if self.pointing is not None else self.pfit.eval()
		plan     = self.plan     if self.plan     is not None else self._make_plan(pointing)
		t2 = gutils.cutime()
		gpu_mm.tod2map(glmap, gtod, pointing, plan)
		t3 = gutils.cutime()
		L.print("P'core pt %6.4f gpu %6.4f" % (t2-t1,t3-t2), level=3)
		return glmap
	def precalc_setup(self):
		t1 = gutils.cutime()
		self.pointing = self.pfit.eval()
		self.plan     = self._make_plan(self.pointing)
		t2 = gutils.cutime()
		L.print("Pprep %6.4f" % (t2-t1), level=3)
	def precalc_free (self):
		self.pointing = None
		self.plan     = None
	def _make_plan(self, pointing):
		# Here it would have been convenient if BufAllocator could grow.
		# Instead we have to calculate and reserve the size needed
		size = scratch.plan.aligned_size(self.preplan.plan_nbytes, self.preplan.plan_constructor_tmp_nbytes)
		scratch.plan.reserve(size)
		with scratch.plan.as_allocator():
			return gpu_mm.PointingPlan(self.preplan, pointing)

# Cuts

class PmatCutNull:
	def __init__(self, cuts):
		self.cuts  = cuts
		self.ndof  = 1
	def forward(self, tod, junk): pass
	def backward(self, tod, junk): pass
	def clear(self, tod): pass

class PmatCutFullGpu:
	def __init__(self, cuts):
		dets, starts, lens = cuts
		self.dets   = cupy.asarray(dets,   np.int32)
		self.starts = cupy.asarray(starts, np.int32)
		self.lens   = cupy.asarray(lens,   np.int32)
		self.ndof   = np.sum(lens)  # number of values to solve for
		self.nsamp  = self.ndof     # number of samples covered
		self.offs   = cupy.asarray(gutils.cumsum0(lens), np.int32)
	def forward(self, tod, junk):
		gpu_mm.insert_ranges(tod, junk, self.offs, self.dets, self.starts, self.lens)
	def backward(self, tod, junk):
		gpu_mm.extract_ranges(tod, junk, self.offs, self.dets, self.starts, self.lens)
		# Zero-out the samples we used, so the other signals (e.g. the map)
		# don't need to care about them
		self.clear(tod)
	def clear(self, tod):
		gpu_mm.clear_ranges(tod, self.dets, self.starts, self.lens)

# PmatCutPolyGpu does not have orthogonal basis functions due to
# the truncated ranges, so it needs a better preconditioner. Since there
# are only bsize = 400 possible truncation lengths, we can precompute the
# optimal [order=3,order=3] preconditioner for each length, which would
# not take much space at all. How would we apply them effectively though?
# If we could efficiently orthogonalize like this, then we could just
# bake it into the pointing matrix itself, couldn't we? For the map, the
# precon covers much less data than the pointing matrix, but for the cuts
# that's not the case, so I don't think it makes sense to save time
# in the pointing matrix if it makes the preconditioenr harder to deal with.
#
# In any case, we would need to be able to multiply by a 4x4 matric for
# each block.
#
# 1. Indexed matrix multiply: y[ia] = A[q[i]ab]*x[ib]
#    Takes much less space, but requires the support for indirection.
# 2. Explicit matrix multiply: y[ia] = A[iab]*x[ib].
#    This can be done using a simple A.dot(x)
#
# #1 doesn't work. It's not supported directly, and looping over
# slices is very slow.
# #2 works, but can't be done with .dot, must be done with einsum.
# Seems to take at least 0.15 ms.
#
# TODO: Build P'P once, then invert the 400 different truncations
# of it. The ivar scaling will be handled with a separate product.

class PmatCutPolyGpu:
	def __init__(self, cuts, basis=None, order=None, bsize=None):
		dets, starts, lens = cuts
		# Either construct or use an existing basis
		if basis is None:
			if bsize is None: bsize = 400
			if order is None: order = 3
			self.basis = gutils.legbasis(order, bsize)
		else:
			assert order is None and bsize is None, "Specify either basis or order,bsize, not both"
			order = basis.shape[0]-1
			bsize = basis.shape[1]
			self.basis = cupy.asarray(basis, dtype=np.float32)
		# Subdivide ranges that are longer than our block size
		dets, starts, lens = split_ranges(dets, starts, lens, bsize)
		self.dets   = cupy.asarray(dets,   np.int32)
		self.starts = cupy.asarray(starts, np.int32)
		self.lens   = cupy.asarray(lens,   np.int32)
		# total number of samples covered
		self.nsamp  = np.sum(lens)
		# output buffer information. Offsets
		padlens     = (lens+bsize-1)//bsize*bsize
		self.nrange = len(lens)
		self.ndof   = self.nrange*(order+1)
		self.offs   = cupy.asarray(gutils.cumsum0(padlens), np.int32)
	def forward(self, tod, junk):
		# B[nb,bsize], bjunk[nrange,nb], blocks[nrange,bsize] = bjunk.dot(B.T)
		bjunk  = junk.reshape(self.nrange,self.basis.shape[0])
		with scratch.cut.as_allocator():
			blocks = bjunk.dot(self.basis)
			gpu_mm.insert_ranges(tod, blocks, self.offs, self.dets, self.starts, self.lens)
	def backward(self, tod, junk):
		with scratch.cut.as_allocator():
			blocks = cupy.zeros((self.nrange, self.basis.shape[1]), np.float32)
			gpu_mm.extract_ranges(tod, blocks, self.offs, self.dets, self.starts, self.lens)
			self.clear(tod)
			bjunk   = blocks.dot(self.basis.T)
			junk[:] = bjunk.reshape(-1)
	def clear(self, tod):
		gpu_mm.clear_ranges(tod, self.dets, self.starts, self.lens)


# Misc

def calc_pointing(ctime, bore, offs, polang, site="so", weather="typical", dtype=np.float32):
	offs, polang = np.asarray(offs), np.asarray(polang)
	ndet, nsamp = len(offs), bore.shape[1]
	sightline = so3g.proj.coords.CelestialSightLine.az_el(ctime, bore[1], bore[0], site="so", weather="typical")
	fplane    = so3g.proj.coords.FocalPlane.from_xieta(offs[:,1], offs[:,0], np.pi/2-polang)
	pos_equ   = np.moveaxis(sightline.coords(fplane),2,0) # [{ra,dec,c1,s1},ndet,nsamp]
	pos_equ[:2] = pos_equ[1::-1] # [{dec,ra,c1,s1},ndet,nsamp]
	return pos_equ

class PointingFit:
	def __init__(self, shape, wcs, ctime, bore, offs, polang,
			subsamp=200, site="so", weather="typical", dtype=np.float64,
			nt=1, nx=3, ny=3, store_basis=False, positive_x=False):
		"""Jon's polynomial pointing fit. This predicts each detector's celestial
		coordinates based on the array center's celestial coordinates. The model
		fit is
		 pos_det = B a + n
		where
		 B = [1,t**{1},ra**{1,2,3,4},dec**{1,2,3},t*ra,t*dec,ra*dec]
		The ML fit for this is
		 a = (B'B)"B'pos_det
		Actually, going all the way to pixels will be just as cheap as going to
		ra, dec. What's the best way to handle this?
		1. Build everything into this class
		2. Make the interpolator more general, so it takes a function that provides
		pointing as an argument.
		For now I'll stick with the simple #1"""
		self.shape, self.wcs = shape, wcs
		self.nt, self.nx, self.ny = nt, nx, ny
		self.dtype = dtype
		self.store_basis = store_basis
		self.subsamp     = subsamp
		self.nphi = utils.nint(360/np.abs(wcs.wcs.cdelt[0]))
		# 1. Find the typical detector offset
		off0 = np.mean(offs, 0)
		# 2. We want to be able to calculate y,x,psi for any detector offset
		p0 = enmap.pix2sky(shape, wcs, [0,0]) # [{dec,ra}]
		dp = wcs.wcs.cdelt[::-1]*utils.degree # [{dec,ra}]
		def calc_pixs(ctime, bore, offs, polang):
			offs, polang = np.asarray(offs), np.asarray(polang)
			ndet, nsamp  = len(offs), bore.shape[1]
			pixs      = np.empty((3,ndet,nsamp),dtype) # [{y,x,psi},ndet,nsamp]
			pos_equ   = calc_pointing(ctime, bore, offs, polang, site=site, weather=weather, dtype=dtype)
			# Unwind avoids angle wraps, which are bad for interpolation
			pos_equ[0]= utils.unwind(pos_equ[0])
			pixs[:2]  = (pos_equ[:2]-p0[:,None,None])/dp[:,None,None]
			pixs[2]   = utils.unwind(np.arctan2(pos_equ[3],pos_equ[2]))
			return pixs
		# 3. Calculate the full pointing for the reference pixel
		ref_pixs = cupy.array(calc_pixs(ctime, bore, off0[None], [0])[:,0])
		# 4. Calculate a sparse pointing for the individual detectors
		det_pixs = cupy.array(calc_pixs(ctime[::subsamp], bore[:,::subsamp], offs, polang))
		# 4b. Add multiples of the sky wrapping so all x pixels are positive.
		# This is useful if we do map-space wrapping instead of per-sample wrapping
		if positive_x:
			# Assume we move at most 1 pixel per sample when accounting for worst
			# case underestimate of minimum x position. The tiling must be wide enough
			# to accomodate this of course
			offset = (subsamp-cupy.min(det_pixs[1])+self.nphi)//self.nphi*self.nphi
			ref_pixs[1] += offset
			det_pixs[1] += offset
		# 4b. This is where we would add a multiple of nphi to ref_pixs and det_pixs if we wanted
		# all pixel indices to be positive
		# 5. Calculate the basis
		B        = self.basis(ref_pixs)
		# Store either the basis or the reference pointing
		if store_basis: self.B = B
		else:           self.ref_pixs = ref_pixs
		# 6. Calculate and store the interpolation coefficients coefficients
		self.coeffs = self.fit(det_pixs, B[:,::subsamp])
	def basis(self, ref_pixs):
		"""Calculate the interpolation basis"""
		nsamp = ref_pixs.shape[-1]
		B     = cupy.empty((1+self.nt+self.nx+self.ny+3,nsamp),self.dtype)
		mins  = cupy.min(ref_pixs[:2],1)
		maxs  = cupy.max(ref_pixs[:2],1)
		t     = cupy.linspace(-1,1,nsamp,self.dtype)
		y, x  = ref_pixs[:2]
		B[0]  = 1
		# I wish python had a better way to write this
		for i in range(self.nt): B[1+i]                 = t**(i+1)
		for i in range(self.nx): B[1+self.nt+i]         = x**(i+1)
		for i in range(self.ny): B[1+self.nt+self.nx+i] = y**(i+1)
		B[1+self.nt+self.nx+self.ny+0] = t*x
		B[1+self.nt+self.nx+self.ny+1] = t*y
		B[1+self.nt+self.nx+self.ny+2] = x*y
		return B
	def fit(self, det_pixs, B=None):
		"""Fit the interpolation coefficients given det_pixs[{y,x,psi},ndet,nsamp]"""
		if B is None: B = self.basis(self.ref_pixs) # [ndof,nsamp]
		# The fit needs to be done in double precision. The rest is fine in single precision
		B64 = B.astype(np.float64)
		v64 = det_pixs.astype(np.float64)
		idiv= cupy.linalg.inv(B64.dot(B64.T))
		coeffs = v64.dot(B64.T).dot(idiv)
		coeffs = coeffs.astype(self.dtype)
		return coeffs
	def eval(self, coeffs=None, B=None):
		if B is None:
			B = self.B if self.store_basis else self.basis(self.ref_pixs)
		if coeffs is None:
			coeffs = self.coeffs
		pointing = scratch.pointing.empty(coeffs.shape[:-1]+B.shape[1:], B.dtype)
		coeffs.dot(B, out=pointing)
		return pointing
