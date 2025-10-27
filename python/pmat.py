import numpy as np
import quaternion
import so3g
from pixell import utils, enmap, coordsys, bench
from . import device, gutils
from .logging import L

class PmatDummy:
	def __init__(self): pass
	def forward(self, gtod, glmap): pass
	def backward(self, gtod, glmap): pass
	def precalc_setup(self, reset_buffer=True): pass
	def precalc_free (self): pass

class PmatMap:
	def __init__(self, shape, wcs, ctime, bore, offs, polang, sys="cel", response=None, ncomp=3, dev=None, dtype=np.float32):
		"""shape, wcs should be for a fullsky geometry, since we assume
		x wrapping and no negative y pixels

		* shape, wcs: geometry
		* ctime[nsamp]
		* bore[{ra,dec},nsamp]
		* offs[ndet,{eta,xi}]
		* polang[ndet]
		* response[{T,P},ndet] or None
		"""
		self.ctime = ctime
		self.bore  = bore
		self.offs  = offs
		self.polang= polang
		self.dtype = dtype
		self.ncomp = ncomp
		self.sys   = sys
		self.dev   = dev or device.get_device()
		self.response = dev.np.array(response) if response is not None else None
		self.pfit  = PointingFit(shape, wcs, ctime, bore, offs, polang, sys=self.sys, dtype=dtype, dev=self.dev)
		self.preplan  = self.dev.lib.PointingPrePlan(self.pfit.eval(), shape[-2], shape[-1], periodic_xcoord=True)
		self.pointing = None
		self.plan     = None
	def forward(self, gtod, glmap):
		"""Argument is a LocalMap or equivalent"""
		t1 = self.dev.time()
		pointing = self.pointing if self.pointing is not None else self.pfit.eval()
		plan     = self.plan     if self.plan     is not None else self._make_plan(pointing)
		t2 = self.dev.time()
		self.dev.lib.map2tod(gtod, glmap, pointing, plan, response=self.response)
		t3 = self.dev.time()
		L.print("Pcore pt %6.4f gpu %6.4f" % (t2-t1,t3-t2), level=3)
		return gtod
	def backward(self, gtod, glmap):
		t1 = self.dev.time()
		pointing = self.pointing if self.pointing is not None else self.pfit.eval()
		plan     = self.plan     if self.plan     is not None else self._make_plan(pointing)
		t2 = self.dev.time()
		self.dev.lib.tod2map(glmap, gtod, pointing, plan, response=self.response)
		t3 = self.dev.time()
		L.print("P'core pt %6.4f gpu %6.4f" % (t2-t1,t3-t2), level=3)
		return glmap
	def precalc_setup(self, reset_buffer=True):
		t1 = self.dev.time()
		self.pointing = self.pfit.eval(reset_buffer=reset_buffer)
		self.plan     = self._make_plan(self.pointing, reset_buffer=reset_buffer)
		t2 = self.dev.time()
		L.print("Pprep %6.4f" % (t2-t1), level=3)
	def precalc_free (self):
		self.pointing = None
		self.plan     = None
	def _make_plan(self, pointing, reset_buffer=True):
		with self.dev.pools["plan"].as_allocator(reset=reset_buffer):
			return self.dev.lib.PointingPlan(self.preplan, pointing)

# Cuts

class PmatCutNull:
	def __init__(self, cuts):
		self.cuts  = cuts
		self.ndof  = 1
	def forward(self, tod, junk): pass
	def backward(self, tod, junk): pass
	def clear(self, tod): pass

class PmatCutFull:
	def __init__(self, cuts, dev=None):
		dets, starts, lens = cuts
		self.dev    = dev or device.get_device()
		self.dets   = self.dev.np.asarray(dets,   np.int32)
		self.starts = self.dev.np.asarray(starts, np.int32)
		self.lens   = self.dev.np.asarray(lens,   np.int32)
		self.ndof   = np.sum(lens)  # number of values to solve for
		self.nsamp  = self.ndof     # number of samples covered
		self.offs   = self.dev.np.asarray(gutils.cumsum0(lens), np.int32)
	def forward(self, tod, junk):
		self.dev.lib.insert_ranges(tod, junk, self.offs, self.dets, self.starts, self.lens)
	def backward(self, tod, junk=None, clear=True):
		if junk is None: junk = self.dev.np.zeros(self.nsamp)
		self.dev.lib.extract_ranges(tod, junk, self.offs, self.dets, self.starts, self.lens)
		# Zero-out the samples we used, so the other signals (e.g. the map)
		# don't need to care about them
		if clear: self.clear(tod)
		return junk
	def clear(self, tod):
		self.dev.lib.clear_ranges(tod, self.dets, self.starts, self.lens)

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

class PmatCutPoly:
	def __init__(self, cuts, dev=None, basis=None, order=None, bsize=None):
		dets, starts, lens = cuts
		self.dev = dev or device.get_device()
		# Either construct or use an existing basis
		if basis is None:
			if bsize is None: bsize = 400
			if order is None: order = 3
			self.basis = gutils.legbasis(order, bsize)
		else:
			assert order is None and bsize is None, "Specify either basis or order,bsize, not both"
			order = basis.shape[0]-1
			bsize = basis.shape[1]
			self.basis = self.dev.np.asarray(basis, dtype=np.float32)
		# Subdivide ranges that are longer than our block size
		dets, starts, lens = gutils.split_ranges(dets, starts, lens, bsize)
		self.dets   = self.dev.np.asarray(dets,   np.int32)
		self.starts = self.dev.np.asarray(starts, np.int32)
		self.lens   = self.dev.np.asarray(lens,   np.int32)
		# total number of samples covered
		self.nsamp  = np.sum(lens)
		# output buffer information. Offsets
		padlens     = (lens+bsize-1)//bsize*bsize
		self.nrange = len(lens)
		self.ndof   = self.nrange*(order+1)
		self.offs   = self.dev.np.asarray(gutils.cumsum0(padlens), np.int32)
	def forward(self, tod, junk):
		# B[nb,bsize], bjunk[nrange,nb], blocks[nrange,bsize] = bjunk.dot(B.T)
		bjunk  = junk.reshape(self.nrange,self.basis.shape[0])
		with self.dev.pools["cut"].as_allocator():
			blocks = bjunk.dot(self.basis)
			self.dev.lib.insert_ranges(tod, blocks, self.offs, self.dets, self.starts, self.lens)
	def backward(self, tod, junk):
		with self.dev.pools["cut"].as_allocator():
			blocks = self.dev.np.zeros((self.nrange, self.basis.shape[1]), np.float32)
			self.dev.lib.extract_ranges(tod, blocks, self.offs, self.dets, self.starts, self.lens)
			# Zero-out the samples we used, so the other signals (e.g. the map)
			# don't need to care about them
			self.clear(tod)
			bjunk   = blocks.dot(self.basis.T)
			junk[:] = bjunk.reshape(-1)
	def clear(self, tod):
		self.dev.lib.clear_ranges(tod, self.dets, self.starts, self.lens)

# Basic model is tod = 1/sin(el) * amp. However, getting el[t] for each
# detector requires a rotation operation, which is wasteful. We know our
# elevation amplitude will be tiny, so easier to just do an expansion:
#  tod[d,t] = Δbel[t]*amp[d] + Δbel**2*amp2[d] + ... + n[d,t]
# Probably no point in keeping more than the 1st order, but easy to keep
# it general. Would then solve for the amps per detector. There's no
# zeroth-order term because 1. it would be infinitely noisy, and 2.
# it's degenerate with the map.
class PmatElMod:
	def __init__(self, el, dev=None, order=None, B=None, dtype=np.float32):
		self.dev   = dev or device.get_device()
		assert dtype == np.float32, "Only float32 supported"
		self.dtype = dtype
		self.nsamp = len(el)
		# Either construct or use an existing basis
		if B is None:
			if order is None: order = 1
			B = self.dev.np.zeros((order,self.nsamp), self.dtype)
			# B[0] = Δel
			el = np.ascontiguousarray(el, dtype=dtype)
			self.dev.copy(el, B[0])
			B[0] -= 0.5*(dev.np.max(B[0])+dev.np.min(B[0]))
			# Fill in any higher orders
			for i in range(1, order):
				B[i] = B[0]**(i+1)
			self.B = B
		else:
			assert order is None and bsize is None, "Specify either basis or order, not both"
			self.B = self.dev.np.asarray(B, dtype=self.dtype)
		self.order = self.B.shape[0]
	def forward(self, tod, amps):
		# tod[d,t] += amps[d,a]*B[a,t]. Remember that sgemm is in Fortran order, so
		# to it we're doing tod[t,d] += B[t,a]*amps[a,d]
		ndet, nsamp = tod.shape
		assert amps.shape == (ndet, self.order)
		assert amps.flags["C_CONTIGUOUS"]
		assert tod.dtype == self.dtype
		assert amps.dtype == self.dtype
		self.dev.lib.sgemm("N", "N", nsamp, ndet, self.order, 1, self.B, nsamp, amps, self.order, 1, tod, nsamp)
	def backward(self, tod, amps):
		# amps[d,a] += B[a,t]*tod[d,t] => fortran amps[a,d] += B'[a,t]*tod[t,d]
		ndet, nsamp = tod.shape
		assert amps.shape == (ndet, self.order)
		assert amps.flags["C_CONTIGUOUS"]
		assert tod.dtype == self.dtype
		assert amps.dtype == self.dtype
		self.dev.lib.sgemm("T", "N", self.order, ndet, nsamp, 1, self.B, nsamp, tod, nsamp, 1, amps, self.order)

# Misc

def calc_pointing(ctime, bore, offs, polang, sys="cel", site="so", weather="typical",
		dtype=np.float32, use_so3g="auto"):
	offs, polang = np.asarray(offs), np.asarray(polang)
	ndet, nsamp = len(offs), bore.shape[1]
	# Transform the boresight
	icoord = coordsys.Coords(az=bore[1], el=bore[0], roll=bore[2])
	ocoord = coordsys.transform("hor", sys, icoord, ctime=ctime, site=site, weather=weather)
	# Apply the detector offsets. Like so3g, we assume that they're all affected
	# by the same refraction, which is only an approximation
	if use_so3g == "auto":
		use_so3g = utils.can_import("so3g")
	if use_so3g:
		import so3g
		fplane  = so3g.proj.coords.FocalPlane.from_xieta(offs[:,1], offs[:,0], polang)
		p       = so3g.ProjEng_CAR_TQU_NonTiled((1, 1, 1., 1., 1., 1.))
		q       = so3g.proj.quat.G3VectorQuat(quaternion.as_float_array(ocoord.q))
		pos_equ     = np.moveaxis(p.coords(q, fplane.quats, None),2,0)
		pos_equ[:2] = pos_equ[1::-1] # [{dec,ra,c1,s1},ndet,nsamp]
		# Go from c1,s1 to psi
		pos_equ[2]  = np.arctan2(pos_equ[3],pos_equ[2])
		pos_equ     = pos_equ[:3]
	else:
		# This part is nicer, but makes the function take 440 ms
		# vs. 210 ms with so3g. The main slow part is the decomposition,
		# which seems like it needs to be moved to a lower-level language
		qdet   = coordsys.rotation_xieta(offs[:,1], offs[:,0], polang)
		ocoord = ocoord*qdet[:,None] # {ndet,nsamp}
		# Decompose into ra,dec. This step is surprisingly slow
		pos_equ= np.array([ocoord.dec, ocoord.ra, ocoord.psi]) # [{dec,ra,psi}]
	return pos_equ

class PointingFit:
	def __init__(self, shape, wcs, ctime, bore, offs, polang,
			subsamp=200, sys="cel", site="so", weather="typical", dtype=np.float64,
			nt=1, nx=3, ny=3, store_basis=False, positive_x=False, dev=None):
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
		For now I'll stick with the simple #1

		* shape, wcs: geometry
		* ctime[nsamp]
		* bore[{ra,dec},nsamp]
		* offs[ndet,{eta,xi}]
		* polang[ndet]
		"""
		# TODO: Consider generalizing to more than just car
		assert wcs.wcs.ctype[0][-3:] == "CAR", "Only CAR supported for now"
		self.shape, self.wcs = shape, wcs
		self.nt, self.nx, self.ny = nt, nx, ny
		self.dev   = dev or device.get_device()
		self.dtype = dtype
		self.store_basis = store_basis
		self.subsamp     = subsamp
		self.nphi = utils.nint(360/np.abs(wcs.wcs.cdelt[0]))
		self.sys  = sys
		self.site = site
		self.weather = weather
		# 1. Find the typical detector offset
		off0 = np.mean(offs, 0)
		# 2. We want to be able to calculate y,x,psi for any detector offset
		# We calculate p0 mangually avoid annoying %360 that wcslib seems to do
		self.p0 = (wcs.wcs.crval+(1-wcs.wcs.crpix)*wcs.wcs.cdelt)[::-1]*utils.degree # [{dec,ra}]
		#self.p0 = enmap.pix2sky(shape, wcs, [0,0]) # [{dec,ra}]
		self.dp = wcs.wcs.cdelt[::-1]*utils.degree # [{dec,ra}]
		# 3. Calculate the full pointing for the reference pixel
		ref_pixs = self.dev.np.array(self.eval_exact(ctime, bore, off0[None], [0])[:,0])
		# 4. Calculate a sparse pointing for the individual detectors
		# [{y,x,psi},ndet,nsamp]
		det_pixs = self.dev.np.array(self.eval_exact(ctime[::subsamp], bore[:,::subsamp], offs, polang))
		# 4b. Add multiples of the sky wrapping so all x pixels are positive.
		# This is useful if we do map-space wrapping instead of per-sample wrapping
		if positive_x:
			# Assume we move at most 1 pixel per sample when accounting for worst
			# case underestimate of minimum x position. The tiling must be wide enough
			# to accomodate this of course
			offset = (subsamp-self.dev.np.min(det_pixs[1])+self.nphi)//self.nphi*self.nphi
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
		## Test the pointing
		#det_pixs_model = self.eval(B=B[:,::subsamp])
		#print(np.std(det_pixs_model-det_pixs))
	def basis(self, ref_pixs):
		"""Calculate the interpolation basis"""
		# FIXME: Scanning pattern type 3 with elevation nods gets 0.1 pixel RMS accuracy
		# here
		nsamp = ref_pixs.shape[-1]
		B     = self.dev.np.empty((1+self.nt+self.nx+self.ny+3,nsamp),self.dtype)
		mins  = self.dev.np.min(ref_pixs[:2],1)
		maxs  = self.dev.np.max(ref_pixs[:2],1)
		t     = self.dev.np.linspace(-1,1,nsamp,self.dtype)
		# Must normalze these to avoid numerical problems. Pixel numbers can be
		# quite high compared to the t values
		y     = normalize(ref_pixs[0])
		x     = normalize(ref_pixs[1])
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
		idiv= self.dev.np.linalg.inv(B64.dot(B64.T))
		coeffs = v64.dot(B64.T).dot(idiv)
		coeffs = coeffs.astype(self.dtype)
		return coeffs # [{y,x,psi},ndet,ndof]
	def eval(self, coeffs=None, B=None, reset_buffer=True):
		t1 = self.dev.time()
		if B is None:
			B = self.B if self.store_basis else self.basis(self.ref_pixs)
		if coeffs is None:
			coeffs = self.coeffs
		t2 = self.dev.time()
		pointing = self.dev.pools["pointing"].empty(coeffs.shape[:-1]+B.shape[1:], B.dtype, reset=reset_buffer)
		t3 = self.dev.time()
		#coeffs.dot(B, out=pointing)
		# coeffs[{y,x,psi},ndet,ndof]*B[ndof,nsamp] => pointing[{y,x,psi},ndet,nsamp]
		# Cublas is column-major though, so we're doing pointing = B*coeffs
		npre = coeffs.shape[0]*coeffs.shape[1]
		ndof, nsamp = B.shape
		self.dev.lib.sgemm("N", "N", nsamp, npre, ndof, 1, B, nsamp, coeffs.reshape(npre,ndof), ndof, 0, pointing.reshape(npre,nsamp), nsamp)
		t4 = self.dev.time()
		L.print("eval pointing %8.4f %8.4f %8.4f" % (t2-t1,t3-t2,t4-t3), level=3)
		return pointing # [{y,x,psi},ndet,nsamp], on device
	def eval_exact(self, ctime, bore, offs, polang, sys=None, site=None, weather=None):
		if sys     is None: sys     = self.sys
		if site    is None: site    = self.site
		if weather is None: weather = self.weather
		offs, polang = np.asarray(offs), np.asarray(polang)
		ndet, nsamp  = len(offs), bore.shape[1]
		pixs      = np.empty((3,ndet,nsamp),self.dtype) # [{y,x,psi},ndet,nsamp]
		pos_equ   = calc_pointing(ctime, bore, offs, polang, sys=sys, site=site, weather=weather, dtype=self.dtype)
		# Unwind avoids angle wraps, which are bad for interpolation
		pos_equ[1]= utils.unwind(pos_equ[1])
		pixs[:2]  = (pos_equ[:2]-self.p0[:,None,None])/self.dp[:,None,None]
		pixs[2]   = utils.unwind(pos_equ[2])
		return pixs # [{y,x,psi},ndet,nsamp], on cpu

def normalize(x):
	x1 = np.min(x)
	x2 = np.max(x)
	return (x-x1)/(x2-x1) if x2!=x1 else x
