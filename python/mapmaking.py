import numpy as np
import cupy
from pixell import utils, bunch, enmap
from . import nmat, pmat
from .gmem import scratch
from .logging import L

class MLMapmaker:
	def __init__(self, signals=[], noise_model=None, dtype=np.float32, verbose=False, mode="gpu"):
		"""Initialize a Maximum Likelihood Mapmaker.
		Arguments:
		* signals: List of Signal-objects representing the models that will be solved
		  jointly for. Typically this would be the sky map and the cut samples. NB!
		  The way the cuts currently work, they *MUST* be the first signal specified.
		  If not, the equation system will be inconsistent and won't converge.
		* noise_model: A noise model constructor which will be used to initialize the
		  noise model for each observation. Can be overriden in add_obs.
		* dtype: The data type to use for the time-ordered data. Only tested with float32
		* verbose: Whether to print progress messages. Not implemented"""
		self.signals  = signals
		self.dtype    = dtype
		self.verbose  = verbose
		self.noise_model = noise_model or nmat.NmatUncorr()
		self.data     = []
		self.dof      = MultiZipper()
		self.ready    = False
		self.mode     = mode
	def add_obs(self, id, obs, deslope=True, noise_model=None):
		# Prepare our tod
		t1 = time.time()
		ap     = cupy if self.mode == "gpu" else np
		ctime  = obs.ctime
		srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
		tod    = obs.tod.astype(self.dtype, copy=False)
		t2 = time.time()
		if deslope:
			utils.deslope(tod, w=5, inplace=True)
		t3 = time.time()
		gtod = scratch.tod.copy(tod)
		del tod
		# Allow the user to override the noise model on a per-obs level
		if noise_model is None: noise_model = self.noise_model
		# Build the noise model from the obs unless a fully
		# initialized noise model was passed
		if noise_model.ready:
			nmat = noise_model
		else:
			try:
				nmat = noise_model.build(gtod, srate=srate)
			except Exception as e:
				msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
				raise RuntimeError(msg)
		t4 = time.time()
		# And apply it to the tod
		gtod = nmat.apply(gtod)
		t5 = time.time()
		# Add the observation to each of our signals
		for signal in self.signals:
			signal.add_obs(id, obs, nmat, gtod)
		t6 = time.time()
		L.print("Init sys trun %6.3f ds %6.3f Nb %6.3f N %6.3f add sigs %6.3f %s" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, id), level=2)
		# Save only what we need about this observation
		self.data.append(bunch.Bunch(id=id, ndet=len(obs.dets), nsamp=len(ctime),
			dets=obs.dets, nmat=nmat))
	def prepare(self):
		if self.ready: return
		t1 = time.time()
		for signal in self.signals:
			signal.prepare()
			self.dof.add(signal.dof)
		t2 = time.time()
		L.print("Prep sys %6.3f" % (t2-t1), level=2)
		self.ready = True
	def A(self, x):
		t1 = time.time()
		iwork = [signal.to_work(m) for signal,m in zip(self.signals,self.dof.unzip(x))]
		owork = [w*0 for w in iwork]
		ap    = anypy(iwork[0])
		t2 = time.time()
		for di, data in enumerate(self.data):
			# This is the main place that needs to change for the GPU implementation
			ta1 = cutime()
			#gtod= ap.zeros([data.ndet, data.nsamp], self.dtype)
			gtod = scratch.tod.view([data.ndet, data.nsamp], self.dtype)
			ta2 = cutime()
			for si, signal in reversed(list(enumerate(self.signals))):
				signal.precalc_setup(data.id)
				signal.forward(data.id, gtod, iwork[si])
			ta3 = cutime()
			data.nmat.apply(gtod)
			ta4 = cutime()
			for si, signal in enumerate(self.signals):
				signal.backward(data.id, gtod, owork[si])
				signal.precalc_free(data.id)
			ta5 = cutime()
			#gpu_garbage_collect()
			ta6 = cutime()
			L.print("A z %6.3f P %6.3f N %6.3f P' %6.3f gc %6.4f %s" % (ta2-ta1, ta3-ta2, ta4-ta3, ta5-ta4, ta6-ta5, data.id), level=2)
		t3 = cutime()
		result = self.dof.zip(*[signal.from_work(w) for signal,w in zip(self.signals,owork)])
		t4 = cutime()
		L.print("A prep %6.3f PNP %6.3f finish %6.3f" % (t2-t1, t3-t2, t4-t3), level=2)
		return result
	def M(self, x):
		t1 = cutime()
		iwork = self.dof.unzip(x)
		result = self.dof.zip(*[signal.precon(w) for signal, w in zip(self.signals, iwork)])
		t2 = cutime()
		L.print("M %6.3f" % (t2-t1), level=2)
		return result
	def solve(self, maxiter=500, maxerr=1e-6):
		self.prepare()
		rhs    = self.dof.zip(*[signal.rhs for signal in self.signals])
		solver = utils.CG(self.A, rhs, M=self.M, dot=self.dof.dot)
		while solver.i < maxiter and solver.err > maxerr:
			t1 = time.time()
			solver.step()
			x  = self.dof.unzip(solver.x)
			t2 = time.time()
			yield bunch.Bunch(i=solver.i, err=solver.err, x=x, t=t2-t1)

class Signal:
	"""This class represents a thing we want to solve for, e.g. the sky, ground, cut samples, etc."""
	def __init__(self, name, ofmt, output, ext):
		"""Initialize a Signal. It probably doesn't make sense to construct a generic signal
		directly, though. Use one of the subclasses.
		Arguments:
		* name: The name of this signal, e.g. "sky", "cut", etc.
		* ofmt: The format used when constructing output file prefix
		* output: Whether this signal should be part of the output or not.
		* ext: The extension used for the files.
		"""
		self.name   = name
		self.ofmt   = ofmt
		self.output = output
		self.ext    = ext
		self.dof    = None
		self.ready  = False
	def add_obs(self, id, obs, nmat, Nd): pass
	def prepare(self): self.ready = True
	def forward (self, id, tod, x): pass
	def backward(self, id, tod, x): pass
	def precalc_setup(self, id): pass
	def precalc_free (self, id): pass
	def precon(self, x): return x
	def to_work  (self, x): return x.copy()
	def from_work(self, x): return x
	def write   (self, prefix, tag, x): pass

class SignalMapGpu(Signal):
	"""Signal describing a non-distributed sky map."""
	def __init__(self, shape, wcs, comm, name="sky", ofmt="{name}", output=True,
			ext="fits", dtype=np.float32, sys=None, interpol=None, scratch=None):
		"""Signal describing a sky map in the coordinate system given by "sys", which defaults
		to equatorial coordinates. If tiled==True, then this will be a distributed map with
		the given tile_shape, otherwise it will be a plain enmap. interpol controls the
		pointing matrix interpolation mode. See so3g's Projectionist docstring for details."""
		Signal.__init__(self, name, ofmt, output, ext)
		self.comm  = comm
		self.sys   = sys
		self.dtype = dtype
		self.interpol = interpol
		self.data  = {}
		self.comps = "TQU"
		self.ncomp = 3
		self.ishape= tuple(shape[-2:])
		shape      = tuple(round_up(np.array(shape[-2:]), 64))
		self.rhs = enmap.zeros((self.ncomp,)+shape, wcs, dtype=dtype)
		self.div = enmap.zeros(              shape, wcs, dtype=dtype)
		self.hits= enmap.zeros(              shape, wcs, dtype=dtype)
	def add_obs(self, id, obs, nmat, Nd, pmap=None):
		"""Add and process an observation, building the pointing matrix
		and our part of the RHS. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data.
		"""
		#Nd     = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		ctime  = obs.ctime
		t1     = time.time()
		# could pass this in, but fast to construct. Should ideally match what's used in
		# signal_cut to be safest, but only uses .clear which should be the same for all
		# of them anyway
		pcut   = pmat.PmatCutFullGpu(obs.cuts)
		#pcut   = PmatCutNull(obs.cuts)
		if pmap is None:
			pmap = pmat.PmatMapGpu(self.rhs.shape, self.rhs.wcs, obs.ctime, obs.boresight, obs.point_offset, obs.polangle, dtype=Nd.dtype)
			gpu_garbage_collect()
		# Build the RHS for this observation
		t2 = time.time()
		pcut.clear(Nd)
		obs_rhs = pmap.backward(Nd)
		gpu_garbage_collect()
		t3 = time.time()
		# Build the per-pixel inverse variance for this observation.
		# This will be scalar to make the preconditioner fast, but uses
		# ncomp while building since pmat expects that
		ones         = cupy.zeros_like(obs_rhs)
		ones[0]      = 1
		Nd[:]        = 0
		pmap.forward(Nd, ones)
		pcut.clear(Nd)
		Nd = nmat.white(Nd)
		obs_div = pmap.backward(Nd)[0]
		gpu_garbage_collect()
		t4 = time.time()
		# Build hitcount
		Nd[:]        = 0
		pmap.forward(Nd, ones)
		pcut.clear(Nd)
		obs_hits = pmap.backward(Nd)[0]
		gpu_garbage_collect()
		t5 = time.time()
		del Nd, ones
		# Update our full rhs and div. This works for both plain and distributed maps
		obs_rhs  = enmap.ndmap(obs_rhs .get(), self.rhs.wcs)
		obs_div  = enmap.ndmap(obs_div .get(), self.rhs.wcs)
		obs_hits = enmap.ndmap(obs_hits.get(), self.rhs.wcs)
		self.rhs = self.rhs .insert(obs_rhs , op=np.ndarray.__iadd__)
		self.div = self.div .insert(obs_div , op=np.ndarray.__iadd__)
		self.hits= self.hits.insert(obs_hits, op=np.ndarray.__iadd__)
		gpu_garbage_collect()
		t6 = time.time()
		L.print("Init map pmat %6.3f rhs %6.3f div %6.3f hit %6.3f add %6.3f %s" % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,id), level=2)
		# Save the per-obs things we need. Just the pointing matrix in our case.
		# Nmat and other non-Signal-specific things are handled in the mapmaker itself.
		self.data[id] = bunch.Bunch(pmap=pmap, obs_geo=obs_rhs.geometry)
	def prepare(self):
		"""Called when we're done adding everything. Sets up the map distribution,
		degrees of freedom and preconditioner."""
		if self.ready: return
		t1 = time.time()
		if self.comm is not None:
			self.rhs  = utils.allreduce(self.rhs, self.comm)
			self.div  = utils.allreduce(self.div, self.comm)
			self.hits = utils.allreduce(self.hits,self.comm)
		self.dof   = MapZipper(*self.rhs.geometry, dtype=self.dtype)
		#self.idiv  = safe_invert_ivar(self.div)
		self.idiv  = safe_inv(self.div)
		t2 = time.time()
		L.print("Prep map %6.3f" % (t2-t1), level=2)
		self.ready = True
	def forward(self, id, gtod, gmap, tmul=1):
		"""map2tod operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into tod."""
		if id not in self.data: return # Should this really skip silently like this?
		if tmul != 1: gtod *= tmul
		self.data[id].pmap.forward(gtod, gmap)
	def backward(self, id, gtod, gmap, mmul=1):
		"""tod2map operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into map"""
		if id not in self.data: return
		if mmul != 1: gmap *= mmul
		self.data[id].pmap.backward(gtod, gmap)
	def precalc_setup(self, id): self.data[id].pmap.precalc_setup()
	def precalc_free (self, id): self.data[id].pmap.precalc_free()
	def precon(self, map):
		return self.idiv * map
	def to_work(self, map):
		#return cupy.array(map)
		return scratch.map.copy(map)
	def from_work(self, gmap):
		map = enmap.enmap(gmap.get(), self.rhs.wcs, self.rhs.dtype, copy=False)
		if self.comm is None: return map
		else: return utils.allreduce(map, self.comm)
	def write(self, prefix, tag, m):
		if not self.output: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		if self.comm is None or self.comm.rank == 0:
			enmap.write_map(oname, m[...,:self.ishape[-2],:self.ishape[-1]])
		return oname

class SignalCutFullGpu(Signal):
	# Placeholder for when we have a gpu implementation
	def __init__(self, comm, name="cut", ofmt="{name}_{rank:02}", dtype=np.float32,
			output=False):
		"""Signal for handling the ML solution for the values of the cut samples."""
		Signal.__init__(self, name, ofmt, output, ext="hdf")
		self.comm  = comm
		self.data  = {}
		self.dtype = dtype
		self.off   = 0
		self.rhs   = []
		self.idiv  = []
	def add_obs(self, id, obs, nmat, Nd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data."""
		Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutFullGpu(obs.cuts)
		# Build our RHS
		obs_rhs = cupy.zeros(pcut.ndof, self.dtype)
		pcut.backward(Nd, obs_rhs)
		obs_rhs = obs_rhs.get()
		# Build our preconditioner.
		obs_div = cupy.ones(pcut.ndof, self.dtype)
		Nd[:]   = 0
		pcut.forward(Nd, obs_div)
		nmat.white(Nd)
		pcut.backward(Nd, obs_div)
		obs_idiv = 1/obs_div.get() # back to the cpu
		self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off+pcut.ndof)
		self.off += pcut.ndof
		self.rhs.append(obs_rhs)
		self.idiv.append(obs_idiv)
		gpu_garbage_collect()
	def prepare(self):
		"""Process the added observations, determining our degrees of freedom etc.
		Should be done before calling forward and backward."""
		if self.ready: return
		self.rhs = np.concatenate(self.rhs)
		self.idiv= np.concatenate(self.idiv)
		self.dof = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
		self.ready = True
	def forward(self, id, gtod, gjunk):
		if id not in self.data: return
		d = self.data[id]
		d.pcut.forward(gtod, gjunk[d.i1:d.i2])
	def precon(self, junk):
		print("A", np.std(junk))
		print("B", np.std(junk*self.idiv))
		return junk*self.idiv
	def backward(self, id, gtod, gjunk):
		if id not in self.data: return
		d = self.data[id]
		d.pcut.backward(gtod, gjunk[d.i1:d.i2])
	def to_work  (self, x): return cupy.array(x)
	def from_work(self, x): return x.get()
	def write(self, prefix, tag, m):
		if not self.output: return
		if self.comm is None:
			rank = 0
		else:
			rank = self.comm.rank
		oname = self.ofmt.format(name=self.name, rank=rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m
		return oname

class SignalCutPolyGpu(SignalCutFullGpu):
	def __init__(self, comm, order=3, bsize=400, precon="none", name="cut",
			ofmt="{name}_{rank:02}", dtype=np.float32, output=False):
		"""Signal for handling the ML solution for the values of the cut samples."""
		SignalCutFullGpu.__init__(self, comm, name=name, ofmt=ofmt, dtype=dtype, output=output)
		self.order  = order
		self.bsize  = bsize
		self.basis  = gutils.legbasis(order, bsize)
		self.prec   = precon
		if precon == "leginv":
			self.ibases = gutils.leginverses(self.basis)
	def add_obs(self, id, obs, nmat, Nd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data."""
		Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutPolyGpu(obs.cuts, basis=self.basis)
		# Build our RHS
		obs_rhs = cupy.zeros(pcut.ndof, self.dtype)
		pcut.backward(Nd, obs_rhs)
		obs_rhs = obs_rhs.get()
		# Build our preconditioner. We have precomputed the basis.dot(basis.T) inverse for
		# all block lengths, so we just read out the appropriate one for each block.
		# TODO: Understand why the none-preconditioner suddenly works fine, while
		# the other two breaks convergence. Apparently this only happens when I include
		# 1/ivar in them, otherwise they don't hurt but also don't beat none. Very
		# strange.
		if self.prec == "leginv":
			obs_idiv = self.ibases[pcut.lens.get()-1] / nmat.ivar[pcut.dets].get()[:,None,None]
		elif self.prec == "var":
			obs_idiv = 1/nmat.ivar[pcut.dets].get()
		elif self.prec == "none":
			obs_idiv = [1]
		else: raise ValueError("Unknown precon '%s'" % str(self.prec))
		self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off+pcut.ndof)
		self.off += pcut.ndof
		self.rhs.append(obs_rhs)
		self.idiv.append(obs_idiv)
		gpu_garbage_collect()
	def precon(self, junk):
		# For some reason this fails with negative residuals when I use this well-motivated
		# and symmetric preconditioner, but suddenly works fine when I use no preconditioner at all.
		if self.prec == "leginv":
			bjunk = junk.reshape(-1, self.basis.shape[0])
			bjunk = np.einsum("iab,ib->ia", self.idiv, bjunk)
			return bjunk.reshape(-1)
		elif self.prec == "var":
			bjunk = junk.reshape(-1, self.basis.shape[0]).copy()
			bjunk *= self.idiv[:,None]
			return bjunk.reshape(-1)
		elif self.prec == "none":
			return junk.copy()
		else: raise ValueError("Unknown precon '%s'" % str(self.prec))

# Zippers
# These package our degrees of freedomf or the CG solver

class ArrayZipper:
	def __init__(self, shape, dtype, comm=None):
		self.shape = shape
		self.ndof  = int(np.prod(shape))
		self.dtype = dtype
		self.comm  = comm
	def zip(self, arr):  return arr.reshape(-1)
	def unzip(self, x):  return x.reshape(self.shape).astype(self.dtype, copy=False)
	def dot(self, a, b):
		return np.sum(a*b) if self.comm is None else self.comm.allreduce(np.sum(a*b))

class MapZipper:
	def __init__(self, shape, wcs, dtype, comm=None):
		self.shape, self.wcs = shape, wcs
		self.ndof  = int(np.prod(shape))
		self.dtype = dtype
		self.comm  = comm
	def zip(self, map): return np.asarray(map.reshape(-1))
	def unzip(self, x): return enmap.ndmap(x.reshape(self.shape), self.wcs).astype(self.dtype, copy=False)
	def dot(self, a, b):
		return np.sum(a*b) if self.comm is None else utils.allreduce(np.sum(a*b),self.comm)

class MultiZipper:
	def __init__(self):
		self.zippers = []
		self.ndof	= 0
		self.bins	= []
	def add(self, zipper):
		self.zippers.append(zipper)
		self.bins.append([self.ndof, self.ndof+zipper.ndof])
		self.ndof += zipper.ndof
	def zip(self, *objs):
		return np.concatenate([zipper.zip(obj) for zipper, obj in zip(self.zippers, objs)])
	def unzip(self, x):
		res = []
		for zipper, (b1,b2) in zip(self.zippers, self.bins):
			res.append(zipper.unzip(x[b1:b2]))
		return res
	def dot(self, a, b):
		res = 0
		for (b1,b2), dof in zip(self.bins, self.zippers):
			res += dof.dot(a[b1:b2],b[b1:b2])
		return res
