import numpy as np
import time
from pixell import utils, bunch, enmap, colors
from . import nmat, pmat, tiling, gutils, device
from .logging import L

class MLMapmaker:
	def __init__(self, signals=[], dev=None, noise_model=None, dtype=np.float32, verbose=False):
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
		self.dev      = dev or device.get_device()
		self.dtype    = dtype
		self.verbose  = verbose
		self.noise_model = noise_model or nmat.NmatDetvecs(dev=dev)
		self.reset()
	def reset(self):
		"""Reset the mapmaker, as if it had just been constructed. Also
		resets its signals, so it's ready to make new maps."""
		self.data     = []
		self.dof      = MultiZipper()
		self.ready    = False
		for signal in self.signals:
			signal.reset()
	def add_obs(self, id, obs, deslope=True, noise_model=None):
		# Prepare our tod
		t1 = time.time()
		ctime  = obs.ctime
		srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
		tod    = obs.tod.astype(self.dtype, copy=False)
		t2 = time.time()
		if deslope:
			utils.deslope(tod, w=5, inplace=True)
		t3 = time.time()
		gtod = self.dev.pools["tod"].array(tod)
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
				raise gutils.RecoverableError(msg)
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
		owork = [signal.owork()    for signal in self.signals]
		t2 = time.time()
		for di, data in enumerate(self.data):
			# This is the main place that needs to change for the GPU implementation
			ta1  = self.dev.time()
			gtod = self.dev.pools["tod"].empty([data.ndet, data.nsamp], self.dtype)
			ta2  = self.dev.time()
			for si, signal in reversed(list(enumerate(self.signals))):
				signal.precalc_setup(data.id)
				signal.forward(data.id, gtod, iwork[si])
			ta3 = self.dev.time()
			data.nmat.apply(gtod)
			ta4 = self.dev.time()
			for si, signal in enumerate(self.signals):
				signal.backward(data.id, gtod, owork[si])
				signal.precalc_free(data.id)
			ta5 = self.dev.time()
			self.dev.garbage_collect()
			ta6 = self.dev.time()
			L.print("A z %6.3f P %6.3f N %6.3f P' %6.3f gc %6.4f %s %4d %6d" % (ta2-ta1, ta3-ta2, ta4-ta3, ta5-ta4, ta6-ta5, data.id, *gtod.shape), level=2)
		t3 = self.dev.time()
		result = self.dof.zip(*[signal.from_work(w) for signal,w in zip(self.signals,owork)])
		t4 = self.dev.time()
		L.print("A prep %6.3f PNP %6.3f finish %6.3f" % (t2-t1, t3-t2, t4-t3), level=2)
		return result
	def M(self, x):
		t1 = self.dev.time()
		iwork = self.dof.unzip(x)
		result = self.dof.zip(*[signal.precon(w) for signal, w in zip(self.signals, iwork)])
		t2 = self.dev.time()
		L.print("M %6.3f" % (t2-t1), level=2)
		return result
	def solve(self, maxiter=500, maxerr=1e-6):
		self.prepare()
		rhs    = self.dof.zip(*[signal.rhs for signal in self.signals])
		solver = utils.CG(self.A, rhs, M=self.M, dot=self.dof.dot)
		#solver = utils.Minres(self.A, rhs, dot=self.dof.dot)
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
	def reset(self):
		self.dof   = None
		self.ready = False
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
	def write_misc(self, prefix): pass

# This will be updated to take a tiling instead of a shape, wcs, comm.
# Or could build the tiling internally, hiding that detail from the user.
# No, we need obsinfo to set up the tiling. So have to change the interface
# anyway.

# rhs, div, hits would not be preallocated, but would be left to grow
# in the dynamic pmap.
# During prepare, the tiling would be finalized, and we would perform
# our first reductions

# Terminology:
#  * gmap:  global tile map on cpu
#  * lmap:  local  tile map on cpu
#  * glmap: local  tile map on gpu
#  * dmap:  dynamic map on gpu

class SignalMap(Signal):
	"""Signal describing a non-distributed sky map."""
	def __init__(self, shape, wcs, comm, dev=None, name="sky", ofmt="{name}", output=True,
			ext="fits", dtype=np.float32, ibuf=None, obuf=None, recenter=None,
			sys=None, interpol=None, autocrop=False):
		"""Signal describing a sky map in the coordinate system given by "sys", which defaults
		to equatorial coordinates."""
		Signal.__init__(self, name, ofmt, output, ext)
		self.sys   = sys
		if sys is not None and sys not in ["cel","equ"]:
			raise NotImplementedError("Coordinate system rotation not implemented yet")
		self.recenter = recenter
		self.dtype = dtype
		self.interpol = interpol
		self.data  = {}
		self.comps = "TQU"
		self.ncomp = 3
		self.comm  = comm
		self.autocrop = autocrop
		self.dev   = dev or device.get_device()
		# Set up our internal tiling
		self.shape,  self.wcs = shape, wcs
		self.fshape, self.fwcs, self.pixbox = tiling.infer_fullsky_geometry(shape, wcs)
		if autocrop: self.pixbox = "auto"
		# Buffers to use
		self.ibuf = self.dev.pools[name+"_iwork"]
		self.obuf = self.dev.pools[name+"_owork"]
		# Not sure how to avoid code duplication with reset here,
		# in light of the inheritance
		self.ids      = []
		self.tiledist = None
		# Dynamic RHS map which we will accumulate into
		self.drhs  = self.dev.lib.DynamicMap(*self.fshape, self.dtype)
	def reset(self):
		Signal.reset(self)
		self.ids      = []
		self.tiledist = None
		self.drhs  = self.dev.lib.DynamicMap(*self.fshape, self.dtype)
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
		print("FIXME: Figure out if I'm handling cuts consistently")
		pcut   = pmat.PmatCutFull(obs.cuts, dev=self.dev)
		if pmap is None:
			pmap = pmat.PmatMap(self.fshape, self.fwcs, obs.ctime, obs.boresight, obs.point_offset, obs.polangle, response=obs.response, recenter=self.recenter, dev=self.dev, dtype=Nd.dtype)
			self.dev.garbage_collect()
		# Precompute pointing for the upcoming pmap observations
		# Accumulate local rhs
		t2 = time.time()
		pcut.clear(Nd)
		pmap.backward(Nd, self.drhs)
		self.dev.garbage_collect()
		t3 = time.time()
		L.print("Init map pmat %6.3f rhs %6.3f %s" % (t2-t1,t3-t2,id), level=2)
		# Save the per-obs things we need. Just the pointing matrix in our case.
		# Nmat and other non-Signal-specific things are handled in the mapmaker itself.
		self.data[id] = bunch.Bunch(pmap=pmap, pcut=pcut, nmat=nmat, tod_shape=Nd.shape, tod_dtype=Nd.dtype)
		self.ids.append(id)
	def calc_hits(self, weight=False):
		"""Calculate the local div or hits map. Tiling must be
		finalized before we do this. Will need a different function for non-scalar div"""
		# Could save one map buffer by calculating these separately,
		# but that would be slower
		glhits = self.tiledist.gwmap(buf=self.obuf, dtype=self.dtype) # LocalMap
		for i, id in enumerate(self.ids):
			d   = self.data[id]
			t1  = time.time()
			tod = self.dev.pools["tod"].full(d.tod_shape, 1, d.tod_dtype)
			if weight: d.nmat.white(tod)
			d.pmap.backward(tod, glhits)
			t2  = time.time()
			L.print("Init map hits %d %6.3f %s" % (weight, t2-t1,id), level=2)
		return glhits
	def prepare(self):
		"""Called when we're done adding everything. Sets up the map distribution,
		degrees of freedom and preconditioner."""
		if self.ready: return
		t1 = time.time()
		# Ok, this is where we finally know enough to finish the tiledist
		glrhs  = self.drhs.finalize(); del self.drhs
		self.tiledist = tiling.TileDistribution(self.fshape, self.fwcs,
				glrhs.pixelization, self.comm, pixbox=self.pixbox, dev=self.dev)
		# Reduce to global tiles and free up the arrays on the dpu
		t2 = time.time()
		self.rhs   = self.tiledist.gwmap2dmap(glrhs);  del glrhs
		t3 = time.time()
		glhits = self.calc_hits(weight=False)
		self.dev.garbage_collect()
		t4 = time.time()
		self.hits  = self.tiledist.gwmap2dmap(glhits); del glhits
		self.hits  = self.hits[:,:1]
		t5 = time.time()
		gldiv = self.calc_hits(weight=True)
		self.dev.garbage_collect()
		t6 = time.time()
		self.div   = self.tiledist.gwmap2dmap(gldiv); del gldiv
		self.div   = self.div[:,:1]
		t7 = time.time()
		# Set up our degrees of freedom
		self.dof   = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
		self.idiv  = gutils.safe_inv(self.div)
		## FIXME
		#print("div", self.div.shape, self.dev.np.max(self.div), self.dev.np.min(self.div))
		#self.idiv  = gutils.safe_invert_ivar(self.div)
		#print("idiv", self.idiv.shape, self.dev.np.max(self.idiv), self.dev.np.min(self.idiv))
		t8 = time.time()
		L.print("Prep fin %6.3f div %6.3f red %6.3f prec %6.3f" % (t2-t1, t4-t3+t6-t5, t3-t2+t5-t4+t7-t6, t8-t7), level=2)
		self.ready = True
	def forward(self, id, gtod, glmap, tmul=1):
		"""map2tod operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into tod."""
		if id not in self.data: return # Should this really skip silently like this?
		if tmul != 1: gtod *= tmul
		self.data[id].pmap.forward(gtod, glmap)
	def backward(self, id, gtod, glmap, mmul=1):
		"""tod2map operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into map"""
		if id not in self.data: return
		if mmul != 1: glmap *= mmul
		self.data[id].pmap.backward(gtod, glmap)
	def precalc_setup(self, id): self.data[id].pmap.precalc_setup()
	def precalc_free (self, id): self.data[id].pmap.precalc_free()
	def precon(self, gmap):
		return self.idiv * gmap
	def to_work(self, gmap): return self.tiledist.dmap2gwmap(gmap, buf=self.ibuf)
	def from_work(self, glmap): return self.tiledist.gwmap2dmap(glmap)
	def owork(self): return self.tiledist.gwmap(self.obuf, dtype=self.dtype)
	def write(self, prefix, tag, m):
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		omap  = self.tiledist.dmap2omap(m)
		if self.comm.rank == 0:
			enmap.write_map(oname, omap)
		return oname
	def write_misc(self, prefix):
		self.write(prefix, "rhs",  self.rhs)
		self.write(prefix, "ivar", self.div[:,0])
		self.write(prefix, "hits", self.hits[:,0])
		self.write(prefix, "bin",  self.idiv*self.rhs)

class SignalCutFull(Signal):
	# Placeholder for when we have a gpu implementation
	def __init__(self, comm, dev=None, name="cut", ofmt="{name}_{rank:02}", dtype=np.float32,
			output=False):
		"""Signal for handling the ML solution for the values of the cut samples."""
		Signal.__init__(self, name, ofmt, output, ext="hdf")
		self.comm  = comm
		self.dev   = dev or device.get_device()
		self.dtype = dtype
		self.off   = 0
		self.rhs   = []
		self.idiv  = []
		self.data  = {}
	def reset(self):
		Signal.reset(self)
		self.off   = 0
		self.rhs   = []
		self.idiv  = []
		self.data  = {}
	def add_obs(self, id, obs, nmat, Nd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data."""
		Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutFull(obs.cuts, dev=self.dev)
		# Build our RHS
		obs_rhs = self.dev.np.zeros(pcut.ndof, self.dtype)
		pcut.backward(Nd, obs_rhs)
		obs_rhs = obs_rhs.get()
		# Build our preconditioner.
		obs_div = self.dev.np.ones(pcut.ndof, self.dtype)
		Nd[:]   = 0
		pcut.forward(Nd, obs_div)
		nmat.white(Nd)
		pcut.backward(Nd, obs_div)
		obs_idiv = 1/self.dev.get(obs_div) # back to the cpu
		self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off+pcut.ndof)
		self.off += pcut.ndof
		self.rhs.append(obs_rhs)
		self.idiv.append(obs_idiv)
		self.dev.garbage_collect()
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
		return junk*self.idiv
	def backward(self, id, gtod, gjunk):
		if id not in self.data: return
		d = self.data[id]
		d.pcut.backward(gtod, gjunk[d.i1:d.i2])
	# TODO: Check if these should use mempools
	def to_work  (self, x): return self.dev.np.array(x)
	def from_work(self, x): return self.dev.get(x)
	def owork(self): return self.dev.np.zeros(self.rhs.shape, self.rhs.dtype)
	def write(self, prefix, tag, m):
		if self.comm is None:
			rank = 0
		else:
			rank = self.comm.rank
		oname = self.ofmt.format(name=self.name, rank=rank)
		oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m
		return oname

class SignalCutPoly(SignalCutFull):
	def __init__(self, comm, dev=None, order=3, bsize=400, precon="none", name="cut",
			ofmt="{name}_{rank:02}", dtype=np.float32, output=False):
		"""Signal for handling the ML solution for the values of the cut samples."""
		SignalCutFull.__init__(self, comm, dev=dev, name=name, ofmt=ofmt, dtype=dtype, output=output)
		self.order  = order
		self.bsize  = bsize
		self.basis  = gutils.legbasis(order, bsize)
		self.prec   = precon
		if precon != "none":
			raise NotImplementedError("SignalCutPoly precons other than 'none' currently don't work")
		if precon == "leginv":
			self.ibases = gutils.leginverses(self.basis)
	def add_obs(self, id, obs, nmat, Nd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		nmat a noise model, representing the inverse noise covariance matrix,
		and Nd the result of applying the noise model to the detector time-ordered data."""
		Nd      = Nd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutPoly(obs.cuts, basis=self.basis, dev=self.dev)
		# Build our RHS
		obs_rhs = self.dev.np.zeros(pcut.ndof, self.dtype)
		pcut.backward(Nd, obs_rhs)
		obs_rhs = self.dev.get(obs_rhs)
		# Build our preconditioner. We have precomputed the basis.dot(basis.T) inverse for
		# all block lengths, so we just read out the appropriate one for each block.
		# TODO: Understand why the none-preconditioner suddenly works fine, while
		# the other two breaks convergence. Apparently this only happens when I include
		# 1/ivar in them, otherwise they don't hurt but also don't beat none. Very
		# strange.
		if self.prec == "leginv":
			obs_idiv = self.ibases[self.dev.get(pcut.lens)-1] / self.dev.get(nmat.ivar[pcut.dets])[:,None,None]
		elif self.prec == "var":
			obs_idiv = 1/self.dev.get(nmat.ivar[pcut.dets])
		elif self.prec == "none":
			obs_idiv = [1]
		else: raise ValueError("Unknown precon '%s'" % str(self.prec))
		self.data[id] = bunch.Bunch(pcut=pcut, i1=self.off, i2=self.off+pcut.ndof)
		self.off += pcut.ndof
		self.rhs.append(obs_rhs)
		self.idiv.append(obs_idiv)
		self.dev.garbage_collect()
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

# Mapmaking function
def make_map(mapmaker, loader, obsinfo, comm, inds=None, prefix=None, dump=[], maxiter=500, maxerr=1e-7, prealloc=True, ignore="recover"):
	if prefix is None: prefix = ""
	if inds is None:
		dist = tiling.distribute_tods_semibrute(obsinfo, comm.size)
		inds = np.where(dist.owner == comm.rank)[0]
	# Set up exception types we will ignore
	if   ignore == "all":     etypes = (Exception,)
	elif ignore == "missing": etypes = (utils.DataMissing,)
	elif ignore == "recover": etypes = (utils.DataMissing, gutils.RecoverableError)
	elif ignore == "none":    etypes = ()
	else: raise ValueError("Unrecognized error ignore setting '%s'" % str(ignore))
	# Accept either a list of integers or a single integer
	try: dump = list(dump)
	except TypeError: dump = [dump]
	dev = mapmaker.dev
	# Set up memory pools. Setting these up before-hand is
	# actually more memory-efficient, as long as our estimate is
	# good.
	if prealloc:
		ntot_max = np.max(obsinfo.ndet[inds]*obsinfo.nsamp[inds])
		setup_buffers(dev, ntot_max)
	# Start map from scartch
	mapmaker.reset()
	# Add our observations
	for i, ind in enumerate(inds):
		id    = obsinfo.id[ind]
		t1    = time.time()
		try:
			data  = loader.load(id)
		except etypes as e:
			L.print("Skipped %s: %s" % (id, str(e)), level=2, color=colors.red)
			continue
		t2    = time.time()

		print("FIXME handle empty cuts")
		if data.cuts.size == 0: data.cuts = np.array([[0],[10],[1]],np.int32)

		try:
			mapmaker.add_obs(id, data, deslope=False)
		except etypes as e:
			L.print("Skipped %s: %s" % (id, str(e)), level=2, color=colors.red)
			continue
		dev.garbage_collect()
		del data
		t3    = time.time()
		L.print("Processed %s in %6.3f. Read %6.3f Add %6.3f" % (id, t3-t1, t2-t1, t3-t2), level=2)

	nobs = comm.allreduce(len(mapmaker.data))
	if nobs == 0:
		L.print("No tods survived!", id=0, level=0, color=colors.red)
		return None

	mapmaker.prepare()
	# Write rhs and ivar
	for signal in mapmaker.signals:
		if signal.output:
			signal.write_misc(prefix)

	# Solve the equation system
	for step in mapmaker.solve(maxiter=maxiter, maxerr=maxerr):
		will_dump = len(dump) > 0 and (step.i in dump or step.i % dump[-1] == 0)
		L.print("CG %4d %15.7e  %6.3f s%s" % (step.i, step.err, step.t, " (write)" if will_dump else ""), id=0, level=1, color=colors.lgreen)
		if will_dump:
			for signal, val in zip(mapmaker.signals, step.x):
				if signal.output:
					signal.write(prefix, "map%04d" % step.i, val)
	# Write the final result
	for signal, val in zip(mapmaker.signals, step.x):
		if signal.output:
			signal.write(prefix, "map", val)

def make_maps_perobs(mapmaker, loader, obsinfo, comm, comm_per, inds=None, prefix=None, dump=[], maxiter=500, maxerr=1e-7, prealloc=True, ignore="recover"):
	"""Like make_map, but makes one map per subobs. NB! The communicators in the mapmaker
	signals must be COMM_SELF for this to work. TODO: Make this simpler."""
	if inds is None:
		inds = list(range(comm.rank, len(obsinfo), comm.size))
	if prefix is None:
		prefix = ""
	ntot_max = np.max(obsinfo.ndet[inds]*obsinfo.nsamp[inds])
	if prealloc: setup_buffers(mapmaker.dev, ntot_max)
	# Map indivdual tods
	for ind in inds:
		subinfo = obsinfo[ind:ind+1]
		id      = subinfo.id[0]
		subpre  = prefix + id.replace(":","_") + "_"
		L.print("Mapping %s" % id)
		make_map(mapmaker, loader, subinfo, comm_per, prefix=subpre, dump=dump, maxiter=maxiter, maxerr=maxerr, prealloc=False, ignore=ignore)

def make_maps_depth1(mapmaker, loader, obsinfo, comm, comm_per, prefix=None, dump=[], maxiter=500, maxerr=1e-7, fullinfo=None, prealloc=True, ignore="recover"):
	# Find scanning periods. We want to base this on the full
	# set of observations, so depth-1 maps cover consistent periods
	# even if 
	from pixell import bench
	if prefix   is None: prefix = ""
	if fullinfo is None: fullinfo = loader.query("type='obs'")
	periods = gutils.find_scan_periods(fullinfo)
	# Which period each obs belongs to
	pinds   = utils.find_range(periods, obsinfo.ctime+obsinfo.dur/2)
	# Get rid of observations that don't belong to a period. This shouldn't happen
	bad = pinds<0
	if np.any(bad):
		L.print("Warning: %d obs with no period! %s" % (np.sum(bad), ", ".join(obsinfo.id[bad])), color=colors.red, id=0)
		obsinfo = obsinfo[~bad]
		pinds   = pinds  [~bad]
	# * We could split by band, tube, wafer etc. here, but for now, we will
	# keep things simple by requiring the user to run the mapmaker separately
	# for things they want to keep separate. E.g. one run for f090 and one for f150,
	# just like with the other mapmaking modes.
	# * How to paralellize? For big runs, there will be many more periods than
	# obs per period, so one would want to paralellize over periods. But for small
	# runs it's the other way around. And there's an awkward intermediate regime
	# where one would want both. Let's do it over periods for now
	# * period i has obsinds order[edges[i]:edges[i+1]] and trange periods[pids[i]],
	# which should be the same as periods[i]
	pids, order, edges = utils.find_equal_groups_fast(pinds)
	# Set up buffers. We get all the tods we will process to calculate how big
	# buffers we will need
	my_pinds   = list(range(comm.rank, len(pids), comm.size))
	my_obsinds = np.concatenate([order[edges[pind]:edges[pind+1]] for pind in my_pinds])
	ntot_max   = np.max(obsinfo.ndet[my_obsinds]*obsinfo.nsamp[my_obsinds])
	if prealloc: setup_buffers(mapmaker.dev, ntot_max)
	# Finally map each period
	for pind in my_pinds:
		pid    = pids[pind]
		subinfo= obsinfo[order[edges[pind]:edges[pind+1]]]
		# Name period after starting ctime for now
		name   = "%10.0f" % periods[pid][0]
		subpre = prefix + name + "_"
		L.print("Mapping period %10.0f:%10.0f with %d obs" % (*periods[pid], len(subinfo)))
		make_map(mapmaker, loader, subinfo, comm_per, prefix=subpre, dump=dump, maxiter=maxiter, maxerr=maxerr, prealloc=False, ignore=ignore)


def setup_buffers(dev, ntot, dtype=np.float32, ndet_guess=1000):
	"""Pre-allocate memory buffers for mapmaking. Pass in the worst-case
	ntot = ndet*nsamp.

	Pre-allocation isn't really necessary,
	but automatically growing the buffers has some overhead, and I've unexpectedly
	run out of memory when not doing this. This should be investigated futher.

	This function will need to be updated if which memory pools are used is changed."""
	# These are the big ones
	ctype = utils.complex_dtype(dtype)
	ftot = ntot//2 + ndet_guess
	dev.pools["pointing"]   .empty((3, ntot), dtype=dtype)
	dev.pools["tod"]        .empty(ntot, dtype=dtype)
	dev.pools["ft" ]        .empty(ftot, dtype=ctype)
	dev.pools["fft_scratch"].empty(ftot, dtype=ctype)
	dev.pools.reset()
	# The remaining are:
	# * plan
	# * cut
	# * sky_iwork
	# * sky_owork
	# * nmat_work
	# We don't have good estimates for these, so we don't pre-allocate them.
	# But they should be small

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
