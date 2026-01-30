import numpy as np, os, warnings
import time
from pixell import utils, bunch, enmap, colors, wcsutils, bench, config
from . import nmat, pmat, tiling, gutils, device, socal
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
		t1 = self.dev.time()
		ctime  = obs.ctime
		srate  = (len(ctime)-1)/(ctime[-1]-ctime[0])
		tod    = obs.tod.astype(self.dtype, copy=False)
		t2 = self.dev.time()
		if deslope:
			utils.deslope(tod, w=5, inplace=True)
		t3 = self.dev.time()
		gtod = self.dev.pools["tod"].array(tod)
		del tod
		# Allow the user to override the noise model on a per-obs level
		if noise_model is None: noise_model = self.noise_model
		# Build the noise model from the obs unless a fully
		# initialized noise model was passed
		if noise_model.ready:
			iN = noise_model
		else:
			try:
				iN = noise_model.build(gtod, srate=srate, obs=obs)
				#iN.write("test_iN.hdf")
				#gtod0 = gtod.copy()
				#ft   = self.dev.lib.rfft(gtod)
				#iNft = iN.apply(ft.copy(), nofft=True)
				#ps = self.dev.get(gutils.downgrade(self.dev.np.mean(self.dev.np.abs(iNft)**2,0),10))
				#f  = gutils.downgrade(np.fft.rfftfreq(gtod.shape[1], 1/srate),10)
				#np.savetxt("test_ps_iNd.txt", np.array([f,ps]).T, fmt="%15.7e")
				#chisq = self.dev.np.sum(self.dev.np.conj(ft)*iNft,0)
				#print(self.dev.np.std(chisq.real))
				#print(self.dev.np.std(chisq.imag))
				#chisq = self.dev.get(gutils.downgrade(chisq.real,10))
				#np.savetxt("test_chisq.txt", np.array([f,chisq]).T, fmt="%15.7e")

				#df = srate/gtod.shape[1]
				#i  = int(0.25/df)
				#print(i)
				#np.save("test_ft_025.npy", self.dev.get(ft[:,i-10:i+10]))
				#np.save("test_iNft_025.npy", self.dev.get(iNft[:,i-10:i+10]))
				#1/0
			except Exception as e:
				msg = f"FAILED to build a noise model for observation='{id}' : '{e}'"
				raise gutils.RecoverableError(msg)
		t4 = self.dev.time()
		# And apply it to the tod
		gtod = iN.apply(gtod)
		t5 = self.dev.time()
		# This is our last chance to safely abort, so check that our data makes sense
		if not self.dev.np.isfinite(self.dev.np.sum(gtod)):
			raise gutils.RecoverableError(f"Invalid value in N\"tod")
		# Add the observation to each of our signals
		for signal in self.signals:
			signal.add_obs(id, obs, iN, gtod)
		# TODO: update our stats/info here
		t6 = self.dev.time()
		L.print("Init sys trun %6.3f ds %6.3f Nb %6.3f N %6.3f add sigs %6.3f %s" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, id), level=2)
		# Save only what we need about this observation
		self.data.append(bunch.Bunch(id=id, ndet=len(obs.dets), nsamp=len(ctime),
			dets=obs.dets, iN=iN))
	def prepare(self):
		if self.ready: return
		t1 = self.dev.time()
		for signal in self.signals:
			signal.prepare()
			self.dof.add(signal.dof)
		t2 = self.dev.time()
		L.print("Prep sys %6.3f" % (t2-t1), level=2)
		self.ready = True
	def A(self, x):
		t1 = self.dev.time()
		iwork = [signal.to_work(m) for signal,m in zip(self.signals,self.dof.unzip(x))]
		owork = [signal.owork()    for signal in self.signals]
		t2 = self.dev.time()
		for di, data in enumerate(self.data):
			# This is the main place that needs to change for the GPU implementation
			ta1  = self.dev.time()
			gtod = self.dev.pools["tod"].empty([data.ndet, data.nsamp], self.dtype)
			ta2  = self.dev.time()
			for si, signal in reversed(list(enumerate(self.signals))):
				signal.precalc_setup(data.id)
				signal.forward(data.id, gtod, iwork[si])
			ta3 = self.dev.time()
			data.iN.apply(gtod)
			ta4 = self.dev.time()
			for si, signal in enumerate(self.signals):
				signal.backward(data.id, gtod, owork[si])
				signal.precalc_free(data.id)
			ta5 = self.dev.time()
			self.dev.garbage_collect()
			ta6 = self.dev.time()
			L.print("A z %6.3f P %6.3f N %6.3f P' %6.3f gc %6.4f %s %4d %6d" % (ta2-ta1, ta3-ta2, ta4-ta3, ta5-ta4, ta6-ta5, data.id, *gtod.shape), level=2)
		t3 = self.dev.time()
		result = self.dof.zip([signal.from_work(w) for signal,w in zip(self.signals,owork)])
		t4 = self.dev.time()
		L.print("A prep %6.3f PNP %6.3f finish %6.3f" % (t2-t1, t3-t2, t4-t3), level=2)
		return result
	def M(self, x):
		t1 = self.dev.time()
		iwork = self.dof.unzip(x)
		result = self.dof.zip([signal.precon(w) for signal, w in zip(self.signals, iwork)])
		t2 = self.dev.time()
		L.print("M %6.3f" % (t2-t1), level=2)
		return result
	def solve(self, maxiter=500, maxerr=1e-6):
		self.prepare()
		rhs    = self.dof.zip([signal.rhs for signal in self.signals])
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
	def __init__(self, name, ofmt, outputs, ext):
		"""Initialize a Signal. It probably doesn't make sense to construct a generic signal
		directly, though. Use one of the subclasses.
		Arguments:
		* name: The name of this signal, e.g. "sky", "cut", etc.
		* ofmt: The format used when constructing output file prefix
		* outputs: Which of our information to output. Empty list or
		  None to output nothing. "all" for all valid products.
		* ext: The extension used for the files.
		"""
		self.name   = name
		self.ofmt   = ofmt
		self.ext    = ext
		self.dof    = None
		self.ready  = False
		self.set_outputs(outputs)
	def reset(self):
		self.dof   = None
		self.ready = False
	def add_obs(self, id, obs, iN, iNd): pass
	def prepare(self): self.ready = True
	def forward (self, id, tod, x): pass
	def backward(self, id, tod, x): pass
	def precalc_setup(self, id): pass
	def precalc_free (self, id): pass
	def precon(self, x): return x
	def to_work  (self, x): return x.copy()
	def from_work(self, x): return x
	def owork(self): raise NotImplementedError
	def set_outputs(self, outputs):
		self.outputs= check_outputs(outputs, self.valid_outputs, self.__class__.__name__)
	def written(self, prefix): raise NotImplementedError
	def write   (self, prefix, x, tag="dummy", suffix="", force=False): pass
	def write_misc(self, prefix): pass
	valid_outputs = []

class SignalMap(Signal):
	"""Signal describing a sky map."""
	def __init__(self, shape, wcs, comm, dev=None, name="sky", ofmt="{name}",
			outputs=["map","ivar"], precon="ivar", ext="fits", dtype=np.float32, ibuf=None, obuf=None,
			sys=None, interpol=None, autocrop=False, ocomps=None):
		"""Signal describing a sky map in the coordinate system given by "sys", which defaults
		to equatorial coordinates."""
		Signal.__init__(self, name, ofmt, outputs, ext)
		self.sys   = sys or "cel"
		#if self.sys not in ["cel","equ","hor"]:
		#	raise NotImplementedError("Coordinate system rotation not implemented yet")
		self.dtype = dtype
		self.interpol = interpol
		self.data  = {}
		# The Stokes components we will *output*. This only affects the
		# disk writes, not the solution or memory use
		self.ocomps= (0,1,2) if ocomps is None else utils.astuple(ocomps)
		# Number of Stokes components we solve for. Our low-level code
		# only supports 3
		self.ncomp = 3
		self.comm  = comm
		self.autocrop = autocrop
		if precon not in ["ivar", "div"]:
			raise ValueError("precon must be 'ivar' or 'div'")
		self.prec_mode = precon
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
		# Used as reference point for time maps
		self.tref  = np.inf
	def reset(self):
		Signal.reset(self)
		self.ids      = []
		self.tiledist = None
		self.drhs  = self.dev.lib.DynamicMap(*self.fshape, self.dtype)
	def add_obs(self, id, obs, iN, iNd, pmap=None, pcut=None):
		"""Add and process an observation, building the pointing matrix
		and our part of the RHS. "obs" should be an Observation axis manager,
		iN a noise model, representing the inverse noise covariance matrix,
		and iNd the result of applying the noise model to the detector time-ordered data.
		"""
		ctime     = obs.ctime
		self.tref = min(self.tref, ctime[0])
		t1        = time.time()
		# could pass this in, but fast to construct. Should ideally match what's used in
		# signal_cut to be safest, but only uses .clear which should be the same for all
		# of them anyway
		if pmap is None:
			try:
				pmap = pmat.PmatMap(self.fshape, self.fwcs, obs.ctime, obs.boresight, obs.point_offset, obs.polangle, sys=self.sys, response=obs.response, site=obs.site, dev=self.dev, dtype=iNd.dtype)
			except RuntimeError:
				# This happens if the pointing fit goes badly wrong, which
				# can happen if the tod crosses the poles. Too late to cancel
				# the tod at this point, so instead we will skip it just in this
				# signal
				pmap = pmat.PmatDummy()
			self.dev.garbage_collect()
		if pcut is None:
			# This doesn't have to match the exact cut type in SignalCut.
			# Since we only use .clear() here, they only need to agree on
			# which samples are cut.
			pcut = pmat.PmatCutFull(obs.cuts, dev=self.dev)
		# Precompute pointing for the upcoming pmap observations
		# Accumulate local rhs
		t2 = time.time()
		pcut.clear(iNd)
		pmap.backward(iNd, self.drhs)
		self.dev.garbage_collect()
		t3 = time.time()
		L.print("Init map pmat %6.3f rhs %6.3f %s" % (t2-t1,t3-t2,id), level=2)
		# Save the per-obs things we need. Just the pointing matrix in our case.
		# Nmat and other non-Signal-specific things are handled in the mapmaker itself.
		self.data[id] = bunch.Bunch(pmap=pmap, pcut=pcut, iN=iN, tod_shape=iNd.shape, tod_dtype=iNd.dtype, ctime=ctime)
		self.ids.append(id)
	def calc_hits(self, weight=False, fill=1, name="hits"):
		"""Calculate the local ivar or hits map. Tiling must be
		finalized before we do this."""
		# Could save one map buffer by calculating these separately,
		# but that would be slower
		glhits = self.tiledist.gwmap(buf=self.obuf, dtype=self.dtype) # LocalMap
		for i, id in enumerate(self.ids):
			d   = self.data[id]
			t1  = time.time()
			tod = self.dev.pools["tod"].empty(d.tod_shape, d.tod_dtype)
			if isinstance(fill, str):
				if fill == "time": tod[:] = self.dev.np.array(d.ctime)-self.tref
				else: raise ValueError("Unrecognized fill '%s'" % fill)
			else: tod[:] = fill
			if weight: d.iN.white(tod)
			d.pcut.clear(tod)
			d.pmap.backward(tod, glhits)
			t2  = time.time()
			L.print("Init map %s %d %6.3f %s" % (name, weight, t2-t1,id), level=2)
		return glhits
	def calc_div(self, type="full", weight=True, name="div"):
		"""Calculate the full [3,3] div map. This is a bit tricker than
		calc_hits because LocalMaps are always [3]. Se we will return a
		list of 3 of them"""
		self.obuf.reset()
		gldiv = []
		for ci in range(self.ncomp):
			imap = self.tiledist.gwmap(buf=self.ibuf, dtype=self.dtype)
			imap.arr[:,ci] = 1
			omap = self.tiledist.gwmap(buf=self.obuf, dtype=self.dtype, reset=False)
			for i, id in enumerate(self.ids):
				d   = self.data[id]
				t1  = time.time()
				tod = self.dev.pools["tod"].empty(d.tod_shape, d.tod_dtype)
				d.pmap.forward(tod, imap)
				# d.pcut.clear(tod) # not needed because iN.white is diag
				if weight: d.iN.white(tod)
				d.pcut.clear(tod)
				d.pmap.backward(tod, omap)
				t2  = time.time()
				L.print("Init map %s [%d,:] %d %6.3f %s" % (name, ci, weight, t2-t1,id), level=2)
			gldiv.append(omap)
		return gldiv
	def prepare(self):
		"""Called when we're done adding everything. Sets up the map distribution,
		degrees of freedom and preconditioner."""
		if self.ready: return
		mybench = bench.Bench()
		with mybench.mark("fin"):
			self.tref = min(self.comm.allgather(self.tref))
			if not np.isfinite(self.tref): self.tref = 0
			# Ok, this is where we finally know enough to finish the tiledist
			glrhs  = self.drhs.finalize(); del self.drhs
			self.tiledist = tiling.TileDistribution(self.fshape, self.fwcs,
					glrhs.pixelization, self.comm, pixbox=self.pixbox, dev=self.dev)
		# Reduce to global tiles and free up the arrays on the dpu
		with mybench.mark("red"):
			self.rhs   = self.tiledist.gwmap2dmap(glrhs);  del glrhs
		with mybench.mark("misc"):
			if "hits" in self.outputs:
				glhits = self.calc_hits(weight=False)
				self.dev.garbage_collect()
		with mybench.mark("red"):
			if "hits" in self.outputs:
				self.hits  = self.tiledist.gwmap2dmap(glhits); del glhits
				self.hits  = self.hits[:,0]
		# Build ivar or div. This is a bit messy
		with mybench.mark("iprec"):
			if self.prec_mode == "ivar":
				gliprec = [self.calc_hits(weight=True, name=self.prec_mode)]
			else:
				gliprec = self.calc_div(weight=True, name=self.prec_mode)
			self.dev.garbage_collect()
		with mybench.mark("red"):
			self.iprec  = [self.tiledist.gwmap2dmap(a) for a in gliprec]; del gliprec
			if self.prec_mode == "ivar":
				self.iprec = self.iprec[0][:,0]  # [ntile,ty,tx]
				self.ivar  = self.iprec
			else:
				self.iprec = np.moveaxis(self.iprec,0,1) # [ntile,3,3,ty,tx]
				self.ivar  = self.iprec[:,0,0] # still needed for the time-map
		# Set up our degrees of freedom
		self.dof   = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
		with mybench.mark("prec"):
			self.prec = gutils.safe_invert_prec(self.iprec)
		with mybench.mark("misc"):
			if "time" in self.outputs:
				gltime     = self.calc_hits(weight=True, fill="time", name="time")
		with mybench.mark("red"):
			if "time" in self.outputs:
				self.tmap  = self.tiledist.gwmap2dmap(gltime)[:,0]; del gltime
		with mybench.mark("misc"):
			if "time" in self.outputs:
				if self.prec_mode == "ivar": var = self.prec
				else: var = gutils.safe_invert_prec(self.ivar)
				self.tmap *= var
		L.print("Prep fin %6.3f iprec %6.3f misc %6.3f red %6.3f prec %6.3f" % tuple([
			mybench.t_tot[name] for name in ["fin","iprec","misc","red","prec"]]), level=2)
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
	def precalc_setup(self, id, reset_buffer=True): self.data[id].pmap.precalc_setup(reset_buffer=reset_buffer)
	def precalc_free (self, id): self.data[id].pmap.precalc_free()
	def precon(self, map):
		return gutils.apply_prec(self.prec, map)
	def to_work(self, gmap): return self.tiledist.dmap2gwmap(gmap, buf=self.ibuf)
	def from_work(self, glmap): return self.tiledist.gwmap2dmap(glmap)
	def owork(self): return self.tiledist.gwmap(self.obuf, dtype=self.dtype)
	def written(self, prefix):
		done  = True
		for tag in self.outputs:
			oname = self.ofmt.format(name=self.name)
			oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
			done  = done and os.path.exists(oname)
		return done
	def write(self, prefix, m, tag="map", suffix="", force=False, extra={}, oslice=None):
		if not force and tag not in self.outputs: return
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s%s.%s" % (prefix, oname, tag, suffix, self.ext)
		omap  = self.tiledist.dmap2omap(m)
		if self.comm.rank == 0:
			if oslice is None: oslice = np.ix_(*(self.ocomps,)*(omap.ndim-2))
			enmap.write_map(oname, omap[oslice], extra=extra)
		return oname
	def write_misc(self, prefix):
		# Need the if test here despite write also testing, because
		# we're referring to members that might not exist
		if "rhs"  in self.outputs: self.write(prefix, self.rhs,  tag="rhs")
		if "ivar" in self.outputs: self.write(prefix, self.ivar, tag="ivar")
		if "div"  in self.outputs:
			if self.prec_mode == "div": self.write(prefix, self.iprec,  tag="div")
			else: warnings.warn("Cannot output div: Not calculated with precon '%s'" % str(self.prec_mode))
		if "hits" in self.outputs: self.write(prefix, self.hits, tag="hits")
		if "bin"  in self.outputs: self.write(prefix, self.precon(self.rhs), tag="bin")
		if "time" in self.outputs:
			self.write(prefix, self.tmap, tag="time", extra={"TREF":self.tref})
	valid_outputs = ["map","ivar","hits","bin","rhs","div","time"]

# What do we need for multi-band mapmaking?
# Just loop over SignalMaps, but override their TileDistribution
# ompi to a new one built from their pixbox union

class SignalMapMulti(Signal):
	"""Signal describing a multiband sky map."""
	def __init__(self, shape, wcs, bands, comm, dev=None, name="sky", ofmt="{name}", outputs=["map","ivar"],
			ext="fits", dtype=np.float32, ibuf=None, obuf=None, sys=None, interpol=None, precon="ivar", autocrop=False,
			ocomps=None):
		self.mapsigs = [
			SignalMap(shape, wcs, comm, dev=dev, name=band + "_" + name, ofmt=ofmt, outputs=outputs,
				ext=ext, dtype=dtype, ibuf=ibuf, obuf=obuf, sys=sys, interpol=interpol, precon=precon,
				autocrop=autocrop, ocomps=ocomps)
			for band in bands]
		# Must happen after self.mapsigs has been created
		Signal.__init__(self, name, ofmt, outputs, ext)
		self.bands = bands
		self.dev   = dev or device.get_device()
		self.data  = {}
		self.ids   = []
		self.ready = False
	def reset(self):
		for mapsig in self.mapsigs: mapsig.reset()
	def add_obs(self, id, obs, iN, iNd):
		# Find the det-ranges for each of our bands in obs
		band_info = []
		for bi, band in enumerate(self.bands):
			inds   = np.where(obs.bands == band)[0]
			# Skip band that's not present
			if len(inds) == 0: continue
			d1, d2 = inds[0], inds[-1]+1
			# Ensure that they're contiguous!
			assert d2-d1 == len(inds), "Detectors must be sorted by band!"
			# Make obs for the band subset
			bobs = obs.copy()
			bobs.point_offset = obs.point_offset[d1:d2]
			bobs.polangle     = obs.polangle[d1:d2]
			bobs.cuts         = obs.cuts[d1:d2]
			bobs.response     = obs.response[:,d1:d2] if obs.response is not None else None
			# Also need to restrict iN. iN is only used for the precondtioner, so
			# only need to handle this part. This is a bit hacky though, as theoretically
			# a future preconditioner could use more of iN...
			biN = nmat.NmatWhite(ivar=iN.ivar[d1:d2], nwin=iN.nwin, dev=self.dev)
			# And finally forward to our band signals
			self.mapsigs[bi].add_obs(id, bobs, biN, iNd[d1:d2])
			band_info.append((bi,band,d1,d2))
		self.data[id] = bunch.Bunch(band_info=band_info)
		self.ids.append(id)
	def prepare(self):
		"""Called when we're done adding everything. Sets up the map distribution,
		degrees of freedom and preconditioner."""
		# Make all our children prepare, and build up the total degrees of freedom
		self.dof = MultiZipper()
		for mapsig in self.mapsigs:
			mapsig.prepare()
			self.dof.add(mapsig.dof)
		self.rhs = [mapsig.rhs for mapsig in self.mapsigs]
		## Harmonize the output pixelization, so all bands cover the same pixels.
		## This makes plotting easier, but also means we can't have different
		## resolution for the different band maps. But to allow that, we would need
		## to pass in multiple geometries anyway, which we don't do. So this doesn't
		## make us less general than we already area.
		## This is a bit hacky, since it messes with the internals of TileDistribution.
		#pixboxes = np.array([mapsig.tiledist.pixbox for mapsig in self.mapsigs])
		#pixbox   = np.array([np.min(pixboxes[:,0,:],0),np.max(pixboxes[:,1,:],0)])
		#for mi, mapsig in enumerate(self.mapsigs):
		#	td = mapsig.tiledist
		#	mapsig.pixbox = pixbox
		#	_, mapsig.tiledist.pwcs = enmap.crop_geometry(mapsig.shape, mapsig.wcs, pixbox=pixbox, recenter=True)
		#	td.ompi = tiling.build_omap_mpi(mapsig.shape, mapsig.wcs, td.tshape, td.dist.cell_inds, mapsig.comm, pixbox=pixbox)
		self.ready = True
	def forward(self, id, gtod, glmap, tmul=1):
		"""map2tod operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into tod."""
		if id not in self.data: return # Should this really skip silently like this?
		for bi, band, d1, d2 in self.data[id].band_info:
			self.mapsigs[bi].forward(id, gtod[d1:d2], glmap[bi], tmul=tmul)
	def backward(self, id, gtod, glmap, mmul=1):
		"""tod2map operation. For tiled maps, the map should be in work distribution,
		as returned by unzip. Adds into map"""
		if id not in self.data: return
		for bi, band, d1, d2 in self.data[id].band_info:
			self.mapsigs[bi].backward(id, gtod[d1:d2], glmap[bi], mmul=mmul)
	def precalc_setup(self, id):
		self.dev.pools["pointing"].reset()
		self.dev.pools["plan"].reset()
		for bi, band, d1, d2 in self.data[id].band_info:
			self.mapsigs[bi].precalc_setup(id, reset_buffer=False)
	def precalc_free (self, id):
		for bi, band, d1, d2 in self.data[id].band_info:
			self.mapsigs[bi].precalc_free(id)
	def precon(self, gmap):
		return [mapsig.precon(m) for mapsig,m in zip(self.mapsigs, gmap)]
	def to_work(self, gmap):
		return [mapsig.to_work(m) for mapsig, m in zip(self.mapsigs, gmap)]
	def from_work(self, glmap):
		return [mapsig.from_work(m) for mapsig, m in zip(self.mapsigs, glmap)]
	def owork(self):
		return [mapsig.owork() for mapsig in self.mapsigs]
	def set_outputs(self, outputs):
		for mapsig in self.mapsigs:
			mapsig.set_outputs(outputs)
	def written(self, prefix):
		for mapsig in self.mapsigs:
			if not mapsig.written(prefix):
				return False
		return True
	def write(self, prefix, m, tag="map", suffix="", force=False, oslice=None):
		onames = []
		for bi, band in enumerate(self.bands):
			oname = self.mapsigs[bi].write(prefix, m[bi], tag=tag, suffix=suffix, force=force, oslice=oslice)
			onames.append(oname)
		return ", ".join(onames)
	def write_misc(self, prefix):
		for mapsig in self.mapsigs:
			mapsig.write_misc(prefix)
	valid_outputs = SignalMap.valid_outputs

class SignalCutFull(Signal):
	def __init__(self, comm, dev=None, name="cut", ofmt="{name}_{rank:02}", dtype=np.float32,
			outputs=[]):
		"""Signal for handling the ML solution for the values of the cut samples."""
		Signal.__init__(self, name, ofmt, outputs, ext="hdf")
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
	def add_obs(self, id, obs, iN, iNd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		iN a noise model, representing the inverse noise covariance matrix,
		and iNd the result of applying the noise model to the detector time-ordered data."""
		iNd     = iNd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutFull(obs.cuts, dev=self.dev)
		# Build our RHS
		obs_rhs = self.dev.np.zeros(pcut.ndof, self.dtype)
		pcut.backward(iNd, obs_rhs)
		obs_rhs = obs_rhs.get()
		# Build our preconditioner.
		obs_div = self.dev.np.ones(pcut.ndof, self.dtype)
		iNd[:]   = 0
		pcut.forward(iNd, obs_div)
		iN.white(iNd)
		pcut.backward(iNd, obs_div)
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
		self.rhs = np.concatenate(self.rhs)  if len(self.rhs) > 0 else np.zeros(0, self.dtype)
		self.idiv= np.concatenate(self.idiv) if len(self.rhs) > 0 else np.zeros(0, self.dtype)
		self.dof = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
		self.ready = True
	def forward(self, id, gtod, gjunk):
		if id not in self.data: return
		d = self.data[id]
		d.pcut.forward(gtod, gjunk[d.i1:d.i2])
	def precon(self, junk):
		return junk*self.idiv
	def backward(self, id, gtod, gjunk):
		# This function doesn't just accumulate into junk, it also
		# zeros out the cut samples from tod. This lets all the other signals
		# not have to deal with the cuts, as long as the cuts are the first
		# in the signal list (so first backwards, last forwards)
		if id not in self.data: return
		d = self.data[id]
		d.pcut.backward(gtod, gjunk[d.i1:d.i2])
	# TODO: Check if these should use mempools
	def to_work  (self, x): return self.dev.np.array(x)
	def from_work(self, x): return self.dev.get(x)
	def owork(self): return self.dev.np.zeros(self.rhs.shape, self.rhs.dtype)
	def written(self, prefix):
		if self.comm is None:
			rank = 0
		else:
			rank = self.comm.rank
		done = True
		for tag in self.outputs:
			oname = self.ofmt.format(name=self.name, rank=rank)
			oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
			done  = done and os.path.exists(oname)
		if self.comm is not None:
			done = self.comm.allreduce(done)==self.comm.size
		return done
	def write(self, prefix, m, tag="map", suffix="", force=False):
		if not force and tag not in self.outputs: return
		import h5py
		if self.comm is None:
			rank = 0
		else:
			rank = self.comm.rank
		oname = self.ofmt.format(name=self.name, rank=rank)
		oname = "%s%s_%s%s.%s" % (prefix, oname, tag, suffix, self.ext)
		with h5py.File(oname, "w") as hfile:
			hfile["data"] = m
		return oname
	valid_outputs = ["map"]

class SignalCutPoly(SignalCutFull):
	def __init__(self, comm, dev=None, order=3, bsize=400, precon="none", name="cut",
			ofmt="{name}_{rank:02}", dtype=np.float32, outputs=[]):
		"""Signal for handling the ML solution for the values of the cut samples."""
		SignalCutFull.__init__(self, comm, dev=dev, name=name, ofmt=ofmt, dtype=dtype, outputs=outputs)
		self.order  = order
		self.bsize  = bsize
		self.basis  = gutils.legbasis(order, bsize)
		self.prec   = precon
		if precon != "none":
			raise NotImplementedError("SignalCutPoly precons other than 'none' currently don't work")
		if precon == "leginv":
			self.ibases = gutils.leginverses(self.basis)
	def add_obs(self, id, obs, iN, iNd):
		"""Add and process an observation. "obs" should be an Observation axis manager,
		iN a noise model, representing the inverse noise covariance matrix,
		and iNd the result of applying the noise model to the detector time-ordered data."""
		iNd     = iNd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutPoly(obs.cuts, basis=self.basis, dev=self.dev)
		# Build our RHS
		obs_rhs = self.dev.np.zeros(pcut.ndof, self.dtype)
		pcut.backward(iNd, obs_rhs)
		obs_rhs = self.dev.get(obs_rhs)
		# Build our preconditioner. We have precomputed the basis.dot(basis.T) inverse for
		# all block lengths, so we just read out the appropriate one for each block.
		# TODO: Understand why the none-preconditioner suddenly works fine, while
		# the other two breaks convergence. Apparently this only happens when I include
		# 1/ivar in them, otherwise they don't hurt but also don't beat none. Very
		# strange.
		if self.prec == "leginv":
			obs_idiv = self.ibases[self.dev.get(pcut.lens)-1] / self.dev.get(iN.ivar[pcut.dets])[:,None,None]
		elif self.prec == "var":
			obs_idiv = 1/self.dev.get(iN.ivar[pcut.dets])
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

class SignalElMod(Signal):
	def __init__(self, comm, order=1, dev=None, name="elmod", ofmt="{name}", dtype=np.float32, outputs=["map"]):
		Signal.__init__(self, name, ofmt, outputs, ext="txt")
		self.comm  = comm
		self.dev   = dev or device.get_device()
		self.order = order
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
	def add_obs(self, id, obs, iN, iNd):
		iNd     = iNd.copy() # This copy can be avoided if build_obs is split into two parts
		pcut    = pmat.PmatCutFull(obs.cuts, dev=self.dev)
		pel     = pmat.PmatElMod(obs.boresight[1], order=self.order, dev=self.dev, dtype=self.dtype)
		# Build our RHS
		ndet    = obs.tod.shape[0]
		ndof    = ndet*self.order
		obs_rhs = self.dev.np.zeros((ndet,pel.order), self.dtype)
		pcut.clear(iNd)
		pel.backward(iNd, obs_rhs)
		obs_rhs = self.dev.get(obs_rhs)
		# Build our preconditioner. Fully diagonal for now
		obs_div = self.dev.np.ones((ndet,pel.order), self.dtype)
		iNd[:]   = 0
		pel.forward(iNd, obs_div)
		#pcut.clear(iNd) # unnecessary since white is diagonal
		iN.white(iNd)
		pcut.clear(iNd)
		pel.backward(iNd, obs_div)
		obs_idiv = 1/self.dev.get(obs_div) # back to the cpu
		self.data[id] = bunch.Bunch(pel=pel, ndet=ndet, i1=self.off, i2=self.off+ndof)
		self.off += ndof
		self.rhs.append(obs_rhs.reshape(-1))
		self.idiv.append(obs_idiv.reshape(-1))
		self.dev.garbage_collect()
	def prepare(self):
		"""Process the added observations, determining our degrees of freedom etc.
		Should be done before calling forward and backward."""
		if self.ready: return
		self.rhs = np.concatenate(self.rhs)  if len(self.rhs) > 0 else np.zeros(0, self.dtype)
		self.idiv= np.concatenate(self.idiv) if len(self.rhs) > 0 else np.zeros(0, self.dtype)
		self.dof = ArrayZipper(self.rhs.shape, dtype=self.dtype, comm=self.comm)
		self.ready = True
	def forward(self, id, gtod, gamp):
		if id not in self.data: return
		d = self.data[id]
		d.pel.forward(gtod, gamp[d.i1:d.i2].reshape((d.ndet,self.order)))
	def backward(self, id, gtod, gamp):
		if id not in self.data: return
		d = self.data[id]
		d.pel.backward(gtod, gamp[d.i1:d.i2].reshape((d.ndet,self.order)))
	def precon(self, amp):
		return amp*self.idiv
	def to_work  (self, x): return self.dev.np.array(x)
	def from_work(self, x): return self.dev.get(x)
	def owork(self): return self.dev.np.zeros(self.rhs.shape, self.rhs.dtype)
	# have [id][ndet,order] values. Best as just a text file with
	# one line per tod and columns of det-order?
	def write(self, prefix, m, tag="map", suffix="", force=False):
		if not force and tag not in self.outputs: return
		# Collect info on root node
		my_ids   = np.array([id for id in self.data])
		my_ndets = np.array([self.data[id].ndet for id in my_ids])
		ids      = utils.allgatherv(my_ids,   self.comm)
		ndets    = utils.allgatherv(my_ndets, self.comm)
		vals     = utils.allgatherv(m,        self.comm)
		offs     = utils.cumsum(ndets*self.order, endpoint=True)
		order    = np.argsort(my_ids)
		## Conversion from µK/rad to mK/degree
		#uconv    = 1e-3*utils.degree**np.arange(1, self.order+1)
		uconv = 1
		oname = self.ofmt.format(name=self.name)
		oname = "%s%s_%s%s.%s" % (prefix, oname, tag, suffix, self.ext)
		if self.comm.rank == 0:
			with open(oname, "w") as ofile:
				for i in order:
					id   = ids[i]
					amps = vals[offs[i]:offs[i+1]].reshape(ndets[i],self.order)
					amps*= uconv # unit conversion
					# Format line
					msg  = "%s " % id
					for damps in amps:
						msg += " %8.5f"*(len(damps)) % tuple(damps)
					ofile.write(msg + "\n")
		return oname
	def written(self, prefix):
		done  = True
		for tag in self.outputs:
			oname = self.ofmt.format(name=self.name)
			oname = "%s%s_%s.%s" % (prefix, oname, tag, self.ext)
			done  = done and os.path.exists(oname)
		return done
	# "map" means the main output for the signal here, in contrast with
	# e.g. rhs and ivar. It doesn't mean an actual sky map
	valid_outputs = ["map"]

# Actually, doing this as an actual Signal might be cleanest.
# Requries no changes to Mapmaker, make_map, etc.
# Slightly hacky since it would have no degrees of freedom though.
class SignalInfo(Signal):
	def __init__(self, comm, dev=None):
		Signal.__init__(self, name="info", ofmt="{name}", outputs=["info"], ext="hdf")
		self.comm  = comm
		self.dev   = dev if dev is not None else device.get_gevice()
		self.dof   = ArrayZipper(0, dtype=np.float32)
		self.rhs   = np.zeros(0, dtype=np.float32)
		self.info  = []
	def reset(self):
		self.info  = []
		self.ready = False
	def add_obs(self, id, obs, iN, iNd):
		srate = (len(obs.ctime)-1)/(obs.ctime[-1]-obs.ctime[0])
		# Get the scanning info. These are *after* the pointing
		# model has been applied
		def leftw(arr):
			v1, v2 = utils.minmax(arr)
			return v1, v2-v1
		def cenw(arr):
			v1, v2 = utils.minmax(arr)
			return (v1+v2)/2, v2-v1
		ctime,  dur   = leftw(obs.ctime)
		az,     waz   = cenw(obs.boresight[1])
		el,     wel   = cenw(obs.boresight[0])
		roll,   wroll = cenw(obs.boresight[2])
		# Restore overpoleness
		if obs.overpole:
			el   = np.pi-el
			az   = utils.rewind(az+np.pi)
			roll = utils.rewind(roll+np.pi)
		# Detector info
		dsens = (iN.ivar*srate)**-0.5
		# Center and radius per band
		self.info.append(bunch.Bunch(
			id=id, ctime=ctime, dur=dur, az=az, waz=waz, el=el, wel=wel, roll=roll, wroll=wroll,
			dsens=dsens, dets=obs.dets, detids=obs.detids, bands=obs.bands,
			offs=obs.point_offset))
	def owork(self): return np.zeros(self.rhs.shape, self.rhs.dtype)
	def prepare(self):
		# high-level gather. Hopefully not too high overhead
		info = self.comm.gather(self.info)
		if self.comm.rank == 0: self.info = sum(info,[])
		self.ready = True
	def written(self, prefix):
		return os.path.isfile(prefix + "info.hdf")
	def write_misc(self, prefix):
		# We only have a misc product, nothing we're actually solving for
		if "info" not in self.outputs: return
		if self.comm.rank != 0: return
		nobs   = len(self.info)
		# Loop through and find the full set of ids and dets
		ids, detids = [], []
		bands = {}
		for row in self.info:
			ids.append(row.id)
			detids.append(row.detids)
			for detid, band in zip(row.detids, row.bands):
				bands[detid] = band
		ids    = np.char.encode(np.array(ids))
		detids = np.unique(np.concatenate(detids))
		bands  = np.array([bands[detid] for detid in detids])
		ndet   = len(detids)
		# Sort dets by detid and obs by id
		order_obs = np.argsort(ids)
		order_det = np.argsort(detids)
		detids    = detids[order_det]
		bands     = bands [order_det]
		# Get the sensitivity per band
		ubands, order, edges = utils.find_equal_groups_fast(bands)
		nband     = len(ubands)
		nperband  = edges[1:]-edges[:-1]
		# Some of our info fits nicely into a simple per-obs table
		obstab = np.zeros(nobs, [("id",ids.dtype),("ctime","f"),("dur","f"),
			("az","f"),("waz","f"),("el","f"),("wel","f"),
			("roll","f"),("wroll","f"),("ndet","i"),("sens","%df"%nband)]).view(np.recarray)
		simple_fields = ["ctime","dur","az","waz","el","wel","roll","wroll"]
		deg_fields = ["az","waz","el","wel","roll","wroll"]
		for i, ri in enumerate(order_obs):
			row = self.info[ri]
			obstab[i].id   = row.id
			obstab[i].ndet = len(row.detids)
			for field in simple_fields:
				obstab[i][field] = row[field]
			# Get the band sensitvity
			inds  = utils.find(detids, row.detids)
			dsens = self.dev.get(row.dsens)
			for bi, band in enumerate(ubands):
				good = bands[inds]==band
				band_dsens = dsens[good]
				if band_dsens.size == 0: continue
				band_sens  = np.sum(band_dsens**-2)**-0.5
				obstab[i].sens[bi] = band_sens
		# to degrees
		for field in deg_fields:
			obstab[field] /= utils.degree
		# Build obs,dsens matrix. A problem with this matrix is that it is
		# quite sparse for multi-tube obsjoint or multi-wafer waferjoint, and can get
		# pretty big. It would be more useful to separate out the dense blocks, but
		# not easy to do this automatically. Basically a boolean svd.
		# Will leave it as is for now
		dsens = np.zeros((nobs,ndet), np.float32)
		for i, ri in enumerate(order_obs):
			row  = self.info[ri]
			inds = utils.find(detids, row.detids)
			dsens[i,inds] = self.dev.get(row.dsens)
		res = bunch.Bunch(obstab=obstab, detids=detids, bands=bands, ubands=ubands, dsens=dsens)
		bunch.write(prefix + "info.hdf", res)
		# Also write obstab to simple text file, since the info files are a bit tedious to
		# deal with for quick inspection
		idlen = max([len(row.id) for row in obstab])
		with open(prefix + "info.txt", "w") as ofile:
			msg = "#%*s %10s  %6s %8s %7s %7s %5s %8s %5s " % (idlen-1, "id", "ctime",
				"dur", "az", "waz", "el", "wel", "roll", "wroll")
			for bname in ubands:
				msg += " n%4s %7s" % (bname, "sens")
			ofile.write(msg + "\n")
			for row in obstab:
				msg = "%*s %10.0f  %6.1f %8.3f %7.3f %7.3f %5.3f %8.3f %5.3f " % (
						idlen, row.id.decode(), row.ctime, row.dur, row.az, row.waz, row.el, row.wel,
						row.roll, row.wroll,
				)
				for bname, nper, bsens in zip(ubands, nperband, row.sens):
					msg += " %5d %7.3f" % (nper, bsens)
				ofile.write(msg + "\n")

	valid_outputs = ["info"]

# Mapmaking function
config.default("taskdist", "semibrute", "Method used to assign tods to mpi tasks")
# TODO: Investigate best value of this. I had 100 before, but 0.1 has worked better in
# recent depth1 tests. Why did 100 seem necessary before, and isn't 0.1 very low, basically
# just linear gapfilling?
config.default("gapfill_tol", 0.1, "Clip values brighter than this times the RMS when gapfilling")
def make_map(mapmaker, loader, obsinfo, comm, joint=None, inds=None, prefix=None, dump=[], maxiter=500, maxerr=1e-7, prealloc=True, ignore="recover", cont=False, dets=None, detids=None):
	if prefix is None: prefix = ""
	# Skip if we're already done
	if cont and (os.path.exists(prefix + ".empty") or all([signal.written(prefix) for signal in mapmaker.signals])):
		L.print("Skipped %s: Already done" % prefix, level=2, color=colors.gray)
		return
	# Groups is a list of obsinfo entries that should be mapped jointly
	# (as a big super-tod with full noise correlations)
	if joint is None: joint = trivial_joint(obsinfo)
	# Inds is a list of indices into joint groups, giving which groups this mpi task
	# should care about
	if inds is None:
		# Use the first entry in groups as representative
		gfirst = np.array([g[0] for g in joint.groups])
		# FIXME: This doesn't know about the coordinate system we're using!
		# It assumes equatorial coordinates.
		taskdist = config.get("taskdist")
		if taskdist == "simple":
			dist = tiling.distribute_tods_simple(obsinfo[gfirst], comm.size)
		elif taskdist == "semibrute":
			dist = tiling.distribute_tods_semibrute(obsinfo[gfirst], comm.size)
		else:
			raise ValueError("Unrecognized task distribution method '%s'" % str(taskdist))
		inds = np.where(dist.owner == comm.rank)[0]
	# Set up exception types we will ignore
	if   ignore == "all":     etypes, load_catch = (Exception,), "all"
	elif ignore == "missing": etypes, load_catch = (utils.DataMissing,), "expected"
	elif ignore == "recover": etypes, load_catch = (utils.DataMissing, gutils.RecoverableError), "all"
	elif ignore == "none":    etypes, load_catch = (), "none"
	else: raise ValueError("Unrecognized error ignore setting '%s'" % str(ignore))
	# Accept either a list of integers or a single integer
	try: dump = list(dump)
	except TypeError: dump = [dump]
	dev = mapmaker.dev
	# Set up memory pools. Setting these up before-hand is
	# actually more memory-efficient, as long as our estimate is
	# good.
	if prealloc and len(inds) > 0:
		ntot_max = np.max(gutils.obs_group_size(obsinfo, joint.groups, inds=inds, sampranges=joint.sampranges))
		setup_buffers(dev, ntot_max)
	# Start map from scartch
	mapmaker.reset()
	# Add our observations
	for i, ind in enumerate(inds):
		name   = joint.names[ind]
		subids = obsinfo.id[joint.groups[ind]]
		t1     = time.time()
		try:
			data  = loader.load_multi(subids, samprange=joint.sampranges[ind], catch=load_catch, dets=dets, detids=detids)
			# I keep calculating this. It should be a standard member of data
			# Should probably promote data to a full class
			srate = (len(data.ctime)-1)/(data.ctime[-1]-data.ctime[0])
		except etypes as e:
			L.print("Skipped %d %s: %s" % (ind, name, str(e)), level=2, color=colors.red)
			continue
		if len(data.errors) > 0:
			# Partial skip
			L.print("Skipped parts %s" % str(data.errors[-1]), level=2, color=colors.red)
		# Do configurable autocuts that are indpendent of the loading method here.
		# Might want to wrap it into some higher-level load function.
		# Autocut and gapfill cost about the same as add_obs, so they definitely
		# aren't free!
		t2    = time.time()
		socal.autocut(data, dev=dev, id=name)
		# Remove extreme values in the cut areas. This is gentler than trying
		# and failing to gapfill with realistic noise
		gutils.gapfill_extreme(data.tod, data.cuts, dev=dev, tol=config.get("gapfill_tol"))
		# Autocalibration. Controlled by config:elmod_cal and config:cmod_cal
		socal.autocal(data, prefix=prefix + name.replace(":","_") + "_", dev=dev)
		t3    = time.time()
		try:
			mapmaker.add_obs(name, data, deslope=False)
		except etypes as e:
			L.print("Skipped %d %s: %s" % (ind, name, str(e)), level=2, color=colors.red)
			continue
		dev.garbage_collect()
		del data
		t4    = time.time()
		L.print("Processed %d %s in %6.3f. Read %6.3f Autocal %6.3f Add %6.3f" % (ind, name, t4-t1, t2-t1, t3-t2, t4-t3), level=2)

	nobs = comm.allreduce(len(mapmaker.data))
	if nobs == 0:
		L.print("No tods survived!", id=0, level=0, color=colors.red)
		# Mark as done by making an .empty-file. This is used by cont
		# Annoyingly this file will be hidden for the no-prefix case,
		# since it starts with a .
		if comm.rank == 0: open(prefix + ".empty", "a").close()
		return None

	mapmaker.prepare()
	# Write rhs and ivar
	for signal in mapmaker.signals:
		signal.write_misc(prefix)

	# Solve the equation system
	step = None
	for step in mapmaker.solve(maxiter=maxiter, maxerr=maxerr):
		# Dump if we're in dump-list, or a multiple of the last entry, but not if
		# we're on the final iteration anyway, since that would be redundant with
		# the final result
		will_dump = len(dump) > 0 and (step.i in dump or (dump[-1] != 0 and step.i % dump[-1] == 0)) and step.i != maxiter
		if will_dump:
			for signal, val in zip(mapmaker.signals, step.x):
				signal.write(prefix, val, suffix="%04d" % step.i)
		# Safest to print this *after* writing, so the user can safely abort
		# when they see the message
		L.print("CG %4d %15.7e  %6.3f s%s" % (step.i, step.err, step.t, " (write)" if will_dump else ""), id=0, level=1, color=colors.lgreen)
	# Write the final result, unless we didn't do a single CG step
	if step is not None:
		for signal, val in zip(mapmaker.signals, step.x):
			signal.write(prefix, val)

def make_maps_perobs(mapmaker, loader, obsinfo, comm, comm_per, joint=None, inds=None, prefix=None, dump=[], maxiter=500, maxerr=1e-7, prealloc=True, ignore="recover", cont=False, dets=None, detids=None):
	"""Like make_map, but makes one map per subobs. NB! The communicators in the mapmaker
	signals must be COMM_SELF for this to work."""
	if joint is None:
		joint = trivial_joint(obsinfo)
	if inds is None:
		inds = list(range(comm.rank, len(joint.groups), comm.size))
	if prefix is None:
		prefix = ""
	if prealloc:
		ntot_max = np.max(gutils.obs_group_size(obsinfo, joint.groups, inds=inds, sampranges=joint.sampranges))
		setup_buffers(mapmaker.dev, ntot_max)

	# Map indivdual tods
	for ind in inds:
		name    = joint.names[ind]
		subpre  = prefix + name.replace(":","_") + "_"
		L.print("Mapping %s" % name)
		make_map(mapmaker, loader, obsinfo, comm_per, prefix=subpre, dump=dump, joint=joint, inds=[ind], maxiter=maxiter, maxerr=maxerr, prealloc=False, ignore=ignore, cont=cont, dets=dets, detids=detids)

config.default("depth1_maxdur", 24, "Max duration in hours for depth-1 maps. Lower values use less memory to store maps. Longer than 24 hours would no longer be depth-1")
def make_maps_depth1(mapmaker, loader, obsinfo, comm, comm_per, joint=None, prefix=None, dump=[], maxiter=500, maxdur=None, maxerr=1e-7, fullinfo=None, prealloc=True, ignore="recover", cont=False, dets=None, detids=None):
	# Find scanning periods. We want to base this on the full
	# set of observations, so depth-1 maps cover consistent periods
	# even if 
	from pixell import bench
	if prefix   is None: prefix = ""
	if joint    is None: joint = trivial_joint(obsinfo)
	# FIXME: might want obs,+bad here, to make the depth-1 periods independent
	# of our cuts. But currently that causes a segfault in sqlite?
	if fullinfo is None: fullinfo = loader.query("obs")
	maxdur  = config.get("depth1_maxdur")*utils.hour
	periods = gutils.find_scan_periods(fullinfo, maxdur=maxdur)
	# Which period each group belongs to. So pinds is [ngroup]
	gfirst  = np.array([g[0] for g in joint.groups])
	gpids   = utils.find_range(periods, obsinfo.ctime[gfirst]+obsinfo.dur[gfirst]/2)
	# Get rid of groups that don't belong to a period. This shouldn't happen
	bad = gpids<0
	if np.any(bad):
		L.print("Warning: %d obs with no period! %s" % (np.sum(bad), ", ".join(obsinfo.id[gfirst[bad]])), color=colors.red, id=0)
	# Indices of the groups we will map
	inds  = np.where(~bad)[0]
	# Which period each of those groups belongs to
	gpids = gpids[inds]
	# Group the groups by period. group-group i will consist
	# of groups inds[order[edges[i]:edges[i+1]]]
	pids, order, edges = utils.find_equal_groups_fast(gpids)
	my_pinds = list(range(comm.rank, len(pids), comm.size))
	if prealloc:
		# Estimate the max memory a group-group needs, and preallocate
		ntot_max  = 0
		for pind in my_pinds:
			my_inds = inds[order[edges[pind]:edges[pind+1]]]
			ntot    = np.max(gutils.obs_group_size(obsinfo, joint.groups, inds=my_inds, sampranges=joint.sampranges))
			ntot_max= max(ntot, ntot_max)
		setup_buffers(mapmaker.dev, ntot_max)
	# Now loop over and map each of our group-groups
	for pind in my_pinds:
		pid     = pids[pind]
		my_inds = inds[order[edges[pind]:edges[pind+1]]]
		# Name after period
		name    = "%10.0f" % periods[pid][0]
		subpre  = prefix + name + "_"
		L.print("Mapping period %10.0f:%10.0f with %d obs" % (*periods[pid], len(my_inds)))
		make_map(mapmaker, loader, obsinfo, comm_per, joint=joint, inds=my_inds, prefix=subpre, dump=dump, maxiter=maxiter, maxerr=maxerr, prealloc=False, ignore=ignore, cont=cont, dets=dets, detids=detids)

# Mapmaking function
def dump_tod(loader, obsinfo, noise_model, comm, joint=None, inds=None, prefix=None,
		tdown=500, fdown=250, fdown_lowf=10, fmax_lowf=4, prealloc=True, dev=None, ignore="recover", dets=None, detids=None):
	if prefix is None: prefix = ""
	# Groups is a list of obsinfo entries that should be mapped jointly
	# (as a big super-tod with full noise correlations)
	if joint is None: joint = trivial_joint(obsinfo)
	# Inds is a list of indices into joint groups, giving which groups this mpi task
	# should care about
	if inds is None:
		# Use the first entry in groups as representative
		gfirst = np.array([g[0] for g in joint.groups])
		dist = tiling.distribute_tods_semibrute(obsinfo[gfirst], comm.size)
		inds = np.where(dist.owner == comm.rank)[0]
	# Set up exception types we will ignore
	if   ignore == "all":     etypes = (Exception,)
	elif ignore == "missing": etypes = (utils.DataMissing,)
	elif ignore == "recover": etypes = (utils.DataMissing, gutils.RecoverableError)
	elif ignore == "none":    etypes = ()
	else: raise ValueError("Unrecognized error ignore setting '%s'" % str(ignore))
	if dev is None: dev = device.get_device()
	# Set up memory pools. Setting these up before-hand is
	# actually more memory-efficient, as long as our estimate is
	# good.
	if prealloc and len(inds) > 0:
		ntot_max = np.max(gutils.obs_group_size(obsinfo, joint.groups, inds=inds, sampranges=joint.sampranges))
		setup_buffers(dev, ntot_max)
	# Process our observations
	for i, ind in enumerate(inds):
		name   = joint.names[ind]
		subids = obsinfo.id[joint.groups[ind]]
		subpre = prefix + name.replace(":","_") + "_"
		t1     = time.time()
		try:
			data = loader.load_multi(subids, samprange=joint.sampranges[ind], dets=dets, detids=detids)
		except etypes as e:
			L.print("Skipped %s: %s" % (name, str(e)), level=2, color=colors.red)
			continue
		if len(data.errors) > 0:
			# Partial skip
			L.print("Skipped parts %s" % str(data.errors[-1]), level=2, color=colors.red)
		t2    = time.time()

		# Output tod
		otod = dev.get(gutils.downgrade(data.tod, tdown))
		dt   = (data.ctime[-1]-data.ctime[0])/(len(data.ctime)-1)
		twcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[dt*tdown,1])
		enmap.write_map(subpre + "tod.fits", enmap.enmap(otod, twcs))

		# Output ps
		ft   = dev.lib.rfft(data.tod)
		nsamp= data.tod.shape[1]
		normexp = -1
		ft  *= nsamp**normexp

		ps   = dev.np.abs(ft)**2
		ops  = dev.get(gutils.downgrade(ps, fdown)/data.tod.shape[1])
		df   = 1/(dt*data.tod.shape[1])
		fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown,1])
		enmap.write_map(subpre + "ps.fits", enmap.enmap(ops, fwcs))

		# Output high-res ps for low freq
		ops  = dev.get(gutils.downgrade(ps, fdown_lowf)/data.tod.shape[1])
		nbin = utils.floor(fmax_lowf/(df*fdown_lowf))
		ops  = ops[:,:nbin]
		fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown_lowf,1])
		enmap.write_map(subpre + "ps_lowf.fits", enmap.enmap(ops, fwcs))
		del ps

		# Output freq corrmat. Uses same resolution as coarse ps
		ndet, nfreq = ft.shape
		nbin = utils.floor(fmax_lowf/(df*fdown))
		bft  = ft[:,:nbin*fdown].reshape(ndet,nbin,fdown)
		cbft = dev.np.conj(bft)
		fcov = dev.np.einsum("dbf,ebf->deb",cbft,bft).real
		v    = dev.np.einsum("ddb->db", fcov)**-0.5
		fcorr= fcov*v[:,None,:]*v[None,:,:]
		fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown,1])
		enmap.write_map(subpre + "corr_lowf.fits", enmap.enmap(fcorr, fwcs))
		del bft, cbft

		# Same, but high resolution, for a subset of detectors
		ndet, nfreq = ft.shape
		thin = 100
		nbin = utils.floor(fmax_lowf/(df*fdown_lowf))
		bft  = ft[:,:nbin*fdown_lowf].reshape(ndet,nbin,fdown_lowf)
		cbft = dev.np.conj(bft[::thin])
		fcov = dev.np.einsum("dbf,ebf->deb",cbft,bft).real
		v    = dev.np.sum(dev.np.abs(bft)**2,-1)**-0.5
		fcorr= fcov*v[::thin,None,:]*v[None,:,:]
		fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown_lowf,1])
		enmap.write_map(subpre + "corr_lowf2.fits", enmap.enmap(fcorr, fwcs))
		del bft, cbft

		## Build the noise model
		#ft2  = ft.copy()
		#iN   = noise_model.build_fourier(ft2, srate=1/dt, nsamp=data.tod.shape[1])
		#d2   = dev.np.arange(ndet)
		#d1   = d2[::thin]
		#finds= dev.np.arange(nbin*fdown_lowf)
		#fcov = iN.eval_cov(d1=d1, d2=d2, finds=finds)
		## Same averaging as we do with the data
		#fcov = dev.np.sum(fcov.reshape(fcov.shape[:2]+(nbin,fdown_lowf)),-1)
		## Need the diagonal for normalization
		#v    = iN.eval_var(d=d2, finds=finds)
		#v    = dev.np.sum(v.reshape(v.shape[:1]+(nbin,fdown_lowf)),-1)**-0.5
		#fcorr= fcov*v[::thin,None,:]*v[None,:,:]
		#fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown_lowf,1])
		#enmap.write_map(subpre + "corrmodel_lowf2.fits", enmap.enmap(fcorr, fwcs))

		## Choose a single nmat bin and see what happens there.
		## Let's try a bin that's wider than the number of detectors to be safe
		#bi = np.where(iN.bins[:,1]-iN.bins[:,0] > ndet//8)[0][0]
		#b1, b2 = iN.bins[bi]
		#debug = bunch.Bunch(bi=bi, bin=iN.bins[bi], freqs=iN.bins[bi]*df,
		#	V=iN.V[bi], iD=iN.iD[bi], iE=iN.iE[bi], dpre=ft2[:,b1:b2].copy())

		## Try to manually project out the dark modes
		#dark  = np.array(["DARK" in band for band in data.bands])
		#light = ~dark
		#iiN11 = iN.det_slice(light).inv()
		#d2    = ft2.copy(); d2[light] = 0
		#ft2[light] += iiN11.apply(iN.apply(d2, nofft=True)[light], nofft=True)

		#debug.dark  = dark
		#debug.dpost = ft2[:,b1:b2]
		#bunch.write(subpre + "debug.hdf", debug)

		## Output high-res ps for low freq
		#ps   = dev.np.abs(ft)**2
		#ops  = dev.get(gutils.downgrade(ps, fdown_lowf)/data.tod.shape[1])
		#nbin = utils.floor(fmax_lowf/(df*fdown_lowf))
		#ops  = ops[:,:nbin]
		#fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown_lowf,1])
		#enmap.write_map(subpre + "ps_deproj.fits", enmap.enmap(ops, fwcs))
		#del ps

		#nbin = utils.floor(fmax_lowf/(df*fdown_lowf))
		#bft  = ft[:,:nbin*fdown_lowf].reshape(ndet,nbin,fdown_lowf)
		#cbft = dev.np.conj(bft[::thin])
		#fcov = dev.np.einsum("dbf,ebf->deb",cbft,bft).real
		#v    = dev.np.sum(dev.np.abs(bft)**2,-1)**-0.5
		#fcorr= fcov*v[::thin,None,:]*v[None,:,:]
		#fwcs = wcsutils.explicit(crval=[0,0], crpix=[1,1], cdelt=[df*fdown_lowf,1])
		#enmap.write_map(subpre + "corr_deproj.fits", enmap.enmap(fcorr, fwcs))

		dev.garbage_collect()
		del data
		t3    = time.time()
		L.print("Processed %s in %6.3f. Read %6.3f Add %6.3f" % (name, t3-t1, t2-t1, t3-t2), level=2)

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

def trivial_joint(obsids):
	"""Make a group-info corresponding to a no grouping"""
	groups=[[i] for i in range(len(obsids))]
	return bunch.Bunch(groups=groups, names=obsids.id, sampranges=[None for i in range(len(obsids))])

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
		self.ndof    = 0
		self.bins    = []
	def add(self, zipper):
		self.zippers.append(zipper)
		self.bins.append([self.ndof, self.ndof+zipper.ndof])
		self.ndof += zipper.ndof
	def zip(self, objs):
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

def extend_prefix(prefix, extra):
	if len(prefix) == 0 or prefix.endswith("/"):
		return prefix + extra
	else:
		return prefix + "_" + extra

def check_outputs(outputs, valid_outputs, name):
	if   outputs is None: return []
	elif outputs == "all": return valid_outputs
	else:
		res = []
		for output in outputs:
			if output not in valid_outputs:
				raise ValueError("'%s' is not a valid output for %s" % (str(output), name))
			res.append(output)
		return res
