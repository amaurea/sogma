# Exceptions
# ----------
#
# Want to be able to catch categories of exceptions, at least:
#  all: Catch everything. Usually not useful
#  expected: Catch ones that would occur during normal operation,
#   e.g. some data being missing
#  none: Don't catch any exceptions
#
# The hard one is the "expected" category. Each loader will know what is expected
# or not. Naively it would be enough to catch all expected exceptions and reraise as
# an Expected exception for the caller to deal with as they wish. But PostLoader
# loops over multiple subids, and the handling will differ for these. In catch="none",
# we want to show all exceptions as soon as they occur, since we're probably debugging.
# That means any exception should propagate right away, like above. But in catch="expected",
# it's not an error for for some of the subids to fail, so here one should catch them
# internally, record information from them, but not reraise. But if no data makes it through,
# then that should result in an (expected) exception.

# Should classify exceptions as close to where they happen as possible, to let us
# distinguish between expected ones and bugs. Python preserves the original exception information
# if I do something like
#  try:
#    blah blah
#  except ValueError:
#    raise DataMissing
# so safe to catch things early

# What should my exception hierarchy look like?
# class ExpectedError(Exception): pass
# class LoadError(ExpectedError): pass
# class DataMissing(LoadError): pass
# class NothingLeft(LoadError): pass
# Don't need a hierarchy for unexpected errors, since those would only be caught
# with catch="none"

# Benchmarking
# ------------
# How should benchmarking be handled with layers of loading? And what about
# obs.timing? Should it just include the top-level info, or should it
# inherit from subloaders somehow? Let's deal with standard bench first.
# Let's add a short suffix identifying the class and function. Full names
# would be cumbersome I think, e.g.
#  SoFastLoader_load_meta
# It wouldn't be too bad in a file

# obs.subids
# ----------


# But need to be able to
# 1. map back to original subids
# 2. set up buffers. This needs
#    ndet_ndown = np.max(ndet*ndown)
#    nsub_nsamp = np.max(ginfo.nsub*ginfo.nsamp)
#    nsub_ndown = np.max(ginfo.nsub*ndown)
#    ndet_nfdown= np.max(ndet*nfdown)
#    nsub_nf    = np.max(ginfo.nsub*nf)
# Currently query just returns an obsinfo, which is just a numpy table.
# Should it return a bunch containing obsinfo + this info? I so, all
# loaders should return such a bunch. Probably best solution, since
# this info depends on the query, and much of the info needs to be built
# in this function anyway
#
# How will load() work? Given an id to load. Must be able to recover list
# of subids. Would be easy if we could store the mapping internally, but
# the mapping depends on what was passed to query, and it must. After all,
# the id obs_ctime_lati1_111 could include all wafers and bands, or just
# ws0 f090, depending on the query.
#
# Solution 1: loader.query() returns an object that implements load().
#  Remove load() from Loader's interface. Must call query first to get
#  the actual loader first. If so, should Loader be renamed something else?
# Solution 2: Store mapping from last query internally. No!
# Solution 3: load() takes an optional argument with the mapping, which is
#  actually mandatory for PostLoader. Yuck.
# Solution 4: Take the query in loader.__init__. But this means we have to
#  reconstruct the object for each query. Usually there's only one, but
#  don't want to restrict to that
# Solution 5: PostLoader specifically takes a query in __init__ because the
#  definition of its ids depends on it. Perhaps the difficulty here is that
#  query plays two different roles. One is to select subids, the other is
#  to define metaids. The problem with this approach is that PostLoader's
#  user interface becomes very different from the other loaders, so automated
#  construction gets messed up.
# Solution 6 = 1+3:
#  loader.query() → obsmap,
#  loader.load(id, obsmap=None)
#  obsmap.load(id) = obsmap.loader.load(id, obsmap=self)
#  That is, we commit to the "obsmap" object which query returns
#  being a thin wrapper, with loader doing the heavy work. Obsmap
#  may or may not actually be needed by loader.load. In practice it
#  will only be used by PostLoader.
#
# Let's go with #6. Can I get away with a single obsmap class?
# Can, if we use **kwargs to allow for extra information. A bunch
# would do, really, except we want it to have the .load() method
#
# For PostLoader, Obsmap would contain
#  .obsinfo, .loader, .submap (for loader), .demod, .comps, .down, .bands, .nullbands,
#  .sampranges[ng,{i1,i2}], .groups[ng][]
#
# For SofastLoader, Obsmap would contain
#  .obsinfo, .loader
#
# How to handle preallocation? Can either return the information needed, or could
# provide a function that takes care of the loader part of it. The problem is that
# the buffer names and sizes depend on how deep in the chain one is. E.g.
#
#  sofast by itself: tod[ndet,nsamp]
#  post+sofast: itod[ndet,nsamp] → tod[nfull,ndown]
#
# Could pass in names of buffers to use. But where? Needed both for prealloc and .load()
# Could have them as optional arguments to load. Would be cleaner to have them in __init__,
# but that doesn't really work with the way the loaders are allocated. But we could provide
# a set_buffers function that overrides whatever was set in __init__. Probably best approach.
# Will be hard to write if I don't make strong assumptions about the types of buffers needed.
# Relevant ones:
#  tod buffer: Would be tod for post and itod for inner
#
# What if each loader declares which buffers it wants to use? Then loaders that use it
# can ask them to rename ones it already want to use. E.g.
#  sofast: tod, ft
#  post: tod, [dtod, wtod, ft]
# post sees that tod overlaps with sofast's use of it, so it asks sofast to use itod instead.
# So loader would have a .pool_map member that it default-initializes to something sensible,
# and a .avoid_pools(names) method that updates .pool_map to avoid the names given there, e.g.
# by appending to the name
#
# Obsmap is a bad name. Obsinfo is better, but what should the table be called then?
# obstab? That's ok I guess.

class Loader:
	def __init__(self, dev=None, pool_map={}, catch="expected", dtype=np.float32):
		self.dev      = dev or device.get_device()
		self.pool_map = pool_map
		self.catch    = catch
		self.etypes   = catch_types[catch]
		self.dtype    = dtype
	def query(self, query=None):
		raise NotImplementedError
	def probe(self, linfo, id, dets=None, detids=None)
		raise NotImplementedError
	def load(self, linfo, id, dets=None, detids=None, samprange=None, pinfo=None):
		raise NotImplementedError
	def group_obs(self, obsinfo, mode="obs"):
		raise NotImplementedError
	def prealloc(self, linfo):
		raise NotImplementedError
	def avoid_pools(self, names):
		for typname, bufname in self.pool_map:
			if bufname in names:
				self.pool_map[typname] = bufname + '*'
	def pool(self, name): return self.dev.pools[self.pool_map[name]]

class LoadInfo:
	"""Class representing a set of observations to load, and metadata needed to load it.
	Contains at least the .obsinfo member a numpy table of the observations and their properties,
	and provides the load() meathod for reading in an observation"""
	def __init__(self, loader, obsinfo, **kwargs):
		self.loader  = loader
		self.obsinfo = obsinfo
		self.__dict__.update(kwargs)
	def probe(self, id, dets=None, detids=None):
		return self.loader.probe(self, id, dets=dets, detids=detids)
	def load(self, id, dets=None, detids=None, samprange=None):
		return self.loader.load(self, id, dets=dets, detids=detids, samprange=samprange)
	def prealloc(self):
		return self.loader.prealloc(self)

class ProbeInfo:
	"""Class representing the result of probing an observation, which means doing a
	relatively light-weight partial read in order to determine the actually readable number
	of detectors, the number of samples and the absolute sample timing. May also include other
	information as needed by the individual loaders"""
	def __init__(self, ndet, nsamp, t1, srate, **kwargs):
		self.ndet, self.nsamp, self.t1, self.srate = ndet, nsamp, t1, srate
		self.__dict__.update(kwargs)

catch_types = {
	"all":      (Exception,),
	"expected": (Expected,),
	"none":     (),
}

config.default("demod", "auto", "Whether to demodulate. yes, no or auto. yes always tries to demodulate, causing the load to fail if it can't. no never demodulates. auto demodulates if the hwp is present, and otherwise does nothing")
config.default("comps", "TQU", "Which components to construct when demodulating. Can be TQU or QU")
config.default("down", 0.0, "Downsampling factor. Set to 0 to disable downsampling")

class PostLoader(Loader):
	def __init__(self, loader, group="obs", split=None, tsplit=None, order="band", dev=None, dtype=np.float32, catch="expected", pool_map={"tod":"tod", "ft":"ft", "dtod":"dtod", "ftod":"ftod"}):
		super().__init__(dev=dev or loader.dev, pool_map=pool_map, dtype=dtype, catch=catch)
		# Tell the underlying loader which pools we reserve. This is not the same as
		# which pools we use; it only covers the pools that need to be preserved while the sub-loader works
		loader.avoid_pools([self.pool_map["tod"]])
		self.loader     = loader
		# These define how our  higher-level ids will map to actual subids
		self.group      = group
		self.split      = split
		self.tsplit     = tsplit
		self.order      = order
	def query(self, query=None):
		# Forward query to underlying loader, then transform.
		# linfo:[obsinfo:[id,ndet,nsamp,ctime,dur,bas,waz,bel,wel,roll,fhwp,r,sweep], ...]
		linfo   = self.loader.query(query)
		# Settings for demodulation and downsampling
		fhwp  = np.mean(np.abs(linfo.obsinfo.fhwp))
		demod = check_demod(config.get("demod"), has_hwp=fhwp>0)
		comps = config.get("comps")
		down  = config.get("down") or 1
		# Apply grouping
		# joint:[names[ng],groups[ng][nmemb],bands[nband],nullbands[nnull],joint:bool,sampranges=None]
		joint   = self.loader.group_obs(linfo, mode=self.group)
		obsinfo_merged = merge_obsinfo(linfo.obsinfo, joint.groups, joint.names)
		# Should find a better name for this. It splits in time both based on duration and size,
		# and also calculates the total downsampling factor (including demodulation) and detector duplication factor
		tinfo   = time_split(obsinfo_merged, demod=demod, down=down, comps=comps, maxsize=self.split, maxdur=self.tsplit)
		nsamps  = nsplit2bsize(tinfo.nsplits, obsinfo.nsamp, factors=self.dev.lib.fft_factors, direction="above", mul=self.dev.lib.bsize)
		# Perform the split
		obsinfo, ind_map, sampranges = split_obsinfo(obsinfo_merged, nsamps)
		# Update nsamp and ndet to take into account demodulation etc.
		# We do this because the obsinfo we expose refers to the fully processed tods, not the underlying raw ones
		obsinfo.nsamp = utils.nint(obsinfo.nsamp/tinfo.downfact)
		obsinfo.ndet *= obsinfo.detfact
		# Build the final it→id map
		omap = {id:oi for oi,id in enumerate(obsinfo.id)}
		imap = [groups[ind] for ind in ind_map]
		# sampranges refers to the raw subobs ranges we want. It has the same length as obsinfo,
		# but refers to raw samples, not demodulated ones
		return LoadInfo(self, obsinfo, sublinfo=linfo, imap=imap, omap=omap, sampranges=sampranges,
			demod=demod, comps=comps, down=down, downfact=tinfo.downfact, detfact=tinfo.detfact)
	def probe(self, linfo, id, dets=None, detids=None):
		"""The job of this function is to find how our subids' samples align, so we
		can read in consistently aligned samples. It also estimates the resulting ndet and nsamp,
		but these can deviate from what actually gets read due to internal cuts in load()
		(e.g. from sofast.calibrate())"""
		# Expand id subids
		oind     = linfo.omap[id]         # index in our output obsinfo
		sinds    = linfo.imap[oind]       # corresponding sub-obsinfo indices
		subinfo  = linfo.sublinfo.obsinfo[sinds] # the sub-obsinfo entries corresponding to id
		# Will probably want subids with the same band together due to mapmaking
		# contiguity requirements
		if   self.order == "raw":  order = np.arange(len(obsinfo))
		elif self.order == "band": order = np.argsort(subinfo.band)
		else: raise ValueError("Unrecognized subid ordering '%s'" % str(self.order))
		subinfo = subinfo[order]
		# Probe the individual subobs
		pinfos     = []
		exceptions = []
		eids       = []
		for si, subid in enumerate(subinfo.id):
			try:
				with bench.mark("PostLoader subprobe"):
					pinfo = info.sublinfo.probe(subid, dets=dets, detids=detids)
				pinfos.append(pinfo)
			except self.catch_list as e:
				exceptions.append(e)
				eids.append(subid)
		if len(pinfos) == 0:
			# FIXME: DataMissing should inherit from Expected
			raise utils.DataMissing(format_multi_exception(exceptions, eids))
		# Find the overlapping and aligned sample ranges for the given sampranges, if
		# possible. FIXME handle exception here
		subranges = pinfo_overlap(pinfos)
		subranges = srange_chain(subranges, linfo.sampranges[oind])
		# And finally return our result, which we will use in load
		ndet  = sum([pinfo.ndet for pinfo in pinfos])*linfo.detfact
		nsamp, t1, srate = calc_post_samps(pinfos, subranges)
		return ProbeInfo(ndet, nsamp, t1, srate, oind=oind, sinds=sinds, subinfo=subinfo, subranges=subranges, pinfos=pinfos)
	def load(self, linfo, id, dets=None, detids=None, samprange=None):
		if pinfo is None: pinfo = linfo.probe(id, dets=dets, detids=detids)
		subranges = pinfo.subranges
		if samprange is not None:
			subranges = srange_chain(subranges, srange_expand(samprange, linfo.downfact[pinfo.oind]))
		# Set up total obs
		otot = bunch.Bunch(ctime=None, boresight=None, hwp=None, tod=None, subids=[], errors=[], cuts=[], fill=[])
		append_fields = [("dets",0),("detids",0),("detpix",0),("bands",0),("point_offset",0),
			("polangle",0),("response",1)]
		for field, axis in append_fields: otot[field] = []
		# Read the individual observations into our structures
		dcum = 0
		for si, subid in enumerate(pinfo.subinfo.id):
			try:
				with bench.mark("PostLoader subload"):
					subobs = linfo.sublinfo.load(subid, dets=dets, detids=detids, samprange=subranges[si], pinfo=pinfo.pinfos[si])
				if linfo.demod:
					# Handles both demodulation and downsampling in one go
					with bench.mark("PostLoader demodulate"):
						subobs = socommon.demodulate(subobs, comps=linfo.comps, frel=1/linfo.down, dev=self.dev)
				elif linfo.down != 1:
					# Handles plain downsampling
					with bench.mark("PostLoader downsample"):
						subobs = socommon.downsample(subobs, down=linfo.down, dev=self.dev)
				subobs.subids = [subid]
				subobs.errors = []
			except catch_list as e:
				exceptions.append(e)
				eids      .append(subid)
				continue
			# Initial obs setup
			if obs.tod is None:
				obs.ctime     = subobs.ctime
				obs.boresight = subobs.boresight
				obs.hwp       = subobs.hwp
				obs.site      = subobs.site
				obs.bore_ref  = subobs.bore_ref
				obs.sampoff   = subobs.sampoff
				# reminder: self.pool("tod") = self.dev.pools[self.pool_map["tod"]]
				obs.tod       = self.pool("tod").zeros((ndet,len(subobs.ctime)), subobs.tod.dtype)
			# Handle the simple append cases
			for field, axis in append_fields:
				obs[field].append(subobs[field])
			obs.cuts.append(subobs.cuts)
			obs.fill.append(subobs.fill)
			obs.subids += subobs.subids
			obs.errors += subobs.errors
			# Copy tod over to the right part of the output buffer
			obs.tod[dcum:dcum+len(subobs.tod)] = subobs.tod
			dcum += len(subobs.tod)
		# Were we left with anything at all?
		if dcum == 0:
			raise utils.DataMissing(format_multi_exception(exceptions, eids))
		# Concatenate the work-lists into the final arrays
		for field, axis in append_fields:
			obs[field] = np.concatenate(obs[field],axis) if obs[field][0] is not None else None
		obs.cuts = socut.Simplecut.detcat(obs.cuts)
		obs.fill = socut.Simplecut.detcat(obs.fill)
		# Trim tod in case we lost some detectors
		obs.tod = obs.tod[:dcum]
		# Non-fatal errors
		if len(exceptions) > 0:
			obs.errors.append(utils.DataMissing(format_multi_exception(exceptions, eids)))
		return obs
	def prealloc(self, linfo):
		def s(ndet, nsamp): return utils.ceil(np.max(ndet+(nsamp+2))) # fourier-safe size
		# Max size of our output tod
		obsinfo = linfo.obsinfo
		nout = s(obsinfo.ndet, obsinfo.nsamp)
		# max size of full processed subobs. Differs from nout by having fewer dets
		nper = s(obsinfo.ndet/linfo.detfact, obsinfo.nsamp)
		# max size of raw output from subloader
		nsub = s(obsinfo.ndet/linfo.detfact, obsinfo.nsamp*linfo.downfact)
		self.pool("tod").empty(nout, dtype=self.dtype)
		if linfo.demod or linfo.down:
			self.pool("wtod").empty(nsub, dtype=self.dtype)
			self.pool("dtod").empty(nper, dtype=self.dtype)
			self.pool("ft")  .empty(nper, dtype=self.dtype)

# Will put these where they belong later

class SoFastLoader(Loader):
	def __init__(self, context_or_config_or_name, dev=None, dtype=np.float32, catch="expected", pool_map={"tod":"tod", "ft":"ft"}):
		super().__init__(dev=dev, pool_map=pool_map, dtype=dtype, catch_catch)
		self.context   = socommon.get_expanded_context(context_or_config_or_name)
		self.obsdb     = sqlite.open(self.context["obsdb"])
		self.predb     = sqlite.open(socommon.cmeta_lookup(self.context, "preprocess"))
		self.fast_meta = FastMeta(self.context)
		# Precompute the set of valid tags. Sadly this requires a scan through the whole
		# database, but it's not that slow so far
		self.tags      = socommon.get_tags(self.obsdb.conn)
	def query(self, query=None):
		res_db, pycode, slices = socommon.eval_query(self.obsdb.conn, query, tags=self.tags, predb=self.predb)
		obsinfo = socommon.finish_query(res_db, pycode, slices)
		omap    = {id:oi for oi,id in enumerate(obsinfo.id)}
		return LoadInfo(self, obsinfo, omap=omap)
	def probe(self, linfo, id, dets=None, detids=None):
		# TODO: Think about exception classification and propagation here
		with bench.mark("SoFastLoader load_meta"):
			meta = self.fast_meta.read(id, dets=dets, detids=detids)
			if meta.aman.dets.count == 0:
				raise utils.DataMissing("no detectors left after meta: raw %d meta 0" % meta.ndet_full)
		# Find which time-range we cover. Natively sample 0:obsinfo.nsamp cover
		# time obsinfo.ctime:obsinfo.ctime+obsinfo.dur, but we only use the range
		# aman.samps.offset:aman.samps.offset+aman.samps.nsamp of this
		row   = linfo.obsinfo[ind]
		srate = (row.nsamp-1)/row.dur
		t1    = row.ctime + meta.aman.samps.offset/srate
		nsamp = meta.aman.samps.count
		return ProbeInfo(meta.aman.dets.count, nsamp, t1, srate, meta=meta)
	def load(self, linfo, id, dets=None, detids=None, samprange=None, pinfo=None):
		if pinfo is None: pinfo = linfo.probe(id, dets=dets, detids=detids)
		# TODO: Exceptions
		meta = pinfo.meta
		# Restrict to target sample range
		if samprange is not None:
			off = meta.aman.samps.offset
			meta.aman.restrict("samps", slice(samprange[0]+off,samprange[1]+off), in_place=True)
		# Load the raw data
		with bench.mark("SoFastLoader fast_data"):
			data = fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
		# Calibrate the data
		with bench.mark("SoFastLoader calibrate"):
			obs = calibrate(data, meta, mul=self.dev.lib.bsize, dev=self.dev, dtype=self.dtype)
		obs.errors = []
		# Record what data we covered in the obs. Useful for logging
		obs.subids = [srange_suffix(id, samprange)]
		return obs
	def prealloc(self, linfo):
		def s(ndet, nsamp): return utils.ceil(np.max(ndet+(nsamp+2))) # fourier-safe size
		# Max size of our output tod
		obsinfo = linfo.obsinfo
		nout = s(obsinfo.ndet, obsinfo.nsamp)
		self.pool("tod").empty(nout, dtype=self.dtype)
		self.pool("ft") .empty(nout, dtype=self.dtype)

class SimpleLoader(Loader):
	def __init__(self, infofile, dev=None, dtype=np.float32, catch="expected", pool_map={"tod":"tod"}):
		"""context is really just a list of tods and meta here"""
		super().__init__(dev=dev, pool_map=pool_map, dtype=dtype, catch_catch)
		self.obsinfo = read_obsinfo(infofile)
		self.omap    = {id:i for i,id in enumerate(self.obsinfo.id)}
	def query(self, query=None, sweeps=False):
		# No actual querying supported for now
		return LoadInfo(self, self.obsinfo, omap=self.omap)
	def probe(self, linfo, id, dets=None, detids=None):
		# No det-slicing yet, but easy to add
		ind = linfo.omap[id]
		row = linfo.obsinfo[ind]
		return ProbeInfo(row.ndet, row.nsamp, row.ctime, (row.nsamp-1)/row.dur, ind=ind)
	def load(self, linfo, id, dets=None, detids=None, samprange=None):
		if pinfo is None: pinfo = linfo.probe(id, dets=dets, detids=detids)
		with bench.mark("SimpleLoader read"):
			obs = read_tod(self.obsinfo[ind].path, mul=self.dev.lib.bsize)
		with bench.mark("SimpleLoader tod2dev"):
			obs.tod = self.pool("tod").array(obs.tod)
		obs.subids = [srange_suffix(id, samprange)]
		return obs
	def prealloc(self, linfo):
		def s(ndet, nsamp): return utils.ceil(np.max(ndet+(nsamp+2))) # fourier-safe size
		# Max size of our output tod
		obsinfo = linfo.obsinfo
		nout = s(obsinfo.ndet, obsinfo.nsamp)
		self.pool("tod").empty(nout, dtype=self.dtype)

def srange_suffix(id, srange):
	if srange is None: return id
	else: return id + "," + "%d:%d" % list(srange)

def srange_trunc(srange, dev):
	srange = np.array(srange)
	srange[...,1] = srange[...,0] + dev.goodlen(srange[...,1]-srange[...,0])
	return srange

def srange_expand(srange, factor=1):
	return utils.nint(srange*factor)

def srange_chain(srange1, srange2):
	"""If srange2 is a sub-srange to srange1, what absolute sample range does it actually cover?"""
	res = srange1[...,0,None] + srange2
	res[...,1] = np.minimum(res[...,1], srange1[...,1])
	return res

def merge_obsinfo(obsinfo, groups, names):
	"""Merge obsinfo into groups with the given names. Returns new obsinfo"""
	first = np.array([group[0] for group in groups], dtype=int)
	oinfo = obsinfo[first].copy()
	# Override name
	oinfo.id = names
	for gi, group in enumerate(groups):
		# Ndet accumulates
		oinfo.ndet[gi] = np.sum(obsinfo.ndet[group])
		# Band concats
		oinfo.band[gi] = "+".join(obsinfo.band[group])
	return oinfo

def split_obsinfo(obsinfo, nsamps, fmt=":split%d"):
	"""Split each obsinfo[i] into chunks of at most nsamps[i] samples, returning a new obsinfo
	and a sampranges map"""
	nper  = (obsinfo.nsamp+nsamps-1)//nsamps
	oinfo = np.repeat(obsinfo, nper)
	# Sampranges is tedious to do vectorized, so just do it in python
	sampranges = np.zeros((len(oinfo),2),int)
	ind_map    = np.zeros(len(oinfo),int)
	oi = 0
	for ii, irow in enumerate(obsinfo):
		for si in range(nper):
			i1 = si*nsamps[ii]
			i2 = min(i1+nsamps[ii], irow.nsamp)
			sampranges[si] = (i1, i2)
			oinfo.nsamp[oi] = i2-i1
			oinfo.dur[oi]  *= oinfo.nsamp[oi]/row.nsamp
			oinfo.id[oi]   += fmt % si
			ind_map[oi]     = ii
	return oinfo, ind_map, sampranges

def time_split(obsinfo, demod=False, down=None, comps="TQU", maxsize=None, maxdur=None, frel=1):
	# Find our total downsampling factor
	downfact = np.full(len(obsinfo,1))
	detfact  = 1
	if demod:
		ifmax     = obsinfo.dur/obsinfo.nsamp/2
		ofmax     = frel*obsinfo.fhwp
		downfact *= ifmax/ofmax
		detfact  *= len(comps)
	if down is not None:
		downfact *= down
	# Split by total size
	if maxsize is not None:
		# calculate the total size of each group
		sizes = obsinfo.ndet * detfact * obsinfo.nsamp / downfact
		# number of time splits for each
		nsplits = np.maximum(nsplits, utils.floor(sizes/maxsize)+1)
	# Split by duration
	if maxdur is not None:
		nsplits = np.maximum(nsplits, utils.floor(obsinfo.dur/maxdur)+1)
	return bunch.Bunch(nsplits=nsplits, downfact=downfact, detfact=detfact)

def nsplit2bsize(nsplit, nsamp, factors=[2,3,5,7], mul=32):
	return fft.fft_len(utils.ceil(obsinfo.nsamp/nsplits/mul), factors=factors, direction="above")*mul

def pinfo_overlap(pinfos, tol=0.1):
	"""Return overlapping sample ranges for the given list of ProbeInfos"""
	nsamps, t1s, srates = [np.array([pinfo[key] for pinfo in pinfos]) for key in ["nsamp", "t1", "srate"]]
	t1  = np.max(t1s)
	t2  = np.min(t1s+nsamps/srates)
	if t2 <= t1: raise ValueError("no overlap")
	nsamp = utils.nint((t2-t1)*srates[0])
	# These are the sample ranges we're interested in, provided
	# we pass some checks later
	i1s   = utils.nint((t1-t1s)*srates)
	i2s   = i1s+nsamp
	# Check that we are properly aligned
	misalign1 = np.abs((t1s+i1s/srates-t1)*srates)
	misalign2 = np.abs((t1s+i2s/srates-t2)*srates)
	if np.any(misalign1 > tol) or np.any(misalign2 > tol):
		raise ValueError("incompatible timestamps")
	sranges = np.concatenate([i1s[...,None],i2s[...,None]],-1)
	return sranges

def nice_len(n, factors=[2,3,5,7], mul=32):
	return fft.fft_len(utils.floor(n/mul), factors=factors, direction="below")*mul

def check_demod(demod="auto", has_hwp=False):
	if demod not in ["auto", "yes", "no"]:
		raise ValueError("demod must be 'auto', 'yes' or 'no', but got '%s'" % str(demod))
	if demod == "auto": return has_hwp
	elif demod == "no": return False
	elif has_hwp: return True
	else: raise ValueError("Asked to demodulate, but no hwp present")
