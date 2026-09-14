import numpy as np
from pixell import utils, bunch, bench, config, fft
from .. import errors, socut
from . import socommon
from .socommon import srange_chain, srange_expand

config.default("demod", "auto", "Whether to demodulate. yes, no or auto. yes always tries to demodulate, causing the load to fail if it can't. no never demodulates. auto demodulates if the hwp is present, and otherwise does nothing")
config.default("comps", "TQU", "Which components to construct when demodulating. Can be TQU or QU")
config.default("down", 1.0, "Downsampling factor. Set to 0 or 1 to disable downsampling")

class PostLoader(socommon.Loader):
	"""Uses the memory pools tod, ft, wtod, dtod. The last 3 only if demod/down. Pass pool_map
	to change which pools are used"""
	def __init__(self, loader, group="obs", split=None, tsplit=None, order="band", dev=None,
			dtype=None, fast_fail=False, pool_map={}):
		super().__init__(dev=dev or loader.dev, pool_map=pool_map, dtype=dtype or loader.dtype)
		# Tell the underlying loader which pools we reserve. This is not the same as
		# which pools we use; it only covers the pools that need to be preserved while the sub-loader works
		self.conflicts  = ["tod"]
		loader.avoid_pools(utils.vmap(self.pool_map, self.conflicts))
		self.loader     = loader
		self.fast_fail  = fast_fail
		if fast_fail: self.catch = ()
		else:         self.catch = (errors.LoadError,)
		# These define how our  higher-level ids will map to actual subids
		self.group      = group
		self.split      = split
		self.tsplit     = tsplit
		self.order      = order
	def avoid_pools(self, names):
		# Overridden to allow for chaining
		forward = list(names)
		for typname, bufname in self.pool_map:
			if bufname in names:
				newname = bufname + '*'
				self.pool_map[typname] = newname
				if typename in self.conflicts:
					forward.append(newname)
		self.loader.avoid(forward)
	def query(self, query=None, dets=None, detids=None):
		# Forward query to underlying loader, then transform.
		# linfo:[obsinfo:[id,ndet,nsamp,ctime,dur,bas,waz,bel,wel,roll,fhwp,r,sweep], ...]
		linfo   = self.loader.query(query, dets=dets, detids=detids)
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
		post    = prepare_post(obsinfo_merged, demod=demod, down=down, comps=comps, maxsize=self.split, maxdur=self.tsplit)
		nsamps  = nsplit2bsize(post.nsplits, obsinfo_merged.nsamp, factors=self.dev.lib.fft_factors, mul=self.dev.lib.bsize)
		# Perform the split
		obsinfo, ind_map, sampranges = split_obsinfo(obsinfo_merged, nsamps)
		# Update nsamp and ndet to take into account demodulation etc.
		# We do this because the obsinfo we expose refers to the fully processed tods, not the underlying raw ones
		obsinfo.nsamp = utils.nint(obsinfo.nsamp/post.downfact[ind_map])
		obsinfo.ndet *= post.detfact
		# Make our sub-loader aware of our samprange plans. Used only for
		# preallocation
		linfo.maxnsamp = np.max(sampranges[:,1]-sampranges[:,0])
		# Build the final it→id map
		imap = [joint.groups[ind] for ind in ind_map]
		# sampranges refers to the raw subobs ranges we want. It has the same length as obsinfo,
		# but refers to raw samples, not demodulated ones
		return PostLoadInfo(self, obsinfo, sublinfo=linfo, imap=imap,
			sampranges=sampranges, demod=demod, comps=comps, down=down,
			downfact=post.downfact[ind_map], detfact=post.detfact)
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
					pinfo = linfo.sublinfo.probe(subid, dets=dets, detids=detids)
				pinfos.append(pinfo)
			except self.catch as e:
				exceptions.append(e)
				eids.append(subid)
		if len(pinfos) == 0:
			raise errors.LoadError(format_multi_exception(exceptions, eids))
		# Find the overlapping and aligned sample ranges for the given sampranges, if
		# possible. FIXME handle exception here
		subranges = pinfo_overlap(pinfos)
		subranges = srange_chain(subranges, linfo.sampranges[oind])
		ndet  = sum([pinfo.ndet for pinfo in pinfos])*linfo.detfact
		# At this point we know our samples are compatible, so can summarize timing from first
		nsamp = subranges[0,1]-subranges[0,0]
		srate = pinfos[0].srate
		t1    = pinfos[0].t1 + subranges[0,0]/srate
		# And finally return our result, which we will use in load
		return socommon.ProbeInfo(ndet, nsamp, t1, srate, oind=oind, sinds=sinds, subinfo=subinfo, subranges=subranges, pinfos=pinfos, exceptions=exceptions, eids=eids)
	def load(self, linfo, id, dets=None, detids=None, samprange=None, pinfo=None):
		if pinfo is None: pinfo = linfo.probe(id, dets=dets, detids=detids)
		subranges = pinfo.subranges
		if samprange is not None:
			subranges = srange_chain(subranges, srange_expand(samprange, linfo.downfact[pinfo.oind]))
		# Set up total obs
		obs = bunch.Bunch(ctime=None, boresight=None, hwp=None, tod=None, subids=[], errors=[], cuts=[], fill=[])
		append_fields = [("dets",0),("detids",0),("detpix",0),("bands",0),("point_offset",0),
			("polangle",0),("response",1)]
		for field, axis in append_fields: obs[field] = []
		# Read the individual observations into our structures
		dcum = 0
		for si, subid in enumerate(pinfo.subinfo.id):
			try:
				with bench.mark("PostLoader subload"):
					subobs = linfo.sublinfo.load(subid, dets=dets, detids=detids, samprange=subranges[si], pinfo=pinfo.pinfos[si])
				if linfo.demod:
					# Handles both demodulation and downsampling in one go
					with bench.mark("PostLoader demodulate"):
						subobs = demodulate(subobs, comps=linfo.comps, frel=1/linfo.down, dev=self.dev,
							pool_map=self.pool_map)
				elif linfo.down != 1:
					# Handles plain downsampling
					with bench.mark("PostLoader downsample"):
						subobs = downsample(subobs, down=linfo.down, dev=self.dev, pool_map=self.pool_map)
			except self.catch as e:
				pinfo.exceptions.append(e)
				pinfo.eids      .append(subid)
				continue
			# Initial obs setup
			if obs.tod is None:
				obs.ctime     = subobs.ctime
				obs.boresight = subobs.boresight
				obs.hwp       = subobs.hwp
				obs.site      = subobs.site
				obs.bore_ref  = subobs.bore_ref
				obs.sampoff   = subobs.sampoff
				obs.tod       = self.pool("tod").zeros((pinfo.ndet,len(subobs.ctime)), subobs.tod.dtype)
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
			raise errors.NothingLeft(format_multi_exception(pinfo.exceptions, pinfo.eids))
		# Concatenate the work-lists into the final arrays
		for field, axis in append_fields:
			obs[field] = np.concatenate(obs[field],axis) if obs[field][0] is not None else None
		obs.cuts = socut.Simplecut.detcat(obs.cuts)
		obs.fill = socut.Simplecut.detcat(obs.fill)
		# Trim tod in case we lost some detectors
		obs.tod = obs.tod[:dcum]
		# Record non-fatal errors and where they came from
		obs.errors = list(zip(pinfo.eids, pinfo.exceptions))
		return obs
	def prealloc(self, linfo):
		def s(ndet, nsamp): return utils.ceil(np.max(ndet*(nsamp+2))) # fourier-safe size
		# Max size of our output tod
		obsinfo = linfo.obsinfo
		nsamp = np.minimum(obsinfo.nsamp, linfo.maxnsamp)
		nout = s(obsinfo.ndet, nsamp)
		# max size of full processed subobs. Differs from nout by having fewer dets
		nper = s(obsinfo.ndet/linfo.detfact, nsamp)
		# max size of raw output from subloader
		nsub = s(obsinfo.ndet/linfo.detfact, nsamp*linfo.downfact)
		self.pool("tod").empty(nout, dtype=self.dtype)
		if linfo.demod or linfo.down != 1:
			self.pool("wtod").empty(nsub, dtype=self.dtype)
			self.pool("dtod").empty(nper, dtype=self.dtype)
			self.pool("ft")  .empty(nper, dtype=self.dtype)
		# Let our sub-loader preallocate too
		linfo.sublinfo.prealloc()

class PostLoadInfo(socommon.LoadInfo):
	"""Class representing a set of observations to load, and metadata needed to load it.
	Contains at least the .obsinfo member a numpy table of the observations and their properties,
	and provides the load() meathod for reading in an observation"""
	def __init__(self, loader, obsinfo, sublinfo, imap, sampranges, demod, comps, down, downfact, detfact, omap=None):
		super().__init__(loader, obsinfo, omap=omap)
		loc = locals()
		for key in ["sublinfo", "imap", "sampranges", "demod", "comps", "down", "downfact", "detfact"]:
			setattr(self, key, loc[key])
	def __getitem__(self, sel):
		res = super().__getitem__(sel)
		res.sampranges = self.sampranges[sel]
		res.imap       = utils.listslice(self.imap, sel)
		res.downfact   = self.downfact[sel]
		return res

def demodulate(data, frel=1, comps="TQU", mul=32, dev=None, pool_map={}):
	# Ok, if we get here, then we can demodulate
	if dev   is None: dev = device.get_device()
	pool_ft, pool_dtod, pool_wtod = [dev.pools[name] for name in utils.vmap(pool_map, ["ft","dtod","wtod"])]
	ncomp        = len(comps)
	ndet, insamp = data.tod.shape
	duration     = data.ctime[-1]-data.ctime[0]
	srate        = (insamp-1)/duration
	dtype        = data.tod.dtype
	ctype        = utils.complex_dtype(dtype)
	# Estimate hwp rotation speed. A bit inefficient, but we don't
	# require it to be unwound. Should I guarantee that it's unwound
	# after calibration? The disadvantage is that this reduces precision,
	# since float32 has 7 digits of precision, and the integer part can
	# take up 3-4 of those digits, leaving only 3-4 for the important
	# fractional part. To avoid this, the hwp angle would need to be
	# double precision.
	diffs = dev.np.diff(data.hwp)
	speed = dev.np.mean(diffs[dev.np.abs(diffs)<dev.np.pi])*srate
	fhwp  = np.abs(speed/(2*np.pi))
	# Find the last index where we complete a full revolution. We want
	# a whole number of rotations to avoid fourier bleeding. The cost of truncating
	# would be at most 0.5 s
	intrunc = gutils.find_last_crossing(data.hwp, data.hwp[0])
	# Find our output number of samples. This is ideally determined by
	# ofmax, but we are also restricted by fourier and mapmaking
	# considerations via mul
	ofmax   = float(frel*fhwp)
	ifmax   = srate/2
	onsamp  = fft.fft_len(utils.nint(intrunc*ofmax/ifmax/mul), factors=dev.lib.fft_factors)*mul
	# Prepare our resampling. For the tod we use fft-resampling. For the others, we use
	# linear interpolation. Averaging would be better, but these are smooth functions so
	# it should be good enough
	linresamp = gutils.LinResamp(intrunc, onsamp)
	# Prepare our output detectors. Our output data will have 2 or 3 times
	# as many detectors as we started with, since demodulation lets us recover
	# a T, Q and U-timestream from a single detector.
	assert comps == "TQU" or comps == "QU"
	detnames = []
	detids   = []
	modfuns  = []
	if "T"  in comps:
			detnames.append(np.char.add(data.dets,   "_0"))
			detids  .append(np.char.add(data.detids, "_0"))
			# 0.5 compensates for the multiplication by 2 later
			modfuns .append(lambda x:dev.np.full_like(x, 0.5))
	if "QU" in comps:
			detnames.append(np.char.add(data.dets,   "_1"))
			detnames.append(np.char.add(data.dets,   "_2"))
			detids  .append(np.char.add(data.detids, "_1"))
			detids  .append(np.char.add(data.detids, "_2"))
			modfuns .append(dev.np.cos)
			modfuns .append(dev.np.sin)
	ndup    = len(modfuns)
	odets   = np.concatenate(detnames)
	odetids = np.concatenate(detids)
	# Construct an output data with the given downsampling and detector duplication
	odata = bunch.Bunch()
	odata.dets   = odets
	odata.detids = odetids
	odata.ctime  = linresamp(data.ctime[:intrunc])
	odata.hwp    = None # already handled
	odata.point_offset = utils.repeat(data.point_offset, ndup, axis=0)
	odata.bands     = utils.repeat(data.bands, ndup)
	odata.polangle  = np.zeros(   len(odets) , dtype) # filled below
	odata.response  = np.zeros((2,len(odets)), dtype) # filled below
	odata.boresight = np.zeros((3,onsamp), data.boresight.dtype)
	odata.boresight[1] = linresamp(utils.unwind(data.boresight[1,:intrunc])) # az
	odata.boresight[0] = linresamp(data.boresight[0,:intrunc]) # el
	odata.boresight[2] = linresamp(data.boresight[2,:intrunc]) # roll
	# Resample cuts, and duplicate them across the virtual detectors
	recuts      = data.cuts.to_sampcut()[:,:intrunc].to_simple().resample(onsamp).simplify()
	odata.cuts  = socut.Simplecut.detcat([recuts]*len(modfuns))
	odata.tod   = pool_dtod.zeros((len(odets),onsamp), dtype) # filled below
	# Ok, here comes the actual demodulation part
	hwp = dev.np.array(data.hwp[:intrunc].astype(dtype))
	for i, fun in enumerate(modfuns):
		carrier = fun(4*hwp)
		work    = pool_wtod.array(data.tod[:,:intrunc])
		# Modulate
		work    *= carrier
		gutils.deslope(work, dev=dev, inplace=True)
		# Fourier-truncate. This step actually performs the filtering/downsampling
		# Sadly the ft must be contiguous, so we need a work buffer. We use our
		# tod work buffer for this, since its info has been transferred to fourier
		# space by then
		ftod    = pool_ft.empty((ndet, intrunc//2+1), ctype)
		dev.lib.rfft(work, ftod)
		ftod    = pool_ft.array(pool_wtod.array(ftod[:,:onsamp//2+1]))
		ftod   *= 2/intrunc
		# can finally transform back
		dev.lib.irfft(ftod, odata.tod[i*ndet:(i+1)*ndet])
	odata.cuts.gapfill(odata.tod, dev=dev)
	gutils.deslope(odata.tod, dev=dev, inplace=True, w=100)
	# T-detectors have response [1,0,0]
	if comps == "TQU": odata.response[0,:ndet] = 1
	elif comps != "QU": raise ValueError("Only comps='TQU' and comps='QU' supported")
	# cos-detectors have response [0,+detQ,-detU]
	# Equivalent to -ang
	odata.polangle[-2*ndet:-ndet] = -data.polangle
	# sin-detectors have response [0,+detU,+detQ]
	# Equivalent to (-(2*ang-pi/4)+pi/4)/2 = pi/4-ang
	odata.polangle[-ndet:] = np.pi/4-data.polangle
	odata.response[1,-2*ndet:] = 1
	# Everything else will be simply copied over
	for key in data:
		if key not in odata:
			odata[key] = data[key]
	return odata

def downsample(data, fsamp=None, down=None, mul=32, dev=None, pool_map=None):
	"""Downsample data either by the given down-factor, or to the given sample rate fsamp.
	Uses fourier-resampling for the tod, and linear resampling for the rest. The actual
	sample rate will be adjusted slightly to still be fourier- and gpu-friendly."""
	# Ok, if we get here, then we can demodulate
	if dev   is None: dev = device.get_device()
	pool_ft, pool_dtod, pool_wtod = [dev.pools[name] for name in utils.vmap(pool_map, ["ft","dtod","wtod"])]
	ndet, insamp = data.tod.shape
	duration     = data.ctime[-1]-data.ctime[0]
	srate        = (insamp-1)/duration
	dtype        = data.tod.dtype
	ctype        = utils.complex_dtype(dtype)
	# Get our target sample rate
	if fsamp is None: fsamp = srate/down
	# Find our output number of samples. This is ideally determined by
	# fsamp, but we are also restricted by fourier and mapmaking
	# considerations via mul
	onsamp  = fft.fft_len(utils.nint(insamp*fsamp/srate/mul), factors=dev.lib.fft_factors)*mul
	# Prepare our resampling. For the tod we use fft-resampling. For the others, we use
	# linear interpolation. Averaging would be better, but these are smooth functions so
	# it should be good enough
	linresamp = gutils.LinResamp(insamp, onsamp)
	# Construct an output data with the given downsampling and detector duplication
	odata = bunch.Bunch()
	odata.ctime  = linresamp(data.ctime)
	odata.boresight = np.zeros((3,onsamp), data.boresight.dtype)
	odata.boresight[1] = linresamp(utils.unwind(data.boresight[1])) # az
	odata.boresight[0] = linresamp(data.boresight[0]) # el
	odata.boresight[2] = linresamp(data.boresight[2]) # roll
	# Resample cuts, and duplicate them across the virtual detectors
	odata.cuts  = data.cuts.to_sampcut().to_simple().resample(onsamp).simplify().to_simple()
	# Resample the tod
	work    = pool_wtod.array(data.tod)
	ftod    = pool_ft.empty((ndet, data.tod.shape[-1]//2+1), ctype)
	dev.lib.rfft(work, ftod)
	ftod    = pool_ft.array(pool_wtod.array(ftod[:,:onsamp//2+1]))
	ftod   *= 2/insamp
	# can finally transform back
	odata.tod = pool_dtod.zeros((ndet,onsamp), dtype)
	dev.lib.irfft(ftod, odata.tod)
	# Everything else will be simply copied over
	for key in data:
		if key not in odata:
			odata[key] = data[key]
	return odata

def format_multi_exception(exceptions, subids):
	msgs   = np.array([str(ex) for ex in exceptions])
	uvals, order, inds = utils.find_equal_groups_fast(msgs)
	omsgs  = []
	for ui, uval in enumerate(uvals):
		mysubs = [subids[o] for o in order[inds[ui]:inds[ui+1]]]
		omsgs.append(",".join(mysubs) + ": " + uval)
	return ", ".join(omsgs)

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
		oinfo.band[gi] = "+".join(np.unique(obsinfo.band[group]))
	return oinfo

def split_obsinfo(obsinfo, targ_nsamps, fmt=":split%d"):
	"""Split each obsinfo[i] into chunks of at most targ_nsamps[i] samples, returning a new obsinfo
	and a sampranges map"""
	nsplits = (obsinfo.nsamp+targ_nsamps-1)//targ_nsamps
	oinfo   = np.repeat(obsinfo, nsplits)
	# Sampranges is tedious to do vectorized, so just do it in python
	sampranges = np.zeros((len(oinfo),2),int)
	ind_map    = np.zeros(len(oinfo),int)
	oi = 0
	for ii, (irow, targ_nsamp, nsplit) in enumerate(zip(obsinfo, targ_nsamps, nsplits)):
		for si in range(nsplit):
			i1 = si*targ_nsamp
			i2 = min(i1+targ_nsamp, irow.nsamp)
			sampranges[oi] = (i1, i2)
			oinfo.nsamp[oi] = i2-i1
			oinfo.dur[oi]  *= oinfo.nsamp[oi]/irow.nsamp
			oinfo.id[oi]   += fmt % si
			ind_map[oi]     = ii
			oi += 1
	assert oi == len(oinfo)
	return oinfo, ind_map, sampranges

def prepare_post(obsinfo, demod=False, down=None, comps="TQU", maxsize=None, maxdur=None, frel=1):
	# Find our total downsampling factor
	nsplits  = np.full(len(obsinfo),1,int)
	downfact = np.full(len(obsinfo),1,float)
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
		# calculate the total size of each obs
		sizes = obsinfo.ndet * detfact * obsinfo.nsamp / downfact
		# number of time splits for each
		nsplits = np.maximum(nsplits, utils.ceil(sizes/maxsize))
	# Split by duration
	if maxdur is not None:
		nsplits = np.maximum(nsplits, utils.floor(obsinfo.dur/maxdur)+1)
	return bunch.Bunch(nsplits=nsplits, downfact=downfact, detfact=detfact)

def nsplit2bsize(nsplit, nsamp, factors=[2,3,5,7], mul=32):
	return fft.fft_len(utils.ceil(nsamp/nsplit/mul), factors=factors, direction="above")*mul

def pinfo_overlap(pinfos, tol=0.1):
	"""Return overlapping sample ranges for the given list of ProbeInfos"""
	nsamps = np.array([pinfo.nsamp for pinfo in pinfos])
	t1s    = np.array([pinfo.t1    for pinfo in pinfos])
	srates = np.array([pinfo.srate for pinfo in pinfos])
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

def check_demod(demod="auto", has_hwp=False):
	if demod not in ["auto", "yes", "no"]:
		raise ValueError("demod must be 'auto', 'yes' or 'no', but got '%s'" % str(demod))
	if demod == "auto": return has_hwp
	elif demod == "no": return False
	elif has_hwp: return True
	else: raise ValueError("Asked to demodulate, but no hwp present")


###############################################
# Notes from loader reimplementation planning #
###############################################

# Pool remapping
# --------------
# Currently use pool_map, but this is cumbersome. Would be nice to have an object
# that acts just like dev.pools, but
# 1. It remaps names transparently
# 2. We can still access the remapping, since we will need it
#    for the avoid function

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

