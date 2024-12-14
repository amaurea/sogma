import numpy as np, time
from numpy.lib import recfunctions
from pixell import utils, fft, bunch
import gpu_mm
from .gmem import scratch, plan_cache

# Loading of obs lists, metadata and data. The interface needs at least these:
# 1. get a list of observations
# 2. get scanpat info for each observation
# 3. load a calibrated observation

# The last part would ideally be a standard axismanager, but for now it will just
# be a bunch, so the rest of the code doesn't need to be changed

# What should be an "observation"? At least for the sat, files are 10 minutes of 1 wafer,
# but this 10-min-division is treated as an implementation detail, and I can't rely on
# this being available directly in the future. The basic unit is around one hour.

# Getting the observations and getting the info often involves loading the same
# database, so having them as independent functions is wasteful. Let's make a
# class that can be queried.

class SimpleLoader:
	def __init__(self, infofile, mul=32):
		"""context is really just a list of tods and meta here"""
		self.obsinfo = read_obsinfo(infofile)
		self.lookup  = {id:i for i,id in enumerate(self.obsinfo.id)}
		self.mul     = mul
	def query(self, query=None, wafers=None, bands=None):
		if query or wafers or bands: raise NotImplementedError
		return self.obsinfo
	def load(self, id):
		ind = self.lookup[id]
		# Reads pre-calibrated files
		obs = read_tod(self.obsinfo[ind].path, mul=self.mul)
		# Place obs.tod on gpu. Hardcoded to use scratch.tod for now
		obs.tod = scratch.tod.array(obs.tod)
		return obs

# Standard Sotodlib slow load
class SotodlibLoader:
	def __init__(self, contextfile, mul=32, mode="standard"):
		from sotodlib import core
		self.context = core.Context(contextfile)
		self.mul     = mul
		self.mode    = mode
	def query(self, query=None, wafers=None, bands=None):
		# wafers and bands should ideally be a part of the query!
		from sotodlib import mapmaking
		subids = mapmaking.get_subids(args.query, context=self.context)
		subids = mapmaking.filter_subids(subids, wafers=wafers, bands=bands)
		# Need base ids to look up rest of the info in obsdb
		obs_ids, wafs, bands = mapmaking.split_subid(subids)
		info     = self.context.obsdb.query()
		inds     = utils.find(info["obs_id"], obs_ids)
		# Build obsinfo for these ids
		dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("r","d"),("sweep","d",(4,2))]
		obsinfo = np.zeros(len(ids), dtype).view(np.recarray)
		obsinfo.id    = subids
		obsinfo.ndet  = 1000 # no simple way to get this :(
		obsinfo.nsamp = info["n_samples"][inds]
		obsinfo.ctime = info["start_time"][inds]
		obsinfo.dur   = info["end_time"][inds]-info["start_time"][inds]
		# How come the parts that have to do with pointing.
		baz0  = info["az_center"][inds]
		waz   = info["az_throw" ][inds]
		bel0  = info["el_center"][inds]
		# Need the wafer det positions. First find a representative observation
		# that has as many of the wafers we care about as possible. This is really
		# clunky!
		winfo   = get_wafer_info(context, info)
		wind    = utils.find(winfo.wafers, wafs[inds])
		obsinfo.r = winfo.r[wind]
		obsinfo.sweep = make_sweep(obsinfo.ctime, baz0, waz, bel0, winfo.pos[wind])
		return obsinfo
	def load(self, subid):
		import so3g
		meta = self.context.get_meta(subid)
		n    = fft.fft_len(meta.samps.count//mul, factors=[2,3,5,7])*mul
		meta.restrict("samps", [0,n])
		# add dummy cuts if missing
		if "glitch_flags" not in meta.flags:
			meta.flags.wrap("glitch_flags", shape=("dets","samps"), cls=so3g.proj.RangesMatrix.zeros)
		# Restrict to detectors with useful calibration. May raise DataMissing
		meta = validate_meta(meta)
		# Read our actual data
		if   self.mode == "standard":
			obs  = self.context.get_obs(subid, meta=meta)
		elif self.mode == "fast":
			obs  = read_obs_fast(self.context, meta)
		obs  = calibrate_obs_real(obs)
		tod  = scratch.tod.array(obs.signal)
		tod  = calibrate_obs_fourier(obs, tod)
		# Transform it to our current work format
		res  = bunch.Bunch()
		res.dets         = obs.dets.vals
		res.point_offset = np.array([obs.focal_plane.eta,obs.focal_plane.xi]).T
		res.polangle     = obs.focal_plane.gamma
		res.ctime        = obs.timestamps
		res.boresight    = np.array([obs.boresight.el,obs.boresight.az]) # FIXME: roll
		res.tod          = obs.signal
		return res

# Helpers below

def get_filelist(ifiles):
	fnames = []
	for gfile in ifiles:
		for ifile in sorted(utils.glob(gfile)):
			if ifile.startswith("@"):
				with open(ifile[1:],"r") as f:
					for line in f:
						fnames.append(line.strip())
			else:
				fnames.append(ifile)
	return fnames

def read_obsinfo(fname, nmax=None):
	dtype = [("path","U256"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("r","d"),("sweep","d",(4,2))]
	info  = np.loadtxt(fname, dtype=dtype, max_rows=nmax, ndmin=1).view(np.recarray)
	ids   = np.char.rpartition(np.char.rpartition(info.path,"/")[:,2],".")[:,0]
	info  = recfunctions.rec_append_field(info, "id", ids)
	# Convert to standard units
	info.dur   *= utils.minute
	info.r     *= utils.degree
	info.sweep *= utils.degree # [ntod,npoint,{ra,dec}]
	return info

def read_tod(fname, mul=32):
	"""Read a tod file in the simple npz format we use"""
	res = bunch.Bunch()
	# Could do this in a loop, but we do it explicitly so we
	# can document which fields should be present.
	# Change ra,dec and x,y order to dec,ra and y,x, as agreed with Kendrick
	with np.load(fname) as f:
		res.dets         = f["dets"]                 # [ndet]
		res.point_offset = f["point_offset"][:,::-1] # [ndet,{y,x}]
		res.polangle     = f["polangle"]             # [ndet]
		bore = f["boresight"]
		n    = fft.fft_len(bore.shape[1]//mul, factors=[2,3,5,7])*mul
		res.ctime        = bore[0,:n]                   # [nsamp]
		res.boresight    = bore[[2,1],:n]               # [{el,az},nsamp]
		res.tod          = f["tod"][:,:n]               # [ndet,nsamp]
		res.cuts         = mask2cuts(f["cuts"][:,:n])
	for key in res:
		res[key] = np.ascontiguousarray(res[key])
	#print("ndet %d nsamp %d primes %s" % (res.tod.shape[0], res.tod.shape[1], utils.primes(res.tod.shape[1])))
	return res

# Cuts will be represented by det[nrange], start[nrange], len[nrange]. This is similar to
# the format used in ACT, but in our test files we have boolean masks instead, which we need
# convert. This is a bit slow, but is only needed for the test data
def mask2cuts(mask):
	# Find where the mask turns on/off
	t01 = time.time()
	dets, starts, lens = [], [], []
	for idet, dmask in enumerate(mask):
		# index of all on/off and off/on transitions. We put it in a
		# list so we can prepend and append to it
		edges = [1+np.nonzero(np.diff(dmask,1))[0]]
		# Ensure we start with off→on and end with on→off
		if dmask[ 0]: edges.insert(0,[0])
		if dmask[-1]: edges.append([mask.shape[1]])
		edges = np.concatenate(edges) if len(edges) > 1 else edges[0]
		start = edges[0::2].astype(np.int32)
		stop  = edges[1::2].astype(np.int32)
		dets  .append(np.full(len(start),idet,np.int32))
		starts.append(start)
		lens  .append(stop-start)
	dets   = np.concatenate(dets)
	starts = np.concatenate(starts)
	lens   = np.concatenate(lens)
	t02 = time.time()
	return dets, starts, lens

def get_wafer_info(context, obs_info):
	ref_obs = find_ref_obs(obs_info["obs_id"])
	return get_wafer_pointing_rough(context, ref_obs)

def find_ref_obs(ids):
	nwaf = np.char.count(np.char.rpartition(ids, "_")[:,2],"1")
	good = np.where(nwaf==np.max(nwaf))[0]
	return good[len(good)//2]

def get_wafer_pointing_rough(context, ref_obs_id):
	from scipy import ndimage
	meta = context.get_meta(ref_obs_id)
	good = np.isfinite(meta.focal_plane.xi) & np.isfinite(meta.focal_plane.eta)
	# Classify them by wafer slot
	uwafs, inds = np.unique(meta.det_info.wafer_slot, return_inverse=True)
	# We only want the good ones
	inds = inds[good]
	# Get the average pointing
	pos  = np.array([meta.focal_plane[a][good] for a in ["xi","eta"]])
	n    = np.maximum(np.bincount(inds, minlength=len(uwafs)),1)
	mid  = utils.bincount(inds, pos, minlength=len(uwafs))/n
	# Max distance from this point
	rs   = utils.angdist(pos,mid[:,inds])
	wrad = ndimage.maximum(rs, inds+1, np.arange(1,len(uwafs)+1))
	# wafs = list of wafers present
	# pos  = center_pos[nwaf,{xi,eta}] for each wafer
	# r    = max_rad[nwaf] for each wafer
	return bunch.Bunch(wafers=uwafs, pos=mid.T, r=wrad)

def make_sweep(ctime, baz0, waz, bel0, off, npoint=4):
	import so3g
	# make sweeps[ntod,npoint,{ra,dec}]
	baz = baz0[:,None]  + np.linspace(-0.5,0.5,npoint)*waz[:,None]
	bel = bel0[:,None]  + baz*0
	ts  = ctime[:,None] + baz*0
	sightline = so3g.proj.coords.CelestialSightLine.az_el(
		ts.reshape(-1), baz.reshape(-1), bel.reshape(-1), site="so", weather="typical")
	# A single dummy detector at the wafer center
	q_det = so3g.proj.quat.rotation_xieta(off[None,0], off[None,1], 0)
	pos_equ = sightline.coords(q_det) # [{ra,dec,c1,s1},1,nsamp]
	sweep   = np.moveaxis(pos_equ[:2,0].reshape(2,len(ctime),npoint),(1,2,0),(0,1,2)) # [nobs,npoint,{az,el}]
	sweep[:,:,0] = utils.unwind(sweep[:,:,0])
	return sweep

def make_dummy_cuts(ndet, nsamp):
	dets, starts, lens = np.zeros((3,0),np.int32)
	return dets, starts, lens

# This stuff is just temporary. We will call sotodlib stuff when it's ready.
# Much of the validation here would also be unnecessary with more mature metadata

class DataMissing(Exception): pass

def validate_meta(meta):
	# We have the obligatory fields, right?
	for field in ["signal", "boresight", "focal_plane", "det_cal", "timestamps"]:
		if field not in meta:
			raise DataMissing("field '%s' missing" % field)
	# Disqualify overly cut detectors
	good  = mapmaking.find_usable_detectors(obs)
	# Require valid detector offsets
	good &= np.isfinite(obs.focal_plane.xi)
	good &= np.isfinite(obs.focal_plane.eta)
	good &= np.isfinite(obs.focal_plane.gamma)
	# Require valid gains
	good &= np.isfinite(obs.det_cal.phase_to_pW)
	good &= np.isfinite(obs.det_cal.tau_eff)
	if np.sum(good) == 0: raise DataMissing("gain")
	# Restrict to these detectors
	meta.restrict("dets", meta.dets.vals[good])
	return meta # orig was modified, but return can be handy for chaining

def calibrate_obs_real(obs):
	"""The real-space part of the detector calibration. The parts
	that can be done without Fourier transforming. This assumes
	validate_meta has already been called To eliminate detectors
	with invalid metadata."""
	from sotodlib import mapmaking
	from sotodlib.hwp import hwp_angle_model
	# We have the obligatory fields, right?
	for field in ["signal", "boresight", "focal_plane", "det_cal", "timestamps"]:
		if field not in obs:
			raise DataMissing("field '%s' missing" % field)
	# We're scanning, right?
	speeds = estimate_scanspeed(obs, quantile=[0.1,0.9])
	if speeds[0] < 0.05*utils.degree or speeds[1] > 10*utils.degree:
		raise DataMissing("unreasonable scanning speed")
	## Fix buggy hwp angles
	#try: obs = hwp_angle_model.apply_hwp_angle_model(obs)
	#except ValueError: raise DataMissing("ambiguous hwp angle")
	#obs.hwp_angle = utils.unwind(obs.hwp_angle)
	# Fix boresight
	mapmaking.fix_boresight_glitches(obs)
	# Fix the signal itself
	obs.signal *= (obs.det_cal.phase_to_pW * obs.abscal.abscal_cmb * 1e6)[:,None]
	utils.deslope(obs.signal, w=5, inplace=True)
	tod_ops.get_gap_fill(obs, flags=obs.glitch_flags, swap=True)
	# Reject detectors with unreasonable sensitivity
	dsens = estimate_dsens(obs)
	asens = np.sum(dsens**-2)**-0.5
	good  = (dsens > 50)&(dsens < 4000)
	obs.restrict("dets", obs.dets.vals[good])
	if obs.dets.count == 0: raise DataMissing("unrealistic sensitivity")
	# Reject detctors with crazy values. We could make this test more
	# sensitive by putting it after demodulationg, but we're looking for
	# very strong values anyway, so it should be OK to put it here
	sig = estimate_max_signal(obs)
	good = sig < 1e8
	obs.restrict("dets", obs.dets.vals[good])
	if obs.dets.count == 0: raise DataMissing("unrealistic signal strength")
	return obs

def calibrate_obs_fourier(obs, gtod):
	ft = scratch.ft.empty((gtod.shape[0],gtod.shape[1]//2+1),utils.complex_dtype(gtod.dtype))
	gpu_mm.cufft.rfft(gtod, ft, plan_cache=plan_cache)
	wafers, edges, order = utils.find_equal_groups_fast(obs.det_info.wafer.array)
	for wi, wafer in enumerate(wafers):
		params = obs.iir_params["ufm_"+wafer]
		if params["a"] is None or params["b"] is None: raise DataMissing("iir_params")
			ft[order[edges[wi]:edges[wi+1]]] /= cupy.array(filters.iir_filter(iir_params=params)(freq, obs))
		# Timeconst filter handles per-detector values directly, so it's easier to deal with
	ft /= cupy.array(filters.timeconst_filter(timeconst=obs.det_cal.tau_eff)(freq, obs))
	gpu_mm.cufft.irfft(ft, gtod, plan_cache=plan_cache)
	# Normalize?
	gtod /= gtod.shape[1]
	return gtod

def estimate_scanspeed(obs, step=100, quantile=0.5):
	speeds = np.abs(np.diff(obs.boresight.az[::step])/np.diff(obs.timestamps[::step]))
	return np.quantile(speeds, quantile)

def estimate_max_signal(obs):
	return np.max(np.abs(np.diff(obs.signal,axis=1)),1)

def read_obs_fast(context, meta, dtype=np.float32):
	detsets = np.unique(meta.det_info.detset)
	if len(detsets) > 1: raise DataMissing("Detectors from different detsets must be in different subobs")
	detset  = detsets[0]
	obs_id, wafer, band = subid.split(":")
	files = context.obsfiledb.get_files(obs_id)[detset]
	files = [v[0] for v in files]
	# Allocate signal
	obs   = meta
	obs.wrap_new("signal", shape=("dets","samps"), dtype=dtype)
	bore  = core.AxisManager(obs.samps)
	bore.wrap_new("az",   ("samps",), dtype=np.float64)
	bore.wrap_new("el",   ("samps",), dtype=np.float64)
	bore.wrap_new("roll", ("samps",), dtype=np.float64)
	obs.wrap("boresight", bore)
	# And read into these arrays
	fsamp1 = 0
	with fast_g3.open_multi(files) as reader:
		inds = utils.find(reader.fields["signal/data"].names, obs.dets.vals)
		data.queue("signal/data", rows=inds)
		data.queue("signal/times")
		data.queue("ancil/az_enc")
		data.queue("ancil/el_enc")
		data.queue("ancil/boresight_enc")
		for fi, data in enumerate(reader.read):
			n      = data["signal/times"].size
			fsamp2 = fsamp1 + n
			# fsamp1:fsamp2 is the range into the full timestream we cover.
			# But we may not want all this
			off1 = max(obs.samps.offset-fsamp1,0)
			off2 = min(fsamp2,obs.samps.offset+obs.samps.count,0)
			oslice = (Ellipsis,slice(max(fsamp1-off1,0),fsamp2-off1-off2))
			islice = (Ellipsis,slice(off1,n-off2))
			obs.signal        [oslice] = data["signal/data"]        [islice].astype(dtype)
			obs.boresight.az  [oslice] = data["ancil/az_enc"]       [islice]
			obs.boresight.el  [oslice] = data["ancil/el_enc"]       [islice]
			obs.boresight.roll[oslice] = data["ancil/boresight_enc"][islice]
			fsamp1 = fsamp2
	return obs
