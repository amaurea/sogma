import numpy as np, time
from numpy.lib import recfunctions
from pixell import utils, fft, bunch, bench
import gpu_mm, cupy
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

def Loader(dbfile, mul=32):
	if dbfile.endswith(".yaml"): return SotodlibLoader(dbfile, mul=mul)
	else: return SimpleLoader(dbfile, mul=mul)

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

class SotodlibLoader:
	def __init__(self, configfile, mul=32, mode="standard"):
		from . import sodata
		# Set up our metadata loader
		self.config, self.context = sodata.preprocess.preprocess_util.get_preprocess_context(configfile)
		self.fast_meta = sodata.FastMeta(self.config, self.context)
		self.mul     = mul
		self.mode    = mode
	def query(self, query=None, wafers=None, bands=None):
		# wafers and bands should ideally be a part of the query!
		from sotodlib import mapmaking
		subids = mapmaking.get_subids(query, context=self.context)
		subids = mapmaking.filter_subids(subids, wafers=wafers, bands=bands)
		# Need base ids to look up rest of the info in obsdb
		obs_ids, wafs, bands = mapmaking.split_subids(subids)
		info     = self.context.obsdb.query()
		inds     = utils.find(info["obs_id"], obs_ids)
		# Build obsinfo for these ids
		dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("r","d"),("sweep","d",(4,2))]
		obsinfo       = np.zeros(len(inds), dtype).view(np.recarray)
		obsinfo.id    = subids
		obsinfo.ndet  = 1000 # no simple way to get this :(
		obsinfo.nsamp = info["n_samples"][inds]
		obsinfo.ctime = info["start_time"][inds]
		obsinfo.dur   = info["stop_time"][inds]-info["start_time"][inds]
		# How come the parts that have to do with pointing.
		# These arrays have a nasty habit of being object dtype
		baz0  = info["az_center"][inds].astype(np.float64)
		waz   = info["az_throw" ][inds].astype(np.float64)
		bel0  = info["el_center"][inds].astype(np.float64)
		# Need the rough pointing for each observation. This isn't
		# directly available in the obsid. We will assume that each wafer-slot
		# has approximately constant pointing offsets.
		# 1. Find a reference subid for each wafer slot
		ref_ids, inds = get_ref_subids(subids)
		wafer_centers = []
		wafer_rads    = []
		for ri, ref_id in enumerate(ref_ids):
			# 2. Get the focal plane offsets for this subid
			focal_plane = get_focal_plane(self.fast_meta, ref_id)
			mid, rad    = get_fplane_extent(focal_plane)
			wafer_centers.append(mid)
			wafer_rads   .append(rad)
		# 3. Expand to [nobs,{xi,eta}]
		wafer_centers = np.array(wafer_centers)[inds]
		wafer_rads    = np.array(wafer_rads   )[inds]
		# Fill in the last entries in obsinfo
		obsinfo.r     = wafer_rads
		obsinfo.sweep = make_sweep(obsinfo.ctime, baz0, waz, bel0, wafer_centers)
		return obsinfo
	def load(self, subid):
		from . import sodata
		try:
			with bench.show("read meta (total)"):
					meta = self.fast_meta.read(subid)
			# Load the raw data
			with bench.show("read data (total)"):
				data = sodata.fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
			# Calibrate the data
			with bench.show("calibrate (total)"):
				obs = sodata.calibrate_gpu(data, meta)
				#obs = sodata.calibrate(data, meta)
				#obs.tod = scratch.tod.array(obs.tod)
		except Exception as e:
			# FIXME: Make this less broad
			raise utils.DataMissing(str(e))
		# Place obs.tod on gpu. Hardcoded to use scratch.tod for now
		return obs

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
	info  = recfunctions.rec_append_fields(info, "id", ids)
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

def get_ref_subids(subids):
	"""Return one subid for each :ws:band combination"""
	subs = np.char.partition(subids, ":")[:,2]
	usubs, uinds, inds = np.unique(subs, return_index=True, return_inverse=True)
	return subids[inds], inds

def get_focal_plane(fast_meta, subid):
	fp_info     = fast_meta.fp_cache.get_by_subid(subid, fast_meta.det_cache)
	focal_plane = np.array([fp_info["xi"], fp_info["eta"], fp_info["gamma"]]).T # [:,{xi,eta,gamma}]
	good        = np.all(np.isfinite(focal_plane),1)
	focal_plane = focal_plane[good]
	return focal_plane

def get_fplane_extent(focal_plane):
	mid = np.mean(focal_plane,0)[:2]
	rad = np.max(utils.angdist(focal_plane[:,:2],mid[:,None]))
	return mid, rad

def get_wafer_info(context, obs_info):
	print("A", obs_info)
	ref_obs = find_ref_obs(obs_info["obs_id"])
	print("B", ref_obs)
	return get_wafer_pointing_rough(context, ref_obs)

def find_ref_obs(ids):
	nwaf = np.char.count(np.char.rpartition(ids, "_")[:,2],"1")
	good = np.where(nwaf==np.max(nwaf))[0]
	return ids[good[len(good)//2]]

def get_wafer_pointing_rough(context, ref_obs_id):
	from scipy import ndimage
	print("get_meta", ref_obs_id)
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
	from pixell import coordinates
	# given ctime,baz0,waz,bel [ntod], off[ntod,{xi,eta}], make
	# make sweeps[ntod,npoint,{ra,dec}]
	# so3g can't handle per-sample pointing offsets, so it would
	# force us to loop here. We therefore simply modify the pointing
	# offsets to we can simply add them to az,el
	az_off, el = coordinates.euler_rot((0.0, -bel0, 0.0), off.T)
	az  = (baz0+az_off)[:,None] + np.linspace(-0.5,0.5,npoint)*waz[:,None]
	el  = el   [:,None] + az*0
	ts  = ctime[:,None] + az*0
	sightline = so3g.proj.coords.CelestialSightLine.az_el(
		ts.reshape(-1), az.reshape(-1), el.reshape(-1), site="so", weather="typical")
	pos_equ = np.asarray(sightline.coords()) # [ntot,{ra,dec,cos,sin}]
	sweep   = pos_equ.reshape(len(ctime),npoint,4)[:,:,:2]
	return sweep
