import numpy as np, time
from pixell import utils, fft, bunch
import so3g
from sotodlib import core, mapmaking, preprocess
from .. import device

# Standard Sotodlib slow load
class SotodlibLoader:
	def __init__(self, configfile, dev=None, mul=32):
		self.config, self.context = preprocess.preprocess_util.get_preprocess_context(configfile)
		self.mul     = mul
		self.dev     = dev or device.get_device()
	def query(self, query=None, wafers=None, bands=None):
		# wafers and bands should ideally be a part of the query!
		subids = mapmaking.get_subids(query, context=self.context)
		subids = mapmaking.filter_subids(subids, wafers=wafers, bands=bands)
		# Need base ids to look up rest of the info in obsdb
		obs_ids, wafs, bands = mapmaking.split_subids(subids)
		info     = self.context.obsdb.query("type='obs' and az_center is not null")
		iinds, oinds = utils.common_inds([info["obs_id"], obs_ids])
		# Build obsinfo for these ids
		dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("r","d"),("sweep","d",(4,2))]
		obsinfo = np.zeros(len(oinds), dtype).view(np.recarray)
		obsinfo.id    = subids[oinds]
		obsinfo.ndet  = 1000 # no simple way to get this :(
		obsinfo.nsamp = info["n_samples"][iinds]
		obsinfo.ctime = info["start_time"][iinds]
		obsinfo.dur   = info["stop_time"][iinds]-info["start_time"][iinds]
		# How come the parts that have to do with pointing.
		baz0  = info["az_center"][iinds]
		waz   = info["az_throw" ][iinds]
		bel0  = info["el_center"][iinds]
		# Should really have the proper array center, but it's clunky to get this,
		# and ultimately it doesn't matter
		#winfo   = get_wafer_info(self.context, info)
		#wind    = utils.find(winfo.wafers, wafs[inds])
		#obsinfo.r = winfo.r[wind]
		#pos     = winfo.pos[wind]
		obsinfo.r = np.full(len(iinds), 1.0*utils.degree)
		pos       = np.zeros((len(iinds),2))
		obsinfo.sweep = make_sweep(obsinfo.ctime, baz0, waz, bel0, pos)
		return obsinfo
	def load(self, subid):
		# Read in and calibrate all our data. This will depend on the
		# calibration steps listed in the config file
		#obs = self.context.get_obs(subid)
		#bunch.write("test_ptraw_soslow.hdf", bunch.Bunch(az=obs.boresight.az, el=obs.boresight.el, roll=obs.boresight.roll))
		#1/0
		try:
			obs = preprocess.load_and_preprocess(subid, self.config)
			n   = fft.fft_len(obs.samps.count//self.mul, factors=[2,3,5,7])*self.mul
			obs.restrict("samps", [0,n])
			# Merge all the cuts into a single cuts object
			cuts = merge_cuts([
				obs.preprocess.glitches.glitch_flags,
				obs.preprocess.jumps_2pi.jump_flag,
				obs.preprocess.jumps_slow.jump_flag,
			])
			cuts = rangesmatrix_to_simplecuts(cuts)
		except core.metadata.loader.LoaderError as e:
			raise utils.DataMissing(type(e).__name__ + " " + str(e))
		# Transform it to our current work format
		res  = bunch.Bunch()
		res.dets         = obs.dets.vals
		res.point_offset = np.array([obs.focal_plane.eta,obs.focal_plane.xi]).T
		res.polangle     = obs.focal_plane.gamma
		res.ctime        = o:s.timestamps
		res.boresight    = np.array([obs.boresight.el,obs.boresight.az]) # FIXME: roll
		res.cuts         = cuts
		res.tod          = self.dev.pools["tod"].array(obs.signal)
		res.tod         *= 1e6 * self.dev.np.array(obs.abscal.abscal_cmb[:,None])
		return res

# Helpers below

def get_wafer_info(context, obs_info):
	ref_obs = find_ref_obs(obs_info["obs_id"])
	return get_wafer_pointing_rough(context, ref_obs)

def find_ref_obs(ids):
	nwaf = np.char.count(np.char.rpartition(ids, "_")[:,2],"1")
	good = np.where(nwaf==np.max(nwaf))[0]
	return ids[good[len(good)//2]]

def get_wafer_pointing_rough(context, ref_obs_id):
	from scipy import ndimage
	print("ref_obs_id", ref_obs_id)
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

def merge_cuts(cuts):
	ocut = cuts[0]
	for cut in cuts:
		ocut += cut
	return ocut

def rangesmatrix_to_simplecuts(rmat):
	cuts = []
	for di, robj in enumerate(rmat.ranges):
		ranges = robj.ranges()
		det_cuts = np.zeros((len(ranges),3),np.int32)
		det_cuts[:,0] = di
		det_cuts[:,1] = ranges[:,0]
		det_cuts[:,2] = ranges[:,1]-ranges[:,0]
		cuts.append(det_cuts)
	return np.ascontiguousarray(np.concatenate(cuts).T)
