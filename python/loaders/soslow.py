import numpy as np, time
from pixell import utils, fft, bunch, bench
from sotodlib import core, mapmaking, preprocess
from .. import device
from . import soquery, socommon

# Standard Sotodlib slow load
class SotodlibLoader:
	def __init__(self, configfile, dev=None, mul=32):
		self.config, self.context = preprocess.preprocess_util.get_preprocess_context(configfile)
		self.mul     = mul
		self.dev     = dev or device.get_device()
		self.tags    = soquery.get_tags(self.context.obsdb.conn)
	def query(self, query=None, sweeps=True, output="sogma"):
		if output not in ["sqlite", "resultset", "sogma"]:
			raise ValueError("Unrecognized output format '%s" % str(output))
		res_db = soquery.eval_query(self.context.obsdb.conn, query, tags=self.tags)
		if output == "sqlite": return res_db
		info   = core.metadata.resultset.ResultSet.from_cursor(res_db.execute("select * from obs"))
		if output == "resultset": return info
		dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("baz","d"),("waz","d"),("bel","d"),("wel","d"),("r","d"),("sweep","d",(4,2))]
		obsinfo = np.zeros(len(info), dtype).view(np.recarray)
		obsinfo.id    = info["subobs_id"]
		obsinfo.ndet  = utils.dict_lookup(flavor_ndets_per_band, info["tube_flavor"])
		obsinfo.nsamp = info["n_samples"]
		obsinfo.ctime = info["start_time"]
		obsinfo.dur   = info["stop_time"]-info["start_time"]
		# Here come the parts that have to do with pointing.
		# These arrays have a nasty habit of being object dtype
		obsinfo.baz   = info["az_center"].astype(np.float64) * utils.degree
		obsinfo.bel   = info["el_center"].astype(np.float64) * utils.degree
		obsinfo.waz   = info["az_throw" ].astype(np.float64) * utils.degree
		wafer_centers, obsinfo.r = socommon.wafer_info_multi(info["tube_slot"], info["wafer_slots_list"])
		if sweeps:
			obsinfo.sweep = socommon.make_sweep(obsinfo.ctime, obsinfo.baz, obsinfo.waz, obsinfo.bel, wafer_centers)
		return obsinfo
	def load(self, subid):
		# Read in and calibrate all our data. This will depend on the
		# calibration steps listed in the config file
		try:
			with bench.mark("load_and_preprocess"):
				obs = preprocess.load_and_preprocess(subid, self.config)
			with bench.mark("restrict"):
				n   = fft.fft_len(obs.samps.count//self.mul, factors=[2,3,5,7])*self.mul
				obs.restrict("samps", [obs.samps.offset,obs.samps.offset+n])
				# Get rid of invalid stuff
				good = np.isfinite(obs.focal_plane.gamma)
				obs.restrict("dets", obs.dets.vals[good])
			with bench.mark("merge_cuts"):
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
		with bench.mark("reformat"):
			res  = bunch.Bunch()
			res.dets         = obs.dets.vals
			res.point_offset = np.array([obs.focal_plane.eta,obs.focal_plane.xi]).T
			res.polangle     = obs.focal_plane.gamma
			res.ctime        = obs.timestamps
			res.boresight    = np.array([obs.boresight.el,obs.boresight.az,obs.boresight.roll])
			res.hwp          = obs.preprocess.hwp_angle.hwp_angle if "hwp_angle" in obs.preprocess else None
			res.cuts         = cuts
			res.response     = None
			res.tod          = self.dev.pools["tod"].array(obs.signal)
			res.tod         *= 1e6
		# Add timing info
		res.timing = [("load",bench.t.load_and_preprocess),("reformat",bench.t.merge_cuts+bench.t.reformat)]
		return res

# Helpers below

# Hard-coded raw wafer detector counts per band. Independent of
# telescope type, etc.
flavor_ndets_per_band = {"lf":118, "mf": 864, "uhf": 864}

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
