import numpy as np, time
from pixell import utils, fft, bunch, bench, sqlite
from sotodlib import core, mapmaking, preprocess
from .. import device
from . import socommon

# TODO: Add load_multi

# Standard Sotodlib slow load
class SotodlibLoader:
	def __init__(self, configfile, dev=None, mul=32):
		self.config, self.context = preprocess.preprocess_util.get_preprocess_context(configfile)
		self.predb   = sqlite.open(self.config["archive"]["index"])
		self.mul     = mul
		self.dev     = dev or device.get_device()
		self.tags    = socommon.get_tags(self.context.obsdb.conn)
	def query(self, query=None, sweeps=True, output="sogma"):
		res_db, pycode, slices = socommon.eval_query(self.context.obsdb.conn, query, tags=self.tags, predb=self.predb)
		return socommon.finish_query(res_db, pycode, sweeps=sweeps, slices=slices, output=output)
	def load(self, subid, catch="expected"):
		# Read in and calibrate all our data. This will depend on the
		# calibration steps listed in the config file
		catch_list = catch2list(catch)
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
		except catch_list as e:
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

def catch2list(catch):
	if   isinstance(catch, (list,tuple)): return catch
	elif catch == "all":      return (Exception,)
	elif catch == "expected": return (core.metadata.loader.LoaderError,)
	elif catch == "none":     return ()
	else: raise ValueError("Unrecognized catch '%s'" % str(catch))
