import numpy as np, time
from pixell import utils, fft, bunch
from . import socommon
from .. import device, socut

class SimpleLoader(socommon.Loader):
	def __init__(self, infofile, dev=None, dtype=np.float32, pool_map={}):
		"""context is really just a list of tods and meta here"""
		super().__init__(dev=dev, pool_map=pool_map, dtype=dtype)
		self.obsinfo = read_obsinfo(infofile)
		self.omap    = {id:i for i,id in enumerate(self.obsinfo.id)}
	def query(self, query=None, dets=None, detids=None):
		# No actual querying supported for now
		return SimpleLoadInfo(self, self.obsinfo, omap=self.omap, dets=None, detids=None)
	def probe(self, linfo, id, dets=None, detids=None):
		dets   = socommon.det_intersect(linfo.dets,   dets)
		detids = socommon.det_intersect(linfo.detids, detids)
		# No det-slicing yet, but easy to add
		ind = linfo.omap[id]
		row = linfo.obsinfo[ind]
		return ProbeInfo(row.ndet, row.nsamp, row.ctime, (row.nsamp-1)/row.dur, ind=ind)
	def load(self, linfo, id, dets=None, detids=None, samprange=None, pinfo=None):
		if pinfo is None: pinfo = linfo.probe(id, dets=dets, detids=detids)
		with bench.mark("SimpleLoader read"):
			obs = read_tod(self.obsinfo[ind].path, mul=self.dev.lib.bsize)
		with bench.mark("SimpleLoader tod2dev"):
			obs.tod = self.pool("tod").array(obs.tod)
		obs.subids = [srange_suffix(id, samprange)]
		obs.errors = []
		return obs
	def prealloc(self, linfo):
		def s(ndet, nsamp): return utils.ceil(np.max(ndet+(nsamp+2))) # fourier-safe size
		# Max size of our output tod
		obsinfo = linfo.obsinfo
		nsamp = np.minimum(obsinfo.nsamp, linfo.maxnsamp)
		nout = s(obsinfo.ndet, nsamp)
		self.pool("tod").empty(nout, dtype=self.dtype)

class SimpleLoadInfo(socommon.LoadInfo): pass

def read_obsinfo(fname): return np.load(fname).view(np.recarray)
def write_obsinfo(fname, obsinfo): np.save(fname, obsinfo)

def write_data(fname, data, dev=None):
	if dev is None: dev = device.get_device()
	out = data.copy()
	# Make sure tod is on the cpu
	out.tod = dev.get(data.tod)
	for key in ["cuts", "fill"]:
		out[key] = out[key].to_simple().export()
	# Should be ready to write now
	bunch.write(fname, out)

def read_data(fname, dev=None):
	if dev is None: dev = device.get_device()
	data = bunch.read(fname)
	# Make sure tod is where it should be
	data.tod = dev.pools["tod"].array(data.tod)
	for key in ["cuts", "fill"]:
		dets, starts, lens = data[key].T
		data[key] = socut.Simplecut(dets=dets, starts=starts, lens=lens, ndet=data.tod.shape[0], nsamp=data.tod.shape[1])
	if "npad" not in data: data.npad = 0
	return data
