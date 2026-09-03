import numpy as np, time, os
from pixell import utils, bunch, bench
from .. import device, socut

obsinfo_dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("baz","d"),("waz","d"),("bel","d"),("wel","d"),("roll","d"),("fhwp","d"),("r","d"),("sweep","d",(6,2))]

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
	return data

class SimpleLoader:
	def __init__(self, infofile, dev=None, mul=32):
		"""context is really just a list of tods and meta here"""
		self.dir     = os.path.dirname(infofile)
		self.obsinfo = read_obsinfo(infofile)
		self.dev     = dev or device.get_device()
		self.lookup  = {id:i for i,id in enumerate(self.obsinfo.id)}
		self.mul     = mul
	def query(self, query=None):
		if query is not None:
			if query.startswith("@"):
				ids  = np.loadtxt(query[1:], ndmin=1, dtype=str)
				inds = utils.find(self.obsinfo.id, ids)
				return self.obsinfo[inds]
			else:
				raise NotImplementedError
		else:
			return self.obsinfo
	def load(self, id, catch="expected", dets=None, detids=None, samprange=None, dtype=np.float32):
		"""Warning: dets, detids, samprange and dtype not implemented"""
		ind   = self.lookup[id]
		fname = os.path.join(self.dir, self.obsinfo[ind].id + ".hdf")
		# Reads pre-calibrated files
		with bench.mark("read"):
			obs = read_data(fname, dev=self.dev)
		# Add timing info
		obs.timing = [("read",bench.t.read),]
		return obs
	def load_multi(self, subids, samprange=None, catch="expected", dets=None, detids=None, post=None, dtype=np.float32):
		if len(subids) > 1: raise NotImplementedError
		return self.load(subids[0], dets=dets, detids=detids, samprange=samprange, dtype=dtype)
	def group_obs(self, obsinfo, mode=None):
		return bunch.Bunch(names=obsinfo.id, groups=[[i] for i in range(len(obsinfo))],
			bands=["?"], nullbands=[], joint=False, sampranges=None)
