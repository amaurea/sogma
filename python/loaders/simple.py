import numpy as np, time
from numpy.lib import recfunctions
from pixell import utils, fft, bunch
from . import socommon
from .. import device

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
		nout = s(obsinfo.ndet, obsinfo.nsamp)
		self.pool("tod").empty(nout, dtype=self.dtype)

class SimpleLoadInfo(socommon.LoadInfo): pass

# Helpers below

def read_obsinfo(fname, nmax=None):
	# FIXME: Currently missing baz, bel and waz. Not present in current text files
	dtype = [("path","U256"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("r","d"),("sweep","d",(4,2)),("band","U100")]
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
		res.detids       = f["dets"]                 # [ndet]
		res.point_offset = f["point_offset"][:,::-1] # [ndet,{y,x}]
		res.polangle     = f["polangle"]             # [ndet]
		bore = f["boresight"]
		n    = fft.fft_len(bore.shape[1]//mul, factors=[2,3,5,7])*mul
		res.ctime        = bore[0,:n]                   # [nsamp]
		res.boresight    = np.empty((3,n),bore.dtype)
		res.boresight[0] = bore[2,:n] # el
		res.boresight[1] = bore[1,:n] # az
		res.boresight[2] = 0 # no roll in simple format currently
		res.hwp          = np.zeros_like(res.boresight[0]) # no hwp in simple format currently
		res.tod          = f["tod"][:,:n]               # [ndet,nsamp]
		res.cuts         = mask2cuts(f["cuts"][:,:n])
		res.response     = None
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
