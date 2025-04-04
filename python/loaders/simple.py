import numpy as np, time
from numpy.lib import recfunctions
from pixell import utils, fft, bunch
from .. import device

class SimpleLoader:
	def __init__(self, infofile, dev=None, mul=32):
		"""context is really just a list of tods and meta here"""
		self.obsinfo = read_obsinfo(infofile)
		self.dev     = dev or device.get_device()
		self.lookup  = {id:i for i,id in enumerate(self.obsinfo.id)}
		self.mul     = mul
	def query(self, query=None, wafers=None, bands=None):
		if query or wafers or bands: raise NotImplementedError
		return self.obsinfo
	def load(self, id):
		ind = self.lookup[id]
		# Reads pre-calibrated files
		obs = read_tod(self.obsinfo[ind].path, mul=self.mul)
		# Place obs.tod on device
		obs.tod = self.dev.pools.tod.reset().array(obs.tod)
		return obs

# Helpers below

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
