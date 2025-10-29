"""Classes and operations for cuts"""
import numpy as np
from pixell import utils, bench
from . import device

class Simplecut:
	"""Class representing cuts as an array [{det,start,len},ncut].
	This is what's used in the main part of sogma"""
	def __init__(self, dets=None, starts=None, lens=None, ndet=0, nsamp=0):
		if dets is None: dets, starts, lens = np.zeros((3,0), np.int32)
		self.dets   = np.asarray(dets,   np.int32)
		self.starts = np.asarray(starts, np.int32)
		self.lens   = np.asarray(lens,   np.int32)
		self.nsamp  = nsamp # not *really* necessary, but nice to have
		self.ndet   = ndet if ndet is not None else np.max(self.dets)+1
	@property
	def shape(self): return (self.ndet, self.nsamp)
	@property
	def nrange(self): return len(self.dets)
	def __getitem__(self, sel):
		"""Extract a subset of detectors and/or samples. Always functions
		as a slice, so the reslut is a new Sampcut. Only standard slicing is
		allowed in the sample direction - no direct indexing or indexing by lists."""
		if not isinstance(sel, tuple): sel = (sel,)
		if len(sel) == 0: return self
		if len(sel) > 1: raise IndexError("Only detector slicing supported for Simplecut")
		odets = np.arange(self.ndet)[sel]
		inds  = utils.find(odets, self.dets, -1)
		good  = inds>=0
		return Simplecut(dets=inds[good], starts=self.starts[good], lens=self.lens[good],
				ndet=np.sum(good), nsamp=self.nsamp)
	def __iter__(self):
		yield self.dets
		yield self.starts
		yield self.lens
	def to_sampcut(self):
		ranges = np.zeros((self.nrange,2),np.int32)
		ranges[:,0] = self.starts
		ranges[:,1] = self.starts+self.lens
		bins   =  _dets_to_bins(self.dets, self.ndet)
		return Sampcut(bins, ranges, nsamp=self.nsamp)
	@staticmethod
	def merge(cuts):
		# Currently implemented via Sampcut
		return Sampcut.merge([cut.to_sampcut() for cut in cuts]).simplify().to_simple()
	@staticmethod
	def detcat(cuts):
		"""Concatenate list of cuts in detector direction."""
		dcum = 0
		dets, starts, lens = [], [], []
		for cut in cuts:
			dets.append(cut.dets+dcum)
			starts.append(cut.starts)
			lens.append(cut.lens)
			dcum += cut.ndet
		dets, starts, lens = map(np.concatenate, [dets,starts,lens])
		return Simplecut(dets, starts, lens, ndet=dcum, nsamp=cut.nsamp)
	def gapfill(self, tod, w=10, dev=None):
		self.to_sampcut().gapfill(tod, w=w, dev=dev)

# This class represents cut ranges relative to the tod start, after
# any offsets in the OffsetAxis in the axismanager have been applied.
# So it's directly compatible with the signal array
class Sampcut:
	"""Class representing cuts as a flat list of sample ranges[nrange,{from,to}]
	with an auxilliary bins[ndet,{from,to}] array mapping from detectors to
	these ranges. This is mainly used during data loading."""
	def __init__(self, bins=None, ranges=None, nsamp=0):
		if bins   is None: bins   = np.zeros((0,2),np.int32)
		if ranges is None: ranges = np.zeros((0,2),np.int32)
		self.bins   = np.asarray(bins,   dtype=np.int32) # (ndet,  {from,to})
		self.ranges = np.asarray(ranges, dtype=np.int32) # (nrange,{from,to})
		self.nsamp  = nsamp # not *really* necessary, but nice to have
	def __getitem__(self, sel):
		"""Extract a subset of detectors and/or samples. Always functions
		as a slice, so the reslut is a new Sampcut. Only standard slicing is
		allowed in the sample direction - no direct indexing or indexing by lists."""
		if not isinstance(sel, tuple): sel = (sel,)
		if len(sel) == 0: return self
		if len(sel) > 2: raise IndexError("Too many indices for Sampcut. At most 2 indices supported")
		# Handle detector parts
		bins = self.bins[sel[0]]
		if len(sel) == 1:
			return Sampcut(bins, self.ranges, self.nsamp)
		start = sel[1].start or 0
		stop  = sel[1].stop  or self.nsamp
		step  = sel[1].step  or 1
		if start < 0: start += self.nsamp
		if stop  < 0: stop  += self.nsamp
		if step != 1: raise ValueError("stride != 1 not supported")
		ranges = np.clip(self.ranges-start, 0, stop-start)
		return Sampcut(bins, ranges, stop-start).simplify()
	@property
	def shape(self): return (len(self.bins),self.nsamp)
	def sum(self):
		"""Count cut samples per detector"""
		nper   = self.ranges[:,1]-self.ranges[:,0]
		ncum   = utils.cumsum(nper, endpoint=True)
		counts = ncum[self.bins[:,1]]-ncum[self.bins[:,0]]
		return counts
	def simplify(self):
		bins, ranges = _simplify_cuts(self.bins, self.ranges)
		return Sampcut(bins, ranges, nsamp=self.nsamp)
	def range_dets(self):
		bind  = np.full(len(self.ranges),-1,np.int32)
		for bi, bin in enumerate(self.bins):
			bind[bin[0]:bin[1]] = bi
		return bind
	@staticmethod
	def merge(cuts):
		return merge_sampcuts(cuts)
	def to_simple(self):
		ndet, nsamp = self.shape
		dets = self.range_dets()
		good = dets>=0
		return Simplecut(dets[good], self.ranges[good,0],
			self.ranges[good,1]-self.ranges[good,0],
			ndet=ndet, nsamp=nsamp)
	def gapfill(self, tod, w=10, dev=None):
		gapfill_sampcuts(tod, self, w=w, dev=dev)

def gapfill_sampcuts(tod, cuts, w=10, dev=None):
	if dev is None: dev = device.get_device()
	# Define sample ranges just before and after each
	# cut, where we will measure a mean value for dejumping
	with bench.mark("get_sampcut_borders", tfun=dev.time):
		borders = get_sampcut_borders(cuts,w)
	# Get on the format Kendrick wants
	with bench.mark("index_map", tfun=dev.time):
		ndet      = len(cuts.bins)
		ncut      = len(borders)
		nper      = cuts.bins[:,1]-cuts.bins[:,0]
		index_map        = np.zeros((ncut,5), np.int32)
		index_map[:,0]   = np.repeat(np.arange(ndet,dtype=np.int32),nper)
		index_map[:,1:]  = borders.reshape(ncut,-1)
		index_map2       = np.zeros((ncut,4), np.int32)
		index_map2[:,0]  = index_map[:,0]
		index_map2[:,1]  = np.repeat(cuts.bins[:,0],nper)
		index_map2[:,2:] = cuts.ranges
		index_map  = dev.np.array(index_map)
		index_map2 = dev.np.array(index_map2)
		bvals      = dev.np.zeros((ncut,2), tod.dtype)
	# Measure the border values
	with bench.mark("get_border_means", tfun=dev.time):
		dev.lib.get_border_means(bvals, tod, index_map)
	# Use them to deglitch
	with bench.mark("deglitch core", tfun=dev.time):
		dev.lib.deglitch(tod, bvals, index_map2)

def merge_sampcuts(cuts):
	"""Get the union of the given list of Sampcuts"""
	# 1. Flatten each cut. After this, franges will contain
	# a concatenated but not sorted set of flattened cuts.
	ndet = len(cuts[0].bins)
	nsamp= cuts[0].nsamp
	N    = nsamp+1 # +1 to avoid merging across dets
	franges = []
	for ci, cut in enumerate(cuts):
		# Get the det each range belongs to
		bind    = cut.range_dets().astype(np.int64)
		franges.append(np.clip(cut.ranges,0,nsamp) + bind[:,None]*N)
	# 3. Merge overlapping ranges. We count how many times
	# we enter and exit a cut region across all, and look for places
	# where this is nonzero. We flatten franges because we will
	# make a running tally of how many cut starts/ends we have encountred.
	franges = np.concatenate(franges).reshape(-1)
	if franges.size == 0: return Sampcut(nsamp=nsamp)
	# FIXME: Handle everything cut-case
	vals    = np.zeros(franges.shape,np.int32)
	vals[0::2] =  1
	vals[1::2] = -1
	# For starts and ends that happen at the same point, results could
	# be ambiguous, resulting in a variable amount of merging. We can
	# avoid this by making sure starts sort before ends, which will give
	# maximum merging
	sortkey = 2*franges + (np.arange(len(franges))&1) # even=start, odd=end, before sort
	order   = np.argsort(sortkey)
	franges = franges[order]
	vals    = vals   [order]
	# We're in a cut wherever the running tally is positive.
	# incut[0] will always be True, incut[-1] will always be False
	incut   = np.cumsum(vals)>0
	starts, ends = utils.mask2range(incut).T
	oranges = np.array([franges[starts],franges[ends]]).T
	# Unflatten into per-detector
	bind     = oranges[:,0]//N
	oranges -= bind[:,None]*N
	oranges  = oranges.astype(np.int32)
	obins    = _dets_to_bins(bind, ndet)
	# Finally construct a new cut with the result
	ocut     = Sampcut(obins, oranges, nsamp)
	return ocut

# These functions could use some low-level acceleration

def get_sampcut_borders(cuts, n=1):
	bind      = cuts.range_dets()
	border    = np.zeros((len(cuts.ranges),2,2),np.int32)
	border[:,0,1] = cuts.ranges[:,0]
	border[:,1,0] = cuts.ranges[:,1]
	# pad on both ends to make logic simpler
	padbind   = np.concatenate([[-1],bind,[-1]])
	padranges = np.concatenate([[[0,0]],cuts.ranges,[[cuts.nsamp,cuts.nsamp]]])
	left      = np.where(padbind[:-2]==padbind[1:-1],padranges[:-2,1],0)
	right     = np.where(padbind[ 2:]==padbind[1:-1],padranges[ 2:,0],cuts.nsamp)
	border[:,0,0] = np.maximum(cuts.ranges[:,0]-n, left)
	border[:,1,1] = np.minimum(cuts.ranges[:,1]+n, right)
	return border

# Private helpers

def _dets_to_bins(dets, ndet):
	nperdet  = np.bincount(dets, minlength=ndet)
	return _counts_to_bins(nperdet)

def _counts_to_bins(counts):
	edges  = utils.cumsum(counts, endpoint=True)
	bins   = np.zeros((len(counts),2),np.int32)
	bins[:,0] = edges[:-1]
	bins[:,1] = edges[1:]
	return bins

def _simplify_cuts(bins, ranges):
	"""Get sort cuts by detector, and remove empty ones"""
	obins   = []
	oranges = []
	o1 = 0
	for b1,b2 in bins:
		o2 = o1
		for ri in range(b1,b2):
			if ranges[ri,1] > ranges[ri,0]:
				o2 += 1
				oranges.append(ranges[ri])
		obins.append((o1,o2))
		o1 = o2
	obins = np.array(obins, np.int32).reshape((-1,2))
	oranges = np.array(oranges, np.int32).reshape((-1,2))
	return obins, oranges
