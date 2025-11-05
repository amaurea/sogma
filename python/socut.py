"""Classes and operations for cuts"""
import numpy as np
from pixell import utils, bench
from . import device, gutils

# Tempting to store cuts on device, since they will need to be
# transfered there before they can be used anyway, but all cuts
# manipulation is much faster on the cpu, and that's the main
# purpose of these classes.
#
# On the other hand, it's nice to be able to use these classes to
# dejump, gapfill etc, and it's annoying to have a bunch of
# unnecessary conversions happen there. Could try to cache, but
# cache is a pain.
#
# Will instead make a 3rd class, Devicecut, which is fully on the
# gpu and has all the index maps etc. precomputed.

class Simplecut:
	"""Class representing cuts as an array [{det,start,len},ncut].
	This is what's used in the main part of sogma, where Devicecut
	isn't used. Functions for gapfilling, dejumping etc. are provided,
	but are implemented by converting to Devicecut, so working direclty with
	Devicecut is more efficient. On the other hand, cuts manipulation
	is much more efficient with this class than Devicecut"""
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
	def resample(self, n):
		starts = utils.floor(self.starts*n/self.nsamp)
		ends   = utils.ceil ((self.starts+self.lens-1)*n/self.nsamp+1)
		ends   = np.minimum(ends, n)
		lens   = ends-starts
		return Simplecut(dets=self.dets, starts=starts, lens=lens, ndet=self.ndet, nsamp=n)
	def to_device(self, dev=None):
		if dev is None: dev = device.get_device()
		return Devicecut(self, dev=dev)
	# If these are going to be called a lot, then it's
	# more efficient call to_device() once and work on
	# that object instead
	def dejump(self, tod, w=10, dev=None):
		return self.to_device(dev=dev).dejump(tod, w=w)
	def gapfill(self, tod, w=10, dev=None):
		return self.to_device(dev=dev).gapfill(tod, w=w)
	def clear(self, tod, dev=None):
		return self.to_device(dev=dev).clear(tod)
	def insert(self, tod, vals, dev=None):
		return self.to_device(dev=dev).insert(tod, vals)
	def extract(self, tod, out=None, dev=None):
		return self.to_device(dev=dev).extract(tod, out=out)
	# These helpers are used by the helper class Devicecut
	def _borders(self, n=1, out=None):
		if out is None: out = np.zeros((self.nrange,2,2),np.int32)
		# We need the ranges to be sorted det-samp
		out[:,0,1] = self.starts
		out[:,1,0] = self.starts+self.lens
		out[:,0,0] = np.maximum(out[:,0,1]-n,0)
		out[:,1,1] = np.minimum(out[:,1,0]+n,self.nsamp)
		# Handle transition to next detector
		if self.nrange > 1:
			inside = np.where(self.dets[1:]==self.dets[:-1])[0]
			out[inside+1,0,0] = np.maximum(out[inside+1,0,0],out[inside  ,1,0])
			out[inside  ,1,1] = np.minimum(out[inside  ,1,1],out[inside+1,0,1])
		return out
	def _index_map(self, borders):
		index_map        = np.zeros((self.nrange,5), np.int32)
		index_map[:,0]   = self.dets
		index_map[:,1:]  = borders.reshape(self.nrange,4)
		return index_map
	def _index_map2(self, skip_r0=False):
		index_map2 = np.zeros((self.nrange,4), np.int32)
		index_map2[:,0]  = self.dets
		# A bit cumbersome to get the index of each range's first det-sibling
		if not skip_r0 and self.nrange > 0:
			nper = np.bincount(self.dets, minlength=self.ndet)
			offs = gutils.cumsum0(nper)
			index_map2[:,1] = np.repeat(offs, nper)
		index_map2[:,1]  = self.dets
		index_map2[:,2]  = self.starts
		index_map2[:,3]  = self.starts+self.lens
		return index_map2

# This class represents cut ranges relative to the tod start, after
# any offsets in the OffsetAxis in the axismanager have been applied.
# So it's directly compatible with the signal array
class Sampcut:
	"""Class representing cuts as a flat list of sample ranges[nrange,{from,to}]
	with an auxilliary bins[ndet,{from,to}] array mapping from detectors to
	these ranges. This is mainly used during data loading. Functions for
	gapfilling, dejumping etc. are provided, but are implemented by
	converting to Simplecut and then to Devicecut, so working direclty with
	Devicecut is more efficient."""
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
	@staticmethod
	def empty(ndet, nsamp):
		return Sampcut(bins=np.zeros((ndet,2),np.int32), nsamp=nsamp)
	@property
	def shape(self): return (len(self.bins),self.nsamp)
	@property
	def nrange(self): return len(self.ranges)
	def sum(self):
		"""Count cut samples per detector"""
		nper   = self.ranges[:,1]-self.ranges[:,0]
		ncum   = gutils.cumsum0(nper, endpoint=True)
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
	def to_device(self, dev=None):
		return self.to_simple().to_device(dev)
	# If these are going to be called a lot, then it's
	# considerably more efficient call to_device() once and work on
	# that object instead
	def dejump(self, tod, w=10, dev=None):
		return self.to_device(dev=dev).dejump(tod, w=w)
	def gapfill(self, tod, w=10, dev=None):
		return self.to_device(dev=dev).gapfill(tod, w=w)
	def clear(self, tod, dev=None):
		return self.to_device(dev=dev).clear(tod)
	def insert(self, tod, vals, dev=None):
		return self.to_device(dev=dev).insert(tod, vals)
	def extract(self, tod, out=None, dev=None):
		return self.to_device(dev=dev).extract(tod, out=out)

class Devicecut:
	"""Same as Simplecut, but on the device and not meant to be changed"""
	def __init__(self, simplecut, dev=None):
		self.dev       = dev or device.get_device()
		self.simplecut = simplecut
		self.dets      = self.dev.np.asarray(simplecut.dets,   np.int32)
		self.starts    = self.dev.np.asarray(simplecut.starts, np.int32)
		self.lens      = self.dev.np.asarray(simplecut.lens,   np.int32)
		# Calculated when needed
		self._offs      = None
		self._ncutsamps  = None
		self.borders    = None
		# These use a different cache scheme because they
		# depend on potentially variable arguments
		self.index_map  = None
		self.index_map2 = None
		self.prev_w     = None
		self.has_r0     = False
	@property
	def shape(self): return self.simplecut.shape
	@property
	def nrange(self): return self.simplecut.nrange
	@property
	def offs(self):
		if self._offs is None:
			self._offs = self.dev.np.asarray(gutils.cumsum0(self.lens), np.int32)
		return self._offs
	@property
	def ncutsamps(self):
		if self._ncutsamps is None:
			self._ncutsamps = int(self.offs[-1] + self.lens[-1])
		return self._ncutsamps
	def dejump(self, tod, w=10):
		return self._deglitch(tod, self.dev.lib.dejump, w=w)
	def gapfill(self, tod, w=10):
		return self._deglitch(tod, self.dev.lib.gapfill, w=w, skip_r0=True)
	def clear(self, tod):
		self.dev.lib.clear_ranges(tod, self.dets, self.starts, self.lens)
		return tod
	def insert(self, tod, vals):
		if self.offs is None:
			self.offs = self.dev.np.asarray(gutils.cumsum0(self.lens), np.int32)
		self.dev.lib.insert_ranges(tod, vals, self.offs, self.dets, self.starts, self.lens)
		return tod
	def extract(self, tod, out=None):
		if self.offs is None:
			self.offs = self.dev.np.asarray(gutils.cumsum0(self.lens), np.int32)
		if out is None:
			out = self.dev.np.zeros(self.ncutsamps, tod.dtype)
		self.dev.lib.extract_ranges(tod, out, self.offs, self.dets, self.starts, self.lens)
		return out
	# Helpers
	def _deglitch(self, tod, fun, w=10, skip_r0=False):
		# Define sample ranges just before and after each
		# cut, where we will measure a mean value for dejumping
		if self.borders is None or w != self.prev_w:
			with bench.mark("get_sampcut_borders", tfun=self.dev.time):
				self.borders = self.simplecut._borders(w)
		# Get on the format Kendrick wants
		if self.index_map is None or w != self.prev_w:
			with bench.mark("index_map", tfun=self.dev.time):
				self.index_map = self.dev.np.asarray(self.simplecut._index_map(self.borders))
		if self.index_map2 is None or (not skip_r0 and not self.has_r0):
			with bench.mark("index_map2", tfun=self.dev.time):
				self.index_map2 = self.dev.np.asarray(self.simplecut._index_map2(skip_r0=skip_r0))
		# Measure the border values
		with bench.mark("get_border_means", tfun=self.dev.time):
			bvals = self.dev.np.zeros((self.nrange,2), tod.dtype)
			self.dev.lib.get_border_means(bvals, tod, self.index_map)
		# Use them to deglitch
		with bench.mark("deglitch core", tfun=self.dev.time):
			fun(tod, bvals, self.index_map2)
		return tod

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
	starts, ends = gutils.mask2range(incut).T
	oranges = np.array([franges[starts],franges[ends]]).T
	# Unflatten into per-detector
	bind     = oranges[:,0]//N
	oranges -= bind[:,None]*N
	oranges  = oranges.astype(np.int32)
	obins    = _dets_to_bins(bind, ndet)
	# Finally construct a new cut with the result
	ocut     = Sampcut(obins, oranges, nsamp)
	return ocut

# Private helpers

def _dets_to_bins(dets, ndet):
	nperdet  = np.bincount(dets, minlength=ndet)
	return _counts_to_bins(nperdet)

def _counts_to_bins(counts):
	edges  = gutils.cumsum0(counts, endpoint=True)
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
