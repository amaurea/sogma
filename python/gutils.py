import time, contextlib
import numpy as np
from pixell import utils
from . import device
from .logging import L

def round_up  (n, b): return (n+b-1)//b*b
def round_down(n, b): return n//b*b

def apply_window(tod, nsamp, exp=1):
	"""Apply a cosine taper to each end of the TOD."""
	if nsamp <= 0: return
	ap = device.anypy(tod)
	taper   = 0.5*(1-ap.cos(ap.arange(1,nsamp+1)*ap.pi/nsamp))
	taper **= exp
	tod[...,:nsamp]  *= taper
	tod[...,-nsamp:] *= taper[::-1]

def aranges(lens):
	ntot = np.sum(lens)
	itot = np.arange(ntot)
	offs = np.repeat(np.cumsum(np.concatenate([[0],lens[:-1]])), lens)
	return itot-offs

def cumsum0(vals):
	res = np.empty(len(vals), vals.dtype)
	if res.size == 0: return res
	res[0] = 0
	res[1:] = np.cumsum(vals[:-1])
	return res

def split_ranges(dets, starts, lens, maxlen):
	# Vectorized splitting of detector ranges into subranges.
	# Probably premature optimization, since it's a bit hard to read.
	# Works by duplicating elements for too long ranges, and then
	# calculating new sub-offsets inside these
	dets, starts, lens = [np.asarray(a) for a in [dets,starts,lens]]
	nsplit  = (lens+maxlen-1)//maxlen
	odets   = np.repeat(dets, nsplit)
	subi    = aranges(nsplit)
	sublen  = np.repeat(lens, nsplit)
	subns   = np.repeat(nsplit, nsplit)
	offs    = subi*sublen//subns
	ostarts = np.repeat(starts, nsplit)+offs
	offs2   = (subi+1)*sublen//subns
	oends   = offs2-offs
	return odets, ostarts, oends

@contextlib.contextmanager
def leakcheck(dev, msg):
	try:
		dev.garbage_collect()
		t1 = dev.time()
		m1 = dev.memuse()
		yield
	finally:
		dev.garbage_collect()
		t2 = dev.time()
		m2 = dev.memuse()
		L.print("leak %8.4f MB %s" % ((m2-m1)/1024**2, msg), level=2)

def legbasis(order, n, ap=np):
	x   = ap.linspace(-1, 1, n, dtype=np.float32)
	out = ap.zeros((order+1, n),dtype=np.float32)
	out[0] = 1
	if order>0:
		out[1] = x
	for i in range(1,order):
		out[i+1,:] = ((2*i+1)*x*out[i]-i*out[i-1])/(i+1)
	return out

def leginverses(basis):
	"""Given a lebasis B[nmode,nsamp], compute all the nsamp truncated
	inverses of BB'. The result will have shape [nsamp,nmode,nmode]"""
	# This function is 20x faster on the cpu
	ap   = device.anypy(basis)
	nmode, nsamp = basis.shape
	mask = ap.tril(ap.ones((nsamp,nsamp), basis.dtype))
	BBs  = ap.einsum("ai,bi,ji->jab", basis, basis, mask)
	iBBs = ap.zeros_like(BBs)
	# These ones should be safe to invert
	iBBs[nmode-1:] = ap.linalg.inv(BBs[nmode-1:])
	# Handle the unsafe ones manually
	for i in range(0, nmode-1):
		iBBs[i,:i+1,:i+1] = ap.linalg.inv(BBs[i,:i+1,:i+1])
		for j in range(i+1, nmode):
			iBBs[i,j,j] = 1
	return iBBs

def safe_inv(a):
	with utils.nowarn():
		res = 1/a
		res[~np.isfinite(res)] = 0
	return res

def safe_invert_ivar(ivar, tol=1e-3):
	vals = ivar[ivar!=0]
	ref  = np.mean(vals[::100])
	iivar= ivar*0
	good = ivar>ref*tol
	iivar[good] = 1/ivar[good]
	return iivar

# Want something like DataSet or AxisManager that's low-overhead
# while being able to wrap multiple types of data, ideally without
# needng to be invasive.
#
# Approach 1:
# * Internally, fields are wrapped in a simple class that
#   presents a unified interface
# * When adding fields, optional arguments can specify functions
#   that implement things like slicing, downsampling etc.
# * Fields must know which axes have special meaning, but I don't
#   want to store e.g. detector names again and again for each field.
#   I also don't want to have to do expensive loops through every field
#   to check if axes are compatible. So axis information must be factored
#   out, like it is in axismanager. But fields should still be as standalone
#   as possible.
# Possible implementation:
# * A field contains:
#   * data: The underlying object
#   * axes: [(axind,axinfo),(...)]. Axind can be negative. Unmanaged axes not mentioned
#     axinfo is a reference to a shared axinfo object, so it's not duplicated
#   * resample(data, ax, onsamp): optional downsampling function.
#     Necessary for the sample axis, for anything that would be changing the sample rate.
#     The TOD would use fft resampling, the boresight plain thinning, and cuts would
#     broaden. This obsoletes axinfo. How do I handle this? If I make a new one, I'll
#     end up with duplicated axinfo again. Could take the new ax as argument, but that
#     would require redoing the remapping calculation for each field. :/
#   * restrict(data, ax, sel): optional restriction of axis. sel can be a slice,
#     mask or indices. Defaults to using getitem
#   * repeat(data, ax, ncount): optional
# It might be useful to allow a collection of fields to defer normalization
# until later. This would avoid expensive calculations and copying of data
# as field after field is added.

def demodulate(data, frel=2, comps="TQU", dev=None):
	if "hwp" not in data or data.hwp is None:
		raise ValueError("Cannot demodulate without HWP angles")
	if dev is None: dev = device.get_device()
	ncomp        = len(comps)
	ndet, insamp = data.tod.shape
	duration     = data.ctime[-1]-data.ctime[0]
	srate        = duration/(insamp-1)
	# Estimate hwp rotation speed. A bit inefficient, but we don't
	# require it to be unwound. Should I guarantee that it's unwound
	# after calibration? The disadvantage is that this reduces precision,
	# since float32 has 7 digits of precision, and the integer part can
	# take up 3-4 of those digits, leaving only 3-4 for the important
	# fractional part. To avoid this, the hwp angle would need to be
	# double precision.
	diffs = dev.np.diff(hwp)
	fhwp  = dev.np.mean(diffs[dev.np.abs(diffs)<dev.np.pi])/srate
	# Compare the current nyquist frequency with the target nyquist frequency
	# to find our ideal downsampling factor. Currently only integer downsampling
	# is supported
	ofmax   = frel*fhwp
	ifmax   = srate/2
	down    = utils.nint(ifmax/ofmax*1.1)
	onsamp  = (insamp+down-1)//down
	# Prepare our output detectors. Our output data will have 2 or 3 times
	# as many detectors as we started with, since demodulation lets us recover
	# a T, Q and U-timestream from a single detector.
	assert comps == "TQU" or comps == "QU"
	detnames = []
	modfuns  = []
	if "T"  in comps:
			detnames.append(np.char.add(data.dets, "_one"))
			modfuns .append(lambda x:dev.np.full_like(x, 0.5))
	if "QU" in comps:
			detnames.append(np.char.add(data.dets, "_cos"))
			detnames.append(np.char.add(data.dets, "_sin"))
			modfuns .append(dev.np.cos)
			modfuns .append(dev.np.sin)
	ndup  = len(modfuns)
	odets = np.char.concatenate(detnames)
	# Construct an output data with the given downsampling and detector duplication
	odata = bunch.Bunch()
	odata.dets  = odets
	odata.ctime = data.ctime[::down]
	odata.hwp   = data.hwp[::down]
	odata.point_offset = utils.repeat(data.point_offset, ndup, axis=0)
	odata.polangle  = np.zeros(   len(odets) , data.tod.dtype) # filled below
	odata.response  = np.zeros((2,len(odets)), data.tod.dtype) # folled below
	odata.boresight = data.boresight[:,::down]
	odata.cuts      = detdup_cuts(downsample_cuts(data.cuts, down), ndet, ndup)
	odata.tod   = dev.np.zeros((len(odets),onsamp),data.tod.dtype)
	# TODO: Think about how to handle gpu memory allocation here.
	# There are three big arrays involved:
	# * input tod
	# * modulated input tod
	# * ft of modulated input tod
	# The outputs are much smaller.
	# After all the loading is done, none of these big arrays will be needed
	# any more.
	# For now, I'll just use the heap
	# Ok, here comes the actual demodulation part
	for i, fun in enumerate(modfuns):
		carrier = fun(4*data.hwp).astype(data.tod.dtype)
		ftod    = dev.rfft(data.tod*carrier)
		# This step actually perform the filtering/downsampling
		ftod    = ftod[:,:onsamp//2+1].copy()
		ftod   *= 2/insamp
		dev.irfft(ftod, odata.tod[i*ndet:(i+1)*ndet])
	# TODO: gapfill and deslope
	# Update the polangle and response
	iQ = np.cos(2*data.polangle)
	iU = np.sin(2*data.polangle)
	# T-detectors have response [1,0,0]
	if comps == "TQU": odata.response[0,:ndet] = 1
	elif comps != "QU": raise ValueError("Only comps='TQU' and comps='QU' supported")
	# cos-detectors have response [0,+detQ,-detU]
	# Equivalent to -ang
	odata.polangle[-2*ndet:-ndet] = -data.polangle
	# sin-detectors have response [0,+detU,+detQ]
	# Equivalent to (-(2*ang-pi/4)+pi/4)/2 = pi/4-ang
	odata.polangle[-ndet:] = np.pi/4-data.polangle
	odata.response[1,-2*ndet:] = 1
	return odata

def simplify_cuts(cuts):
	"""Return a new cuts with overlapping regions merged"""
	# I came up with a vectorized implementation of this, but it
	# was fragile, assuming that multiple short cuts couldn't be inside
	# one long cut. That's unliekly to happen, but I prefer a correct,
	# readable implementation, especially for something that's probably going
	# to be fast enough anyway.
	if len(cuts) == 0: return cuts
	ocuts = []
	d,s,n = cuts[0]
	for d2,s2,n2 in cuts[1:]:
		if d2 != d or s2 > s+n:
			ocuts.append((d,s,n))
		else:
			n = max(s+n,s2+n2)-s
	ocuts.append((d,s,n))
	return np.array(ocuts)

def downsample_cuts(cuts, down):
	"""Downsample cuts, cutting any osample that's even partially cut in the input"""
	# cuts are [[det,start,len],...]
	ocuts = np.zeros_like(cuts)
	ocuts[:,0] = cuts[:,0]
	ocuts[:,1] = cuts[:,1]//down
	ocuts[:,2] = (cuts[:,1]+ocuts[:,2]-1)//down-ocuts[:,1]+1
	ocuts      = simplify_cuts(ocuts)
	return ocuts

def detdup_cuts(cuts, ndet, ndup):
	ncut  = len(cuts)
	ocuts = utils.repeat(cuts, ndup, axis=0)
	for i in range(ndup):
		ocut[i*ncut:(i+1)*ncut,0] += i*ndet
	return ocuts
