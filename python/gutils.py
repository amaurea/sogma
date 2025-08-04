import time, contextlib
import numpy as np
from pixell import utils, colors
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
		L.print("%sleak %8.4f MB %s%s" % (colors.lbrown, (m2-m1)/1024**2, msg, colors.reset), level=2)

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

# I originally wrote this for sotodlib
def parse_recentering(desc):
	"""Parse an object centering description, as provided by the --center-at argument.
	The format is [from=](ra:dec|name),[to=(ra:dec|name)],[up=(ra:dec|name|system)]
	from: specifies which point is to be centered. Given as either
	  * a ra:dec pair in degrees
	  * the name of a pre-defined celestial object (e.g. Saturn), which should not move
	    appreciably in celestial coordinates during a TOD
	to: the point at which to recenter. Optional. Given as either
	  * a ra:dec pair in degrees
	  * the name of a pre-defined celestial object
	  Defaults to ra=0,dec=0 or ra=0,dec=90, depending on the projection
	up: which direction should point up after recentering. Optional. Given as either
	  * the name of a coordinate system (e.g. hor, cel, gal), in which case
	    up will point towards the north pole of that system
	  * a ra:dec pair in degrees
	  * the name of a pre-defined celestial object
	  Defualts to the celestial north pole
	
	Returns "info", a bunch representing the recentering specification in more python-friendly
	terms. This can later be passed to evaluate_recentering to get the actual euler angles that perform
	the recentering.
	
	Examples:
	  * 120.2:-13.8
	    Centers on ra = 120.2°, dec = -13.8°, with up being celestial north
	  * Saturn
	    Centers on Saturn, with up being celestial north
	  * Uranus,up=hor
	    Centers on Uranus, but up is up in horizontal coordinates. Appropriate for beam mapping
	  * Uranus,up=hor,to=0:90
	    As above, but explicitly recenters on the north pole
	"""
	# If necessary the syntax above could be extended with from_sys, to_sys and up-sys, which
	# so one could specify galactic coordiantes for example. Or one could generalize
	# from ra:dec to phi:theta[:sys], where sys would default to cel. But for how I think
	# this is enough.
	args = desc.split(",")
	info  = {"to":"auto", "up":"cel", "from_sys":"cel", "to_sys":"cel", "up_sys":"cel"}
	for ai, arg in enumerate(args):
		# Split into key,value
		toks = arg.split("=")
		if ai == 0 and len(toks) == 1:
			key, val = "from", toks[0]
		elif len(toks) == 2:
			key, val = toks
		else:
			raise ValueError("parse_recentering wants key=value format, but got %s" % (arg))
		# Handle the values
		if ":" in val:
			val = [float(w)*utils.degree for w in val.split(":")]
		info[key] = val
	if "from" not in info:
		raise ValueError("parse_recentering needs at least the from argument")
	return info

def find_scan_periods(obsinfo, ttol=7200, atol=2*utils.degree, mindur=0):
	"""Given an obsinfo as returned by a loader.query(), return the set
	of contiguous scanning periods in the form [:,{ctime_from,ctime_to}].

	ttol: number of seconds of gap between obs before a new period is started.
	For LAT scans, one doesn't save space by starting a new period before 2 hours
	have passed due to the super-wide arcs one scans in. For shorter-scanning
	surveys, this could be reduced.
	"""
	from scipy import ndimage
	info = np.array([obsinfo[a] for a in ["baz", "bel", "waz", "wel", "ctime", "dur"]]).T
	# Get rid of nan entries
	bad  = np.any(~np.isfinite(info),1)
	# Can eliminate too short tods here. We did this in act because those would
	# have unreliable az bounds, but this isn't an issue with SO
	bad |= info[:,-1] < mindur
	info = info[~bad]
	t1   = info[:,-2]
	info = info[np.argsort(t1)]

	# Start, end
	t1   = info[:,-2]
	t2   = t1 + info[:,-1]
	# Remove angle ambiguities
	info[:,0] = utils.rewind(info[:,0])
	# How to find jumps:
	# 1. It's a jump if the scanning changes
	# 2. It's also a jump if a the interval between tod-ends and tod-starts becomes too big
	changes    = np.abs(info[1:,:4]-info[:-1,:4])
	jumps      = np.any(changes > atol,1)
	jumps      = np.concatenate([[0], jumps]) # from diff-inds to normal inds

	# Time in the middle of each gap
	gap_times = np.mean(find_period_gaps(np.array([t1,t2]).T, ttol=ttol),1)
	gap_inds  = np.searchsorted(t1, gap_times)
	jumps[gap_inds] = True
	# raw:  aaaabbbbcccc
	# diff: 00010001000
	# 0pre: 000010001000
	# cum:  000011112222
	labels  = np.cumsum(jumps)
	linds   = np.arange(np.max(labels)+1)
	t1s     = ndimage.minimum(t1, labels, linds)
	t2s     = ndimage.maximum(t2, labels, linds)
	# Periods is [nperiod,{start,end}] in ctime. Start is the start of the first tod
	# in the scanning period. End is the end of the last tod in the scanning period.
	periods = np.array([t1s, t2s]).T
	return periods

def find_period_gaps(periods, ttol=60):
	"""Helper for find_scan_periods. Given the [:,{ctime_from,ctime_to}] for all
	the individual scans, returns the times at which the gap between the end of
	a tod and the start of the next is greater than ttol (default 60 seconds)."""
	# We want to sort these and look for any places
	# where a to is followed by a from too far away. To to this we need to keep
	# track of which entries in the combined, sorted array was a from or a to
	periods = np.asarray(periods)
	types   = np.zeros(periods.shape, int)
	types[:,1] = 1
	types   = types.reshape(-1)
	ts      = periods.reshape(-1)
	order   = np.argsort(ts)
	ts, types = ts[order], types[order]
	# Now look for jumps
	jumps = np.where((ts[1:]-ts[:-1] > ttol) & (types[1:]-types[:-1] < 0))[0]
	# We will return the time corresponding to each gap
	gap_times = np.array([ts[jumps], ts[jumps+1]]).T
	return gap_times

def detwise_axb(tod, x, a=None, b=None, one=1, inplace=False, tod_mul=0, abmul=1, adjoint=False, dev=None):
	if dev is None: dev = device.get_device()
	if tod.dtype != np.float32: raise ValueError("Only float32 supported")
	if not inplace: tod = tod.copy()
	ndet, nsamp = tod.shape
	B    = dev.np.empty([2,nsamp],dtype=tod.dtype) # [{a,b},nsamp]
	B[0] = x
	B[1] = one
	if adjoint:
		# [coeffs] = [abmul  *B] [otod]
		# [tod   ] = [tod_mul*I]
		# coeffs   = abmul*B.dot(otod.T)
		# tod      = tod_mul*otod
		# transposed: coeffs' = abmul*otod.dot(B')
		coeffs = dev.np.zeros((2,ndet), tod.dtype) # [{a,b},ndet]
		dev.lib.sgemm("T", "N", ndet, 2, nsamp, abmul, tod, nsamp, B, nsamp, 0, coeffs, ndet)
		if tod_mul != 1: tod *= tod_mul
	else:
		# otod = abmul*coeffs.T.dot(B) + tod_mul*tod
		# transposed abmul*B'.dot(coeffs) + tod_mul*tod'
		# In standard form
		# [otod] = [abmul*B' tod_mul*I] [coeffs tod]'
		coeffs = dev.np.array([a,b]) # [{a,b},ndet]
		dev.lib.sgemm("N", "T", nsamp, ndet, 2, abmul, B, nsamp, coeffs, ndet, tod_mul, tod, nsamp)
	return tod, coeffs[0], coeffs[1]

def deslope(signal, v1=None, v2=None, w=10, inplace=False, n=None, dev=None, external_v=False, return_edges=False):
	if dev  is None: dev  = device.get_device()
	if n    is None: n    = signal.shape[-1]
	if not inplace: signal = signal.copy()
	# Allow us to work on other arrays than 2d.
	pre, nsamp = signal.shape[:-1], signal.shape[-1]
	signal = signal.reshape(-1, nsamp)
	# Measure edge values
	if not external_v:
		v1 = dev.np.mean(signal[:, :w],1)
		v2 = dev.np.mean(signal[:,n-w:n],1)
	# Build a basis that ignores the padded area
	x   = linspace(0, 1, n, pad=nsamp-n, dtype=signal.dtype, dev=dev)
	one = dev.np.full(nsamp, 1, dtype=signal.dtype)
	one[n:] = 0
	otod = detwise_axb(signal, x, v2-v1, v1, one=one, tod_mul=1, abmul=-1, dev=dev, inplace=True)[0]
	# Restore to original shape
	otod = otod.reshape(pre+(nsamp,))
	v1   = v1.reshape(pre)
	v2   = v2.reshape(pre)
	if return_edges: return otod, v1, v2
	else: return otod

def adjoint_deslope(otod, v1, v2, w=10, inplace=False, n=None, dev=None, external_v=False, return_edges=False):
	if dev  is None: dev  = device.get_device()
	if n    is None: n    = otod.shape[-1]
	if not inplace: v1, v2 = v1.copy(), v2.copy()
	# Allow us to work on other arrays than 2d.
	pre, nsamp = otod.shape[:-1], otod.shape[-1]
	otod, v1, v2 = otod.reshape(-1, nsamp), v1.reshape(-1), v2.reshape(-1)
	x = linspace(0, 1, n, pad=nsamp-n, dtype=otod.dtype, dev=dev)
	one = dev.np.full(nsamp, 1, dtype=signal.dtype)
	one[n:] = 0
	# Transpose of the desloping itself
	signal, a, b = detwise_axb(otod, x, one=one, tod_mul=1, abmul=-1, dev=dev, inplace=True, adjoint=True)
	# Transpose of [a;b;v1;v2] = [-1 1;1 0;1 0;0 1][v1;v2] => [v1;v2] = [-1 1 1 0;1 0 0 1][a;b;v1;v2]
	v1 += b-a
	v2 += a
	if not external_v:
		# Transpose of measuring v1 and v2 from signal edges
		# [v1]   [1'/w   0 ] [s1]     [s1] = [1/w  0  I  0] [v1]
		# [v2] = [ 0   1'/w] [s2]     [s2]   [ 0  1/w 0  I] [v2]
		# [s1]   [ I     0 ]       =>                       [s1]
		# [s2]   [ 0     I ]                                [s2]
		signal[:,:w]    += v1[:,None]/w
		signal[:,n-w:n] += v2[:,None]/w
	# Restore shape
	signal = signal.reshape(pre+(nsamp,))
	v1     = v1.reshape(pre)
	v2     = v2.reshape(pre)
	if return_edges: return signal, v1, v2
	else: return signal

def linspace(start, stop, num=50, pad=0, endpoint=True, dtype=None, dev=None):
	if dev is None: dev = device.get_device()
	res = dev.np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)
	if pad > 0:
		res_pad = dev.np.zeros(num+pad, dtype=dtype)
		res_pad[:num] = res
		res = res_pad
	return res

def downgrade(tod, bsize, inclusive=False, op=None):
	ap = device.anypy(tod)
	if op is None: op = ap.mean
	nwhole = tod.shape[-1]//bsize
	nall   = (tod.shape[-1]+bsize-1)//bsize
	nblock = nall if inclusive else nwhole
	otod = ap.zeros(tod.shape[:-1]+(nblock,), tod.dtype)
	otod[...,:nwhole] = op(tod[...,:nwhole*bsize].reshape(tod.shape[:-1]+(nwhole,bsize)),-1)
	if nblock > nwhole: otod[...,-1] = op(tod[...,nwhole*bsize:],-1)
	return otod

def block_scale(tod, bscale, bsize=1, inplace=False):
	ap = device.anypy(tod)
	if not inplace: tod = tod.copy()
	nblock = tod.shape[-1]//bsize
	btod   = tod[...,:nblock*bsize].reshape(tod.shape[:-1]+(nblock,bsize))
	btod  *= bscale[...,:nblock,None]
	# incomplete last block
	if tod.shape[-1] > nblock*bsize:
		tod[...,nblock*bsize:] *= bscale[...,-1,None]
	return tod

def linint(arr, x):
	ap = device.anypy(arr)
	ix = ap.floor(x).astype(int)
	ix = ap.clip(ix, 0, arr.shape[-1]-2)
	rx = x-ix
	return arr[...,ix]*(1-rx) + arr[...,ix+1]*rx

def logint(arr, x):
	ap = device.anypy(arr)
	ix = ap.floor(x).astype(int)
	ix = ap.clip(ix, 0, arr.shape[-1]-2)
	# in log-log, it makes sense to start counting from 1 instead of
	# 0 to avoid log(0), hence the +1.
	rx   = (ap.log(x+1)-ap.log(ix+1))/(ap.log(ix+2)-ap.log(ix+1))
	larr = ap.log(arr)
	return ap.exp(larr[...,ix]*(1-rx) + larr[...,ix+1]*rx)
