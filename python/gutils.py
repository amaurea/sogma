import time, contextlib
import numpy as np
from pixell import utils, colors, bunch, fft
from . import device
from .logging import L

class RecoverableError(Exception): pass

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

def cumsum0(vals, endpoint=False):
	ap  = device.anypy(vals)
	res = ap.empty(len(vals)+endpoint, vals.dtype)
	if res.size == 0: return res
	res[0] = 0
	if endpoint: res[1:] = ap.cumsum(vals)
	else:        res[1:] = ap.cumsum(vals[:-1])
	return res

def split_ranges(dets, starts, lens, maxlen):
	# Vectorized splitting of detector ranges into subranges.
	# Probably premature optimization, since it's a bit hard to read.
	# Works by duplicating elements for too long ranges, and then
	# calculating new sub-offsets inside these
	if len(dets) == 0: return dets, starts, lens
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

def mask2range(mask):
	"""Convert a binary mask [True,True,False,True,...] into
	a set of ranges [:,{start,stop}]."""
	# We consider the outside of the array to be False
	ap     = device.anypy(mask)
	mask   = ap.concatenate([[False],mask.astype(bool,copy=False),[False]]).astype(np.int8)
	diffs  = ap.diff(mask)
	start  = ap.where(diffs>0)[0]
	stop   = ap.where(diffs<0)[0]
	return np.array([start,stop]).T

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

# Fast matrix inverse for 3x3 matrices. Taken from
# https://www.dr-lex.be/random/matrix-inv.html

def det3(a, axes=(0,1)):
	"""Determinant of 3x3 matrices. Much faster than np.linalg.det"""
	ap = device.anypy(a)
	(a11, a12, a13),(a21,a22,a23),(a31,a32,a33) = ap.moveaxis(a,axes,(0,1))
	return a11*(a33*a22-a32*a23) - a21*(a33*a12-a32*a13) + a31*(a23*a12-a22*a13)

def inv3(a, axes=(0,1), sym=False, return_det=False):
	"""Inverse of 3x3 matrices. Much faster than np.linalg.inv"""
	ap   = device.anypy(a)
	res  = ap.zeros_like(a)
	wres = ap.moveaxis(res,axes,(0,1))
	(a11, a12, a13),(a21,a22,a23),(a31,a32,a33) = ap.moveaxis(a,axes,(0,1))
	if sym:
		wres[0,0] =  a22*a33-a23*a23; wres[0,1] = -a12*a33+a13*a23; wres[0,2] =  a12*a23-a13*a22
		wres[1,0] =        wres[0,1]; wres[1,1] =  a11*a33-a13*a13; wres[1,2] = -a11*a23+a12*a13
		wres[2,0] =        wres[0,2]; wres[2,1] =        wres[1,2]; wres[2,2] =  a11*a22-a12*a12
		det = a11*wres[0,0] + a12*wres[0,1] + a13*wres[0,2]
	else:
		wres[0,0] =  a33*a22-a32*a23; wres[0,1] = -a33*a12+a32*a13; wres[0,2] =  a23*a12-a22*a13
		wres[1,0] = -a33*a21+a31*a23; wres[1,1] =  a33*a11-a31*a13; wres[1,2] = -a23*a11+a21*a13
		wres[2,0] =  a32*a21-a31*a22; wres[2,1] = -a32*a11+a31*a12; wres[2,2] =  a22*a11-a21*a12
		det = a11*wres[0,0] + a21*wres[0,1] + a31*wres[0,2]
	wres /= det
	if return_det: return res, det
	else: return res

def safe_inv(a):
	ap = device.anypy(a)
	with utils.nowarn():
		res = 1/a
		res[~ap.isfinite(res)] = 0
	return res

def safe_invert_ivar(ivar, tol=1e-6):
	ap   = device.anypy(ivar)
	vals = ivar[ivar!=0]
	ref  = ap.mean(vals[::100])
	iivar= ivar*0
	good = ivar>ref*tol
	iivar[good] = 1/ivar[good]
	return iivar

# Runs at 0.45 ms per tile on the cpu and 0.03 ms per tile on the gpu.
# This is much faster than calling the standard linear algebra functions,
# but still quite slow. The full LAT area has 100k tiles, which would take
# 45 s on the cpu. Of course, a single mpi task is unlikely to have such
# a big area, so this is good enough for now.
def safe_invert_div(div, tol=1e-3):
	ap   = device.anypy(div)
	wdiv = ap.array(ap.moveaxis(div, (-4,-3), (0,1)), order="C")
	with utils.nowarn():
		# inv3 involves the cube of TT, which can easily overflow, so we must start by
		# dividing out a typical value. We do this even if norm is zero, which is not
		# rare. The reason is that setting up masking is slower than just setting the
		# nans to zero in the end, which has the same effect
		norm = wdiv[0,0].copy()
		wdiv /= norm
		# Ok, we now have a normalized matrix ready to invert
		inv, det = inv3(wdiv, axes=(0,1), sym=True, return_det=True)
		# Check if any of the matrices are poorly conditioned.
		# A maximally healthy div is diag(1,0.5,0.5) after norm,
		# so its det should be 0.25. We will be inverting the determinant,
		# so we're in trouble if it's too small. So we will use det << 0.25
		# as a stand-in for the condition number
		bad = det < 0.25*tol
		inv[:,:,bad] = 0
		inv[0,0,bad] = 1/wdiv[0,0,bad]
		# Undo the normalization
		inv /= norm
		# Reshape to original ordering
		odiv = ap.moveaxis(inv, (0,1), (-4,-3))
		ap.nan_to_num(odiv, copy=False, nan=0, posinf=0, neginf=0)
	return odiv

def safe_invert_prec(prec):
	if   prec.ndim == 3: return safe_invert_ivar(prec)
	elif prec.ndim == 5: return safe_invert_div (prec)
	else: raise ValueError("prec must be [ntile,ty,tx] (ivar form) or [ntile,3,3,ty,tx] (div form)")

def apply_prec(prec, gmap):
	ap = device.anypy(prec)
	if   prec.ndim == 3: return prec[:,None]*gmap
	elif prec.ndim == 5: return ap.einsum("tabyx,tbyx->tayx", prec, gmap)
	else: raise ValueError("prec must be [ntile,ty,tx] (ivar form) or [ntile,3,3,ty,tx] (div form)")

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

def obs_group_size(obsinfo, groups, inds=None, sampranges=None):
	if inds is None: inds = np.arange(len(groups))
	if sampranges is None:
		return np.array([np.sum(obsinfo.ndet[groups[i]]*obsinfo.nsamp[groups[i]]) for i in inds])
	else:
		nsamps = sampranges[:,1]-sampranges[:,0]
		return np.array([np.sum(obsinfo.ndet[groups[i]])*nsamps[i] for i in inds])

def obs_group_dur(obsinfo, groups, inds=None, sampranges=None):
	if inds is None: inds = np.arange(len(groups))
	# duration from first member of group. This assumes that a group doesn't
	# contain multiple time-ranges
	durs = np.array([obsinfo.dur[groups[i][0]] for i in inds])
	if sampranges is not None:
		nsamps = sampranges[:,1]-sampranges[:,0]
		tfracs = np.array([nsamps[i]/obsinfo.nsamp[groups[i][0]] for i in inds])
		durs  *= tfracs
	return durs

def time_split(obsinfo, joint, maxsize=None, maxdur=None):
	"""Split obs groups defined by joint.{groups,names,bands,sampranges,joint}
	into subranges so that the total ndet*nsamp size of each is no larger than
	maxsize. This can be needed due to memory constraints."""
	nsplits = np.ones(len(joint.groups), int)
	if maxsize is not None:
		# calculate the total size of each group
		sizes  = obs_group_size(obsinfo, joint.groups, sampranges=joint.sampranges)
		# number of time splits for each
		nsplits= np.maximum(nsplits, utils.floor(sizes/maxsize)+1)
	if maxdur is not None:
		durs   = obs_group_dur(obsinfo, joint.groups, sampranges=joint.sampranges)
		nsplits= np.maximum(nsplits, utils.floor(durs/maxdur)+1)
	# Do the split
	ojoint = bunch.Bunch(groups=[], names=[], sampranges=[],
		bands=joint.bands, nullbands=joint.nullbands, joint=joint.joint)
	for gi, (group, name, nsplit) in enumerate(zip(joint.groups, joint.names, nsplits)):
		# Need the number of samples the group covers. Will assume good time alignment
		if joint.sampranges is None:
			i0    = 0
			nsamp = np.max(obsinfo.nsamp[group])
		else:
			i0    = joint.sampranges[gi,0]
			nsamp = joint.sampranges[gi,1]-joint.sampranges[gi,0]
		for si in range(nsplit):
			i1 = i0 + si*nsamp//nsplit
			i2 = i0 + (si+1)*nsamp//nsplit
			oname = "%s:split%d" % (name,si)
			ojoint.groups.append(group)
			ojoint.names.append(oname)
			ojoint.sampranges.append((i1,i2))
	ojoint.sampranges = np.array(ojoint.sampranges)
	return ojoint

def select_groups(joint, inds):
	return bunch.Bunch(
		groups = [joint.groups[i] for i in inds],
		names  = [joint.names[i]  for i in inds],
		sampranges=np.array([joint.sampranges[i] for i in inds]),
		bands=joint.bands, nullbands=joint.nullbands, joint=joint.joint)

def alloc_rfft(ishape, idtype, axes=[-1], dev=None, pool=None):
	if dev  is None: dev  = device.get_device()
	if pool is None: pool = dev.np
	ctype  = np.result_type(idtype, 0j)
	oshape = fft.rfft_shape(ishape, axes=axes)
	ft     = pool.empty(oshape, ctype)
	return ft

def alloc_irfft(ishape, idtype,  axes=[-1], n=None, dev=None, pool=None):
	if dev  is None: dev  = device.get_device()
	if pool is None: pool = dev.np
	rtype  = np.zeros(1, idtype).real.dtype
	oshape = fft.irfft_shape(ishape, axes=axes, n=n)
	tod    = pool.empty(oshape, rtype)
	return tod

def fourier_resample(itod, otod, normalize=False, nofft=False, ipool=None, opool=None, dev=None):
	"""Fourier resample itod into otod along the last axis.
	If normalize=True, then the output will be normalized by dividing by
	itod.shape[-1]."""
	if dev is None: dev = device.get_device()
	norm = itod.shape[-1]
	# Go to fourier space
	if nofft:
		ift, oft = itod, otod
	else:
		ift = alloc_rfft(itod.shape, itod.dtype, pool=ipool, dev=dev)
		oft = alloc_rfft(otod.shape, otod.dtype, pool=opool, dev=dev)
		dev.lib.rfft(itod, ift)
		if normalize and itod.shape[-1] < otod.shape[-1]:
			# Chepeast to do the division here
			ift /= norm
	# Translate fourier spaces
	n = min(ift.shape[-1], oft.shape[-1])
	oft[...,:n] = ift[...,:n]
	oft[...,n:] = 0
	# Transform back
	if not nofft:
		dev.lib.irfft(oft, otod)
		if normalize and otod.shape[-1] < itod.shape[-1]:
			# Chepeast to do the division here
			otod /= norm
	return otod

def jacobi(tod, op, iop=lambda x:x, niter=4):
	res = op(tod)
	for it in range(niter):
		res += op(tod-iop(res))
	return res

def lowpass(tod, fknee, alpha=-8, srate=1, dev=None, pool=None, inplace=False):
	if dev is None: dev = device.get_device()
	if not inplace: tod = tod.copy()
	ft  = alloc_rfft(tod.shape, tod.dtype, dev=dev, pool=pool)
	dev.lib.rfft(tod, ft)
	f   = dev.np.fft.rfftfreq(tod.shape[-1], 1/srate)
	with utils.nowarn():
		flt  = (1+(f/fknee)**alpha)**-1
	ft *= flt / tod.shape[-1]
	dev.lib.irfft(ft, tod)
	return tod

def svd_filter(tod, eiglim=1e-3, dev=None):
	if dev is None: dev = device.get_device()
	U, S, Vh = dev.np.linalg.svd(tod, full_matrices=False)
	good     = S>np.max(S)*eiglim
	return (U[:,good]*S[good]).dot(Vh[good])

def pca1(tod, tol=None, maxiter=1000, verbose=False):
	"""Extract the largest left principal component of tod"""
	ap = device.anypy(tod)
	if tol is None:
		tol = np.finfo(tod.dtype).resolution
	def norm(x): return ap.sum(x**2)**.5
	def normalize(x): return x/norm(x)
	r = normalize(ap.random.randn(tod.shape[0]).astype(tod.dtype))
	for i in range(maxiter):
		s = tod.dot(tod.T.dot(r))
		λ = ap.dot(r,s)
		err = norm(r-s/λ)
		r = normalize(s)
		if verbose:
			print("%4d %15.7e" % (i+1, err))
		if err <= tol: break
	# Want largest element positive to get consistent sign
	r *= ap.sign(r[ap.argmax(ap.abs(r))])
	return r

# Ideas tested:
# * constrained solution: Slow, memory-expensive. Needs good preconditioenr.
#   Had overflow issues (fixable)
# * svd-filter + per-detector ps interpolation: Slow interpolation, but more
#   importantly, visual inspection showed that the signal was simply not
#   that predictable. Two different interpolations both looked sensible, but
#   differed by as much as the linear gapfilling did.
# * plain common mode. Faster than svd, but at least for multi-band had much
#   bigger residuals. Didn't test that thoroughly. svd fast enough on gpu.
# Ended up with just the simple svd. Currently uses two rounds, but
# only to be extra safe. 1 seems almost indistinguishable.

def estimate_atm_tod(tod, cut=None, srate=400, fmax=10, fmin=0.05, niter=None, eiglim=1e-5,
		dev=None, pool=None):
	"""Given a tod, overwrite it with an estimate of its smooth atmospheric signal.
	This can be used in e.g. atmospheric subtraction. The atmosheric estimation consists
	of removing fourier-fmodes abofe fmax and removing eigenmodes below eiglim.
	If cut is passed, then the atmospheric estimate will be independent of the signal
	in the cut area, which can be useful for planet mapping for example.

	FIXME: This function produces a sensible-looking tod, but to the noise model
	its still very different from the actual data, resulting in strong discontinuities
	when used to gapfill. As it is, it therefore isn't usable for gapfilling.
	"""
	if dev is None: dev = device.get_device()
	if cut is None:
		from . import socut
		cut   = socut.Simplecut(ndet=tod.shape[0], nsamp=tod.shape[1])
	if niter is None:
		# No need to do iteration if we don't have any cuts
		niter = 1 if cut.nrange == 0 else 2
	ndet, nsamp = tod.shape
	# 1. Start with simple gapfilling, so we don't have potentially huge values
	#    messing things up
	cut.gapfill(tod)
	# 2. Downsample to target fmax. This not only greatly reduces our
	#    resource requirements, it also protects us from whtie noise leakage
	n     = utils.floor(tod.shape[1]*fmax/(srate/2))
	dtod  = dev.np.zeros((ndet,n), tod.dtype)
	fourier_resample(tod, dtod, ipool=pool, dev=dev, normalize=True)
	dcut  = cut.resample(n)
	# Iterate over model estimation and gapfilling
	model = None
	for it in range(niter):
		# Redo gapfilling relative to the best model so far
		if model is not None:
			dtod = dcut.gapfill(dtod-model)+model
		work = dtod.copy()
		# 3. Factor out super-slow modes, which we may be unrepresentative.
		model   = lowpass(work, fmin, srate=fmax, dev=dev)
		work   -= model
		# 4. Subtract svd
		dsvd   = svd_filter(work, eiglim=eiglim, dev=dev)
		model += dsvd
		del dsvd
	# Resample back to our original resolution, overwriting our input argument
	fourier_resample(model, tod, opool=pool, dev=dev, normalize=True)
	return dtod

def gapfill_atm(tod, cut, srate=400, fmax=10, fmin=0.05, niter=None, eiglim=1e-5,
		dev=None, fpool=None, mpool=None):
	"""FIXME: This function isn't usable as a gapfiller for normal mapmaking. While
	the output looks sensible by eye, the noise model thinks otherwise."""
	if dev   is None: dev = device.get_device()
	if mpool is None: mpool = dev.np
	# Model the atmosphere
	model = mpool.array(tod)
	estimate_atm_tod(model, cut, srate=srate, fmax=fmax, fmin=fmin, niter=niter,
		eiglim=eiglim, dev=dev, pool=fpool)
	# Use it to gapfill
	vals = cut.extract(model)
	cut.insert(tod, vals)
	return tod

def gapfill_extreme(tod, cut, tol=100, dstep=10, step=100, dev=None):
	"""Remove extreme values in the cut regions, leaving normal samples untouched.
	The idea is that gapfilling only is needed to keep non-local operations like
	noise model estimation sane. A gapfiller must do less harm by distorting the
	noise than one does by letting through glitches. As long as one can't generate
	realistic enough noise, it makes sense to leave almost everything untouched"""
	if dev   is None: dev = device.get_device()
	dcut = cut.to_device(dev)
	# extract the original values
	vals = dcut.extract(tod)
	# get gapfilled values, which we will use temporarily when
	# juding which values are extreme. The original values will
	# be restored later
	dcut.gapfill(tod)
	bg    = dcut.extract(tod)
	vals -= bg
	# estimate the mid-scale rms from the whole tod
	rms   = dev.np.median(dev.np.diff(tod[::dstep,::step])**2)**0.5
	# could do something smooth with the extreme values, but should be
	# practically as good to just clip
	dev.np.clip(vals, -rms*tol, rms*tol, out=vals)
	# restore background and reinsert
	vals += bg
	dcut.insert(tod, vals)
	# done!
	return tod

#def pca1_filter(tod, dev=None):
#	if dev is None: dev = device.get_device()
#	V = pca1(tod)
#	return V[:,None]*V.dot(tod)
#
#def mask_wiener(tod, ips, mask, mtol=0.01, cgtol=1e-8, maxiter=100, dev=None):
#	if dev is None: dev = device.get_device()
#	norm = np.max(ips)**0.5
#	iS   = ips/norm**2
#	d    = tod/norm
#	iM   = mask*(norm/mtol)
#	prec = 1/dev.np.max(iM)
#	# Set up our CG solver. We model the data as
#	# tod = s + n, where icov(s) = ips and icov(n) = N",
#	# N" = 0 in masked region and big elsewhere.
#	# Then (S"+N")ŝ = N"tod
#	def zip(tod): return tod.reshape(-1)
#	def unzip(x): return x.reshape(tod.shape)
#	rhs = zip(iM*d)
#	def A(x):
#		x = unzip(x)
#		res = dev.lib.irfft(iS*dev.lib.rfft(x), n=x.shape[-1])/x.shape[-1] + iM*x
#		return zip(res)
#	def precon(x): return x*prec
#	solver = utils.CG(A, rhs, M=precon)
#	while solver.i < maxiter and solver.err > cgtol:
#		solver.step()
#		print("CG %4d %15.7e" % (solver.i, solver.err))
#	res  = unzip(solver.x)
#	res *= norm
#	return res

# Could model data as d[d,f] = V[d,m]*e[m,f] + u[d,f],
# where var(e) = E[m,f] and var(u) = U[d,f]. Want to
# solve for e and u, masked. Rewrite as d = [V 1][e;u] = Px
# P(x|d) = P(d|x) * P(x)
# -2logP = (d-Px)'M"(d-Px) + x'X"x
#        = (x-(P'M"P+X")"P'M"d)'(P'M"P+X")(x-...)
# So (P'M"P+X")x = P'M"d

#def robust_atm_cov(tod, down=200, nmed=10, high=5, dev=None):
#	"""Estimate the atmospheric covariance using a real-space
#	median of means approach. We effecitvely bandpass filter
#	the tod by first downgrading until we reach the atmospheric
#	regime around 1 Hz, and then do a diference samples high
#	apart to highpass filter."""
#	# TODO: Consider replacing median with a masking of cut values,
#	# since the median takes up >90% of the runtime of the function
#	from pixell import bench
#	if dev is None: dev = device.get_device()
#	dtod   = downgrade(tod, down)
#	nper   = dtod.shape[1]//nmed
#	blocks = dtod[:,:nmed*nper].reshape(-1,nmed,nper)
#	if blocks.shape[2] > high:
#		blocks = blocks[:,:,high:]-blocks[:,:,:-high]
#	else:
#		blocks-= dev.np.mean(blocks,-1)[:,:,None]
#	# Calculate the covariance in each block. Shouldn't be worse
#	# than about 10 MB or so
#	cov    = dev.np.einsum("dbs,ebs->deb", blocks, blocks)
#	# Median across blocks, to avoid outliers. This is the slow step.
#	# Around 300 ms on gpu
#	cov    = dev.np.median(cov,-1)
#	return cov
#
#def top_eigs(cov, eig_lim=1e-3, dev=None):
#	# This is surprisingly slow, but the matrix is
#	# quite large
#	if dev is None: dev = device.get_device()
#	E, V = dev.np.linalg.eigh(cov)
#	good = np.where(E>np.max(E)*eig_lim)[0]
#	E, V = E[good], V[:,good]
#	return E, V
#
#def gapfill_eigvecs(tod, pcut, V, E, prior=1e-6, sub_lim=0.25, dev=None):
#	"""Gapfill cut regions of tod by modelling the tod-signal
#	as detector-correlated with covmat VEV' per sample, with
#	samples uncorrelated. The uncut detectors are used to predict
#	the cut samples. The solution is biased towards zero with
#	strength prior/E, which mainly matters when so many detectors
#	are cut that the per-sample system becomes degenerate.
#
#	pcut is an instance of PcutFull.
#
#	Uses the buffer "pointing"
#	"""
#	if dev is None: dev = device.get_device()
#	# Find which samples we need to concern ourselves with
#	mask = dev.pools["ft"].ones(tod.shape, tod.dtype)
#	pcut.clear(tod)
#	pcut.clear(mask)
#	with dev.pools["pointing"].as_allocator():
#		ngood = dev.np.sum(mask,0)
#		sel   = dev.np.where(ngood<tod.shape[0])[0]
#		do_sub = len(sel) < tod.shape[1]*sub_lim
#		if do_sub:
#			# Extract the relevant parts. Hopefully very few samples
#			wtod, wmask = tod[:,sel], mask[:,sel]
#		else:
#			wtod, wmask = tod, mask
#		# Build our equation system
#		rhs  = wtod.T.dot(V)
#		# Sadly, this is split into pairwise operations internally
#		# by cupy.einsum, which results in a temporary of size ndet*nmode*nsamp
#		# being allocated. That defeats the whole point of calling einsum
#		# in the first place!
#		div  = dev.np.einsum("da,ds,db->sab", V, wmask, V)
#		del wtod, wmask
#		# add prior
#		Q    = dev.np.max(E)/E*prior
#		for i, q in enumerate(Q):
#			div[:,i,i] += q
#		# now solve per element. Sadly solve doesn't have an output
#		# argument, so this will allocate
#		amps = dev.np.linalg.solve(div,rhs[:,:,None])[:,:,0] # [nsamp,nmode]
#		del rhs,div
#		# Build our model
#		model = dev.np.einsum("sa,da->ds", amps, V)
#		if do_sub:
#			# Copy over needed parts to tod
#			mask[:,sel] = model
#		else:
#			mask = model
#		vals = dev.np.zeros(pcut.nsamp, tod.dtype)
#	pcut.backward(mask, vals)
#	pcut.forward(tod, vals)
#
#def gapfill_eigvecs_iterative(tod, pcut, V, niter=4, dev=None):
#	# This function would be cleaner and faster if we had pgood instead of pcut
#	if dev is None: dev = device.get_device()
#	work = dev.pools["ft"].empty(tod.shape, tod.dtype)
#	for it in range(niter):
#		amps = V.T.dot(tod)
#		dev.np.dot(V, amps, out=work)
#		# extract values in cut region from work, and insert into tod
#		vals = pcut.backward(work)
#		pcut.forward(tod, vals)
#
## This was supposed to be a faster and simpler version of the ACT
## atm gapfiller, but it's not working well at all here. It leaves
## sharp edges and in general isn't a good guess.
##
## What could be going wrong?
## 1. Our plain gapfilling is different here. In ACT we did linear
##    interpolation, while sogma does dejumping. Maybe doing dejumping
##    only for the actual jumps would be better, and having linear
##    interpolation for the rest.
## 2. ACT built the cov from the full frequency range, diluting the
##    atm with some white noise. What we're doing here should be better.
## 3. We have more detectors. That shouldn't be a problem, but
##    in my tests, restricting to just 500 dets results in much nicer
##    gapfilling.
#def gapfill_atm(tod, cuts, eig_lim=1e-3, niter=4, dev=None):
#	from . import pmat
#
#	tod0 = tod.copy()
#
#	pcut = pmat.PmatCutFull(cuts, dev=dev)
#	cov  = robust_atm_cov(tod, dev=dev)
#	E, V = top_eigs(cov, eig_lim=eig_lim, dev=dev)
#	#gapfill_eigvecs(tod, pcut, V, E, dev=dev)
#	gapfill_eigvecs_iterative(tod, pcut, V, niter=niter, dev=dev)
#
#	model = downgrade(tod,  10)
#	tod   = downgrade(tod0, 10)
#	pcut = pmat.PmatCutFull(cuts, dev=dev)
#	mask   = dev.np.ones_like(tod0)
#	pcut.clear(mask)
#	mask  = downgrade(mask, 10)
#
#	bunch.write("test.hdf", bunch.Bunch(
#		tod=dev.get(tod), model=dev.get(model), cov=dev.get(cov), E=dev.get(E), V=dev.get(V),
#		mask=dev.get(mask),
#		cuts=bunch.Bunch(starts=cuts.starts//10, dets=cuts.dets, lens=(cuts.lens+9)//10)))
#	#1/0
#
## assume that both iN and N destroy bufs tod and ft
## Argh, lots of trouble with this function too.
## 1. N and iN are not good inverses of each other.
##    Due to windowing? Or a bug?
## 2. I'm getting overflow very quickly, even with very mild q.
##    Both cg and jacobi
## 3. This really shouldn't be hard to get to work, but should maybe
##    try another idea first: First do a gap-safe bandpass filter to
##    isolate the coarse atmosphere. Subtract this as a first approximation.
##    Then low-pass filter result to get the rest of the atmosphere.
##    Build covmat of this, then use it to eigen-gapfill.
##    Then add back in the coarse atmosphere. I think this should be a pretty
##    good procedure, and if it's all done downsampled then it would be fast
##    and low-memory. The final model would then be interpolated to the original
##    resolution. Doing it this way also means we don't need a separate smoothing
##    step.
#def gapfill_nmat(tod, cuts, iN, N, q=1, solver="jacobi", niter=10, strict=False, dev=None):
#	# Solve for the maximum-likelihood values in the cut region,
#	# given the rest of the data. This is just a wiener filter
#	#  (N"+M")x = M"d, M" = inf in unmasked, 0 in masked
#	# Can't actually make it inf, of course. Instead set it to
#	# q*max(N"). Can then solve with CG, where prec = N. Or use
#	# simple jacobi iteration
#	from . import pmat
#	if dev is None: dev = device.get_device()
#	if strict: backup = tod.copy()
#	pcut = pmat.PmatCutFull(cuts, dev=dev)
#	ref  = dev.np.mean(iN.ivar)
#	print("solver", solver)
#	print("A", ref)
#	def iM(x):
#		x *= ref*q
#		pcut.clear(x)
#		return x
#	# This function needs a lot of memory for tod-sized buffers, and
#	# there aren't many free. We'll worry about that later, but may need
#	# to this downsampled in the end
#	rhs  = iM(tod.copy())
#	print("B", dev.np.std(rhs))
#	def A(x):
#		tmp = iM(x.copy())
#		iN.apply(x)
#		x += tmp
#		return x
#	def precon(x):
#		return N.apply(x)
#	if solver == "jacobi":
#		# x    = precon(rhs)
#		# rhs -= A(x)
#		# x   += precon(rhs) ...
#		# Bleh, this buffer juggling is tedious!
#		x = precon(rhs.copy())
#		print("C", dev.np.std(x))
#		for it in range(1,niter):
#			rhs -= A(x.copy())
#			print("D", dev.np.std(rhs))
#			x += precon(rhs.copy())
#			print("E", dev.np.std(x))
#	elif solver == "cg":
#		def zip(tod): return tod.reshape(-1)
#		def unzip(x): return x.reshape(tod.shape)
#		def fA(x): return zip(A(unzip(x)))
#		def myprec(x): return x/ref
#		cg = utils.CG(A=fA, b=zip(rhs), M=myprec, dot=dev.np.dot)
#		while cg.i < niter:
#			print("moo %4d %15.7e" % (cg.i, cg.err))
#			cg.step()
#		x = unzip(cg.x)
#	else: raise ValueError(solver)
#	# copy target part out
#	if strict:
#		tod[:] = backup
#		vals = pcut.backward(x)
#		pcut.forward(tod, vals)
#	else:
#		tod[:] = x
#	return tod

def robust_mean(arr, axis=-1, quantile=0.1):
	ap  = device.anypy(arr)
	axis= axis%arr.ndim
	arr = ap.sort(arr, axis=axis)
	n   = utils.nint(arr.shape[axis]*quantile)
	arr = arr[(slice(None),)*axis+(slice(n,-n),)]
	mean= ap.mean(arr, axis)
	std = ap.std(arr, axis)
	ngood = arr.shape[axis]
	err = std/ngood**0.5
	return mean, err, ngood
