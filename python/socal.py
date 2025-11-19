import numpy as np, os, contextlib
from pixell import config, utils, enmap, ephem, pointsrcs, coordsys, bunch, colors
from . import pmat, tiling, device, socut, gutils, soeph
from .logging import L

config.default("autocut",       "objects,sidelobes", "Comma-separated list of which autocuts to apply. Currently recognized: objects: The object cut. sidelobes: The sidelobe cut")
config.default("object_cut",    "planets:10,asteroids:5")
# planetas and asteroids defined in sogma.soeph

def autocut(obs, id="?", which=None, geo=None, dev=None):
	"""Main driver. Performs all autocuts, merges them with any existing cuts,
	updates obs and gpfills tod"""
	if dev is None: dev = device.get_device()
	which   = config.get("autocut", which)
	if which == "none" or len(which) == 0: return obs
	which   = which.split(",")
	cutfuns = {"objects": object_cut, "sidelobes": sidelobe_cut}
	cuts = [obs.cuts]
	for i, cutname in enumerate(which):
		cuts += cutfuns[cutname](obs, id=id, geo=geo, dev=dev)
	cuts = socut.Simplecut.merge(cuts)
	#cuts.gapfill(obs.tod, dev=dev)
	obs.cuts = cuts
	return obs

# TODO: This function takes around 1 sec, dominated by ephem_map+lmap+pmap
# Could potentially speed up by using pmap.backward to figure out which tiles
# are hit, instead of building a fullsky map.
def object_cut(obs, id="?", object_list=None, geo=None, down=8, base_res=0.5*utils.arcmin,
		dt=100, dr=1*utils.arcsec, dev=None):
	if dev is None: dev = device.get_device()
	object_list = get_object_list(object_list)
	# Set up a low-resolution geometry, either by downgrading a given geometry
	# of by downgrading a fullsky geometry with the base_res resolution.
	# The advantage of passing an existing geometry is that one avoids partially
	# cut pixels, which can have very high noise.
	shape, wcs = geo if geo is not None else enmap.fullsky_geometry2(res=base_res)
	shape, wcs = enmap.downgrade_geometry(shape, wcs, down)
	# Get our time range
	t1, t2 = utils.minmax(obs.ctime)
	map  = ephem_map(shape, wcs, object_list, [t1,t2], dt=dt, dr=dr)
	# map2tod. ft buffer and the pointing buffers are free
	lmap = tiling.enmap2lmap(map, dev=dev)
	pmap = pmat.PmatMap(lmap.fshape, lmap.fwcs, obs.ctime, obs.boresight, obs.point_offset, obs.polangle, site=obs.site, partial=True, dev=dev)
	tod  = dev.pools["ft"].zeros(obs.tod.shape, obs.tod.dtype)
	pmap.forward(tod, lmap.lmap)
	# turn non-zero values into a cuts object. Done with pmap here, so can use pointing pool
	cuts = mask2cut(tod, dev=dev, pool=dev.pools["pointing"])
	# return result as length-1 list of cuts
	# It's a list of cuts because other cut types can return multiple cuts
	# User must merge with other cuts and gapfill as necessary
	return [cuts]

config.default("sun_mask", "/global/cfs/cdirs/sobs/users/sigurdkn/masks/sidelobe/sun.fits", "Location of Sun sidelobe mask")
config.default("moon_mask", "/global/cfs/cdirs/sobs/users/sigurdkn/masks/sidelobe/moon.fits", "Location of Sun sidelobe mask")
sidelobe_cutters = {}
def sidelobe_cut(obs, id="?", object_list=None, geo=None, dev=None):
	if object_list is None: object_list = ["sun", "moon"]
	cutss = []
	for name in object_list:
		if name not in sidelobe_cutters:
			fname = config.get(name + "_mask")
			if not fname: raise ValueError("config setting %s_mask missing for sidelobe cut" % name)
			mask  = enmap.read_map(fname)
			sidelobe_cutters[name] = SidelobeCutter(mask, objname=name, dtype=obs.tod.dtype, dev=dev)
		cutter = sidelobe_cutters[name]
		cuts   = cutter.make_cuts(obs, id=id)
		cutss.append(cuts)
	return cutss

class SidelobeCutter:
	def __init__(self, mask, objname, dtype=np.float32, rise_tol=1*utils.degree, dist_tol=5*utils.degree, dev=None):
		self.dev     = dev or device.get_device()
		self.mask    = mask
		self.distmap = enmap.distance_transform(mask<1).astype(dtype)
		self.lmap    = tiling.enmap2lmap(mask.astype(dtype), dev=self.dev)
		self.objname = objname
		self.sys  = "hor,on=%s" % objname
		self.rise_tol = rise_tol
		self.dist_tol = dist_tol
	def make_cuts(self, obs, id="?"):
		# First check if the object is above the horizon. We just check the endpoints
		# to keep things simple
		ts    = obs.ctime[[0,-1]]
		hor   = coordsys.transform(self.sys, "hor", coordsys.Coords(ra=0, dec=0), ctime=ts, site=obs.site)
		risen = np.any(hor.el > self.rise_tol)
		if not risen:
			return socut.Simplecut(ndet=obs.tod.shape[0], nsamp=obs.tod.shape[1])
		# Ok, it's above the horizon. Estimate if we get close enough to the mask to bother
		mindist = estimate_mindist(obs, self.distmap, sys=self.sys, step=self.dist_tol)
		if mindist > self.dist_tol:
			return socut.Simplecut(ndet=obs.tod.shape[0], nsamp=obs.tod.shape[1])
		# Ok, looks like we get close enough that we must do this properly
		try:
			pmap = pmat.PmatMap(self.lmap.fshape, self.lmap.fwcs, obs.ctime,
				obs.boresight, obs.point_offset, obs.polangle, sys=self.sys, site=obs.site,
				partial=True, dev=self.dev)
			tod  = self.dev.pools["ft"].zeros(obs.tod.shape, obs.tod.dtype)
			pmap.forward(tod, self.lmap.lmap)
		except RuntimeError:
			# This happens when the interpolated pointing ends up outside the -npix:+2npix range
			# that gpu_mm allows. This can happen when a detector moves too close to the north pole
			# in the object-centered coordinates, which in CAR makes it seem to teleport by 180
			# degrees. When combined with unwinding, some jumps being sligthly less than 180 and
			# some slightly more than 180 can lead to the numbers drifting over a large range.
			# It might be better to catch this by just looking for values too close to the poles
			# explicitly instead of relying on gpu_mm to catch things itself
			L.print("Error cutting %s sidelobes for %s: Pointing overflow. Skipping cut" % (self.objname, id), level=2, color=colors.red)
			return socut.Simplecut(ndet=obs.tod.shape[0], nsamp=obs.tod.shape[1])
		# turn non-zero values into a cuts object. Done with pmap here, so can
		# use pointing pool
		cuts = mask2cut(tod, dev=self.dev, pool=self.dev.pools["pointing"])
		return cuts

def estimate_mindist(obs, distmap, sys="equ", step=1*utils.degree):
	# Define the spare points we will sample the boresight path at
	az1, az2 = utils.minmax(obs.boresight[1,::100])
	t1,  t2  = obs.ctime[[0,-1]]
	el       = obs.boresight[0,0] # assume (near) constant-el
	tstep    = step / (15*utils.degree/utils.hour)
	naz      = utils.ceil((az2-az1)/step)+1
	nt       = utils.ceil((t2-t1)/tstep)+1
	azs      = np.linspace(az1, az2, naz)
	ts       = np.linspace(t1,  t2,  nt)
	# Measure fplane radius
	frad     = np.max(np.sum(obs.point_offset**2,1))**0.5
	# Transform these to the target coordinate system
	icoords  = coordsys.Coords(az=azs[None,:], el=el)
	ocoords  = coordsys.transform("hor", sys, icoords, ctime=ts[:,None], site=obs.site)
	# Read off the map
	opos     = np.array([ocoords.dec, ocoords.ra]).reshape(2,-1)
	dist     = distmap.at(opos, order=0, cval=np.inf)
	mindist  = max(np.min(dist)-frad,0)
	return mindist

def autocal(obs, prefix=None, dev=None):
	# Are we doing elmod calibration?
	if elmod_cal(obs, prefix=prefix, dev=dev): return True
	if cmod_cal (obs, prefix=prefix, dev=dev): return True
	return False

config.default("cmod_cal", "none", "Comma-separated list of what to do with the common-mode gain fit. If it contains 'cal', then it will be used to flatfield the tod. If it contains 'dump', then the fit will be dumped to individual files. If it contains neither of these, then no common mode fit will be performed. Common mode calibration will not be done if elevation-modulation calibration is enabled and applicable")
def cmod_cal(obs, fmin=0.05, fmax=3, tol=1e-4, prefix=None, dev=None):
	"""Do elevation-modulation auto-calibration if elevation is actually modulated (determined by
	minamp), and config:elmod_cal has turned this on. Otherwise does nothing"""
	tasks = config.get("cmod_cal").split(",")
	if not ("cal" in tasks or "dump" in tasks or "tdump" in tasks): return False
	# Ok, do a fit
	try:
		fit = measure_cmode_response(obs, fmin=fmin, fmax=fmax, tol=tol, dev=dev)
		if "cal" in tasks:
			obs.tod *= dev.np.array(fit.gain)[:,None]
		# Prefix required if dumping!
		if "dump" in tasks:
			dump_cmod_fit(prefix + "ccal.txt", fit)
		if "tdump" in tasks:
			dump_cmod_tfit(prefix + "tccal.txt", fit)
	except ():
		return False
	return "cal" in tasks

# Status: I've tried measuring this several ways:
# * Fourier-bandpass + plain pca
# * Fourier-bandpass + bunched pca median
# * Group-downsampling + bunched pca median
# All produce sensible-looking results, but they're also
# quite different from each other. All have a few outliers,
# but often different ones. Overall I'm not satisfied with
# the quality, so I'll disable this for now
def measure_cmode_response(obs, fmin=0.01, fmax=3, tol=1e-4, dev=None):
	dev = device.get_device()
	ndet, nsamp = obs.tod.shape
	srate  = (len(obs.ctime)-1)/(obs.ctime[-1]-obs.ctime[0])
	bsize  = utils.nint(srate/2/fmax)
	gsize  = utils.nint(fmax/fmin)
	# Downgrade by bsize to get us out of the white noise regime,
	# and demean by gsize to restrict to the higher parts of the
	# atmospheric modes, where we have more staticits and are less
	# likely to be impacted by other types of picup
	ngroup = nsamp//(bsize*gsize)
	vals   = obs.tod[:,:ngroup*gsize*bsize].reshape(ndet,ngroup,gsize,bsize)
	vals   = dev.np.mean(vals,-1)
	vals  -= dev.np.mean(vals,-1)[:,:,None]
	# Do commmon mode calibration in each. Yuck, looping.
	# Won't be gpu-efficient. Consider writing a multi-pca.
	# Simply reshaping didn't work
	ba = np.zeros((ngroup, ndet), obs.tod.dtype)
	for gi in range(ngroup):
		ba[gi] = dev.get(gutils.pca1(vals[:,gi], tol=tol))
	# each row of ba will have arbitrary sign. Normalize that out
	# using an average
	a, da, n = gutils.robust_mean(ba, 0, quantile=0.25)
	ba /= (np.sum(ba*a,1)/np.sum(a*a))[:,None]
	# robust mean over groups, to reduce the impact of outliers like tunarounds
	# or planets
	a, da, n = gutils.robust_mean(ba, 0, quantile=0.25)
	ba /= np.median(a)
	# flatfield per band, since the atm response should be freq-dependent
	gain = np.full(ndet, 1, obs.tod.dtype)
	ubands, order, edges = utils.find_equal_groups_fast(obs.bands)
	for bi, uband in enumerate(ubands):
		inds       = order[edges[bi]:edges[bi+1]]
		band_a     = a[inds]
		gain[inds] = np.median(band_a)/band_a
	# This is only used for the dump file format
	xieta= obs.point_offset[:,1::-1]
	return bunch.Bunch(a=a, da=da, n=n, gain=gain, ba=ba,
		bands=obs.bands, detids=obs.detids, xieta=xieta)

def dump_cmod_fit(fname, fit):
	with open(fname, "w") as ofile:
		for di in range(len(fit.bands)):
			ofile.write("%s %s %8.5f %8.5f %8.5f %4d %8.5f %8.5f\n" % (
				fit.detids[di], fit.bands[di], fit.gain[di],
				fit.a[di], fit.da[di], fit.n,
				fit.xieta[di,0]/utils.degree, fit.xieta[di,1]/utils.degree))

def dump_cmod_tfit(fname, fit):
	np.savetxt(fname, fit.ba, fmt="%8.5f")

config.default("elmod_cal", "cal,clean", "Comma-separated list of what to do with the elevation modulation gain fit. If it contains 'cal', then it will be used to flatfield the tod. If it contains 'dump', then the fit will be dumped to individual files. If it contains neither of these, then no elmod fit will be performed. If the elvation is not actually modulated, then nothing will be done")
config.default("elmod_minamp", 1, "elmod_cal is skipped if the elevation amplitude is less than this, in arcminutes")
def elmod_cal(obs, nmode=2, bsize=2000, tol=0.1, prefix=None, minamp=None, dev=None):
	"""Do elevation-modulation auto-calibration if elevation is actually modulated (determined by
	minamp), and config:elmod_cal has turned this on. Otherwise does nothing"""
	tasks = config.get("elmod_cal").split(",")
	if not ("cal" in tasks or "clean" in tasks or "dump" in tasks or "tdump" in tasks): return False
	# Check if we're el-modulated
	if minamp is None: minamp = config.get("elmod_minamp")*utils.arcmin
	elamp = np.std(obs.boresight[0,::100])*2**0.5
	if elamp < minamp: return False
	# Ok, do a fit
	try:
		fit = measure_el_response(obs, dev=dev)
		if "clean" in tasks:
			deproj_el_response(obs.tod, fit, dev=dev)
		if "cal" in tasks:
			obs.tod *= dev.np.array(fit.gain)[:,None]
		# Prefix required if dumping!
		if "dump" in tasks:
			dump_elmod_fit(prefix + "elcal.txt", fit)
		if "tdump" in tasks:
			dump_elmod_tfit(prefix + "telcal.txt", fit)
	except ValueError:
		return False
	return "cal" in tasks

def measure_el_response(obs, nmode=2, bsize=2000, tol=0.1, dev=None):
	if dev is None: dev = device.get_device()
	assert nmode >= 2, "elmod_cal needs nmode >= 2. The first two modes are an offset (discarded) and a slope (used for calibration)"
	ndet, nsamp = obs.tod.shape
	bel  = obs.boresight[0]
	# Assume const roll. Have to think about how to handle non-const roll if necessary
	roll = np.mean(obs.boresight[2,::100])
	xieta= obs.point_offset[:,1::-1]
	# f (bel) = a * 1/sin(f(bel)) = a * 1/sin(f(bel0)) + a * f'(bel0) * Δbel + ...
	# Let's call 1/sin(el) = x, and 1/sin(bel) = bx, and x = g(bx)
	# f (bx) = a*g(bx) = a*g(bx0) + a*g'(bx0)*Δbx + ...
	# Could expand around middle of xieta instad of boresight, but nice to have
	# a standard location for all runs.
	bx = 1/np.sin(bel)
	# Need d(det_x)/dbx for each detector. Will use finite difference over the whole
	# elevation range as representative. We assume constant roll
	belrange = utils.minmax(bel)
	bx1, bx2 = 1/np.sin(belrange)
	roll = np.mean(roll)
	x1 = 1/np.sin(calc_det_el(belrange[0], roll, xieta))
	x2 = 1/np.sin(calc_det_el(belrange[1], roll, xieta))
	gprime = ((x2-x1)/(bx2-bx1)).astype(obs.tod.dtype)
	Δbx = dev.np.array((bx-0.5*(bx1+bx2)).astype(obs.tod.dtype))
	# With this, the tod should be const + a*g'*Δbx in each block
	# Build the blocks
	nblock = nsamp//bsize
	btod   = obs.tod[:,:nblock*bsize].reshape(ndet,nblock,bsize)
	bΔbx   = Δbx[:nblock*bsize].reshape(nblock,bsize)
	# Build basis
	B     = dev.np.zeros((nmode,nblock,bsize), btod.dtype)
	B[:]  = bΔbx
	for i in range(nmode): B[i] **= i
	# Fit in blocks
	rhs   = dev.np.einsum("mbs,dbs->bdm", B, btod) # [nblock,ndet,nmode]
	div   = dev.np.einsum("mbs,nbs->bmn", B, B)   # [nblock,nmode,nmode]
	# Reduce to blocks with healthy determinant
	det   = dev.np.linalg.det(div)
	good  = det > 0
	if dev.np.sum(good) == 0: raise ValueError("elmod calibration failed for all blocks. Is elevation actually varying?")
	good  = det > dev.np.median(det[good])*tol
	if dev.np.sum(good) == 0: raise ValueError("elmod calibration failed for all blocks. Is elevation actually varying?")
	rhs, div = rhs[good], div[good]
	# Solve the remaining blocks
	idiv  = dev.np.linalg.inv(div)
	amps  = dev.np.einsum("bmn,bdn->bdm", idiv, rhs) # [nblock,ndet,nmode]
	slope = dev.get(amps[:,:,1]) # µK [nblock,ndet]
	# slope = a*g'. Divide out g' to get a
	ba    = slope/gprime/1e6 # K [nblock,ndet]
	# Use blocks to get robust mean and error
	a, da, n = gutils.robust_mean(ba, 0)
	# Build flatfield
	gain = np.full(ndet, 1, obs.tod.dtype)
	# flatfield per band, since the atm response should be freq-dependent
	ubands, order, edges = utils.find_equal_groups_fast(obs.bands)
	for bi, uband in enumerate(ubands):
		inds       = order[edges[bi]:edges[bi+1]]
		band_a     = a[inds]
		gain[inds] = np.median(band_a)/band_a
	# sanity checks here?
	return bunch.Bunch(a=a, da=da, n=n, gprime=gprime, gain=gain, ba=ba,
		bands=obs.bands, detids=obs.detids, xieta=xieta, Δbx=Δbx)

def deproj_el_response(tod, fit, dev=None):
	if dev is None: dev = device.get_device()
	ndet, nsamp = tod.shape
	amp = dev.np.array(fit.a*fit.gprime)
	# tod[d,i] += amp[d,1]*Δbx[1,s] => tod[i,d] += Δbx[s,1]*amp[1,d]
	dev.lib.sgemm("N", "N", nsamp, ndet, 1, -1, fit.Δbx[None], nsamp, amp[:,None], 1, 1, tod, nsamp)

def calc_det_el(bel, roll, xieta):
	bel, roll, xi, eta = np.broadcast_arrays(bel, roll, *xieta.T[:2])
	# Boresight
	bore = coordsys.Coords(az=bel*0, el=bel, roll=roll)
	# Focal plane
	q    = coordsys.rotation_xieta(xi, eta)
	el   = (bore*q).el
	return el

def dump_elmod_fit(fname, fit):
	with open(fname, "w") as ofile:
		for di in range(len(fit.bands)):
			ofile.write("%s %s %8.5f %8.5f %8.5f %4d %8.5f %8.5f %8.5f\n" % (
				fit.detids[di], fit.bands[di], fit.gain[di],
				fit.a[di], fit.da[di], fit.n,
				fit.xieta[di,0]/utils.degree, fit.xieta[di,1]/utils.degree, fit.gprime[di]))

def dump_elmod_tfit(fname, fit):
	np.savetxt(fname, fit.ba, fmt="%8.5f")

# Helpers

# The previous version of mask2cut was optimized for a small cut fraction, but
# used 9*todsize*cutfrac memory, which would be huge when cutfrac is big.
# This version uses todsize/2 + tiny memory, where tiny would only become
# problematic if the total number of cut ranges gets to O(10%) of the number
# of samples, which should be impossible at this stage
def mask2cut(tod, dev=None, pool=None):
	if dev  is None: dev  = device.get_device()
	ndet, nsamp = tod.shape
	with pool.as_allocator() if pool is not None else contextlib.nullcontext():
		# to boolean. Cost: 1/4
		mask = dev.np.zeros((ndet,nsamp+2), bool)
		mask[:,1:-1] = tod
		# changes. Cost: 1/4
		edges = mask[:,1:] != mask[:,:-1]
		# We assume that there won't be very many edges. Each edge costs 16 bytes,
		# so if that assumption fails, we can run out of memory. This would require
		# the cuts to be oscillating rapidly, which shouldn't happen.
		detsdets, startend = dev.np.nonzero(edges)
		# Because we False-padded, we're guaranteed to have start-end pairs
		dets   = detsdets[0::2]
		starts = startend[0::2]
		lens   = startend[1::2]
		lens  -= starts
	cuts = socut.Simplecut(dev.get(dets), dev.get(starts),
		dev.get(lens), ndet=tod.shape[0], nsamp=tod.shape[1])
	return cuts

def ephem_map(shape, wcs, object_list, trange, dt=1000, dr=1*utils.arcsec):
	"""Make a map where pixels hit by the objects in object_list [(name,rad),
	(name,rad),...], over the given ctime range trange are set to 1, and
	other pixels to zero. A pixel is hit if it's within each object's radius
	in object_list. The painting is done in steps of at most dt, so dt should
	be small enough that objects don't have time to move much relative to
	their masking radius over dt"""
	dtype = np.float32
	if len(object_list) == 0:
		return enmap.zeros((3,)+shape[-2:], wcs, dtype)
	t1, t2     = trange
	# Get object positions for a few times in this range
	eph_times  = np.linspace(t1, t2, 2+utils.floor((t2-t1)/dt))
	poss, rads = [], []
	for name, rad in object_list:
		pos, dist = ephem.eval(name, eph_times)
		poss.append(pos)
		rads.append(rad)
	poss = np.moveaxis(poss,-1,0) # [{ra,dec},nobj,nsamp,{ra,dec}]
	rads = np.array(rads) # [nobj]
	# Pad radius by half a pixel, to effectively round up partially
	# covered pixels
	rads += np.max(np.abs(wcs.wcs.cdelt))*utils.degree
	# Paint these on the map, with per-object max rad.
	# First tried to do this with a 1/r profile, but due to
	# large self-overlap, this resulted in much too high
	# values and therefore too large radius. Must instead
	# make one top-hat profile per unique radius.
	urads, order, edges = utils.find_equal_groups_fast(rads)
	rmax   = np.max(rads)
	prof_r = np.arange(0, rmax+2*dr, dr)
	profs  = []
	pinds  = np.zeros(len(rads),np.int32)
	for ip, rad in enumerate(urads):
		prof_v = prof_r <= rad
		profs.append(np.array([prof_r, prof_v]))
		pinds[order[edges[ip]:edges[ip+1]]] = ip
	amps   = np.ones(len(rads), dtype)
	# Broadcast and flatten
	amps   = np.broadcast_to(amps [:,None], poss.shape[1:]).reshape(-1)
	pinds  = np.broadcast_to(pinds[:,None], poss.shape[1:]).reshape(-1)
	poss   = poss.reshape(2,-1)
	# Finally do the actual painting
	map    = enmap.zeros((3,)+shape[-2:], wcs, dtype)
	pointsrcs.sim_objects(shape, wcs, poss[::-1], amps, profs, prof_ids=pinds, omap=map[0])
	map[0] = map[0] >= 1
	return map

def get_object_list(object_list=None, planet_list=None, asteroid_list=None, dedup=True):
	object_list   = config.get("object_cut",    object_list)
	groups = {
		"planets":   nonempty(config.get("planet_list",   planet_list).split(",")),
		"asteroids": nonempty(config.get("asteroid_list", asteroid_list).split(",")),
	}
	res = []
	for tok in object_list.split(","):
		subs = tok.split(":")
		name = subs[0]
		rad  = float(subs[1]) if len(subs) > 1 else 30
		rad *= utils.arcmin
		if name in groups:
			for member in groups[name]:
				res.append((member,rad))
		else:
			res.append((name,rad))
	if dedup:
		res = keep_last(res)
	return res

# Small utils below

def keep_last(vals):
	seen = set()
	res  = []
	for val in vals[::-1]:
		if val in seen: continue
		res.append(val)
		seen.add(val)
	res = res[::-1]
	return res

def nonempty(toks): return [tok for tok in toks if len(tok) > 0]
