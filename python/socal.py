import numpy as np, os
from pixell import config, utils, enmap, ephem, pointsrcs, coordsys, bunch
from . import pmat, tiling, device, socut, gutils, soeph

config.default("autocut",       "objects", "Comma-separated list of which autocuts to apply. Currently recognized: objects: The object cut.")
config.default("object_cut",    "planets:10,asteroids:5")
# planetas and asteroids defined in sogma.soeph

def autocut(obs, which=None, geo=None, dev=None):
	"""Main driver. Performs all autocuts, merges them with any existing cuts,
	updates obs and gpfills tod"""
	if dev is None: dev = device.get_device()
	which   = config.get("autocut", which).split(",")
	cutfuns = {"objects": object_cut}
	if len(which) == 0: return obs
	cuts = [obs.cuts]
	for i, cutname in enumerate(which):
		cuts.append(cutfuns[cutname](obs, geo=geo, dev=dev))
	cuts = socut.Simplecut.merge(cuts)
	#cuts.gapfill(obs.tod, dev=dev)
	obs.cuts = cuts
	return obs

def object_cut(obs, object_list=None, geo=None, down=8, base_res=0.5*utils.arcmin,
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
	pmap = pmat.PmatMap(lmap.fshape, lmap.fwcs, obs.ctime, obs.boresight, obs.point_offset, obs.polangle, dev=dev)
	tod  = dev.pools["ft"].zeros(obs.tod.shape, obs.tod.dtype)
	pmap.forward(tod, lmap.lmap)
	# turn non-zero values into a cuts object
	cuts = mask2cut(tod, dev=dev)
	# return result. User must merge with other cuts and gapfill as necessary
	return cuts

config.default("elmod_cal", "cal", "Comma-separated list of what to do with the elevation modulation gain fit. If it contains 'cal', then it will be used to flatfield the tod. If it contains 'dump', then the fit will be dumped to individual files. If it contains neither of these, then no elmod fit will be performed. If the elvation is not actually modulated, then nothing will be done")
config.default("elmod_minamp", 1, "elmod_cal is skipped if the elevation amplitude is less than this, in arcminutes")
def elmod_cal(obs, nmode=2, bsize=2000, tol=0.1, prefix=None, minamp=None, dev=None):
	"""Do elevation-modulation auto-calibration if elevation is actually modulated (determined by
	minamp), and config:elmod_cal has turned this on. Otherwise does nothing"""
	elmod = config.get("elmod_cal").split(",")
	if not ("cal" in elmod or "clean" in elmod or "dump" in elmod or "tdump" in elmod): return
	# Check if we're el-modulated
	if minamp is None: minamp = config.get("elmod_minamp")*utils.arcmin
	elamp = np.std(obs.boresight[0,::100])*2**0.5
	if elamp < minamp: return
	# Ok, do a fit
	try:
		fit = measure_el_response(obs, dev=dev)
		if "clean" in elmod:
			deproj_el_response(obs.tod, fit, dev=dev)
		if "cal" in elmod:
			obs.tod *= dev.np.array(fit.gain)[:,None]
		# Prefix required if dumping!
		if "dump" in elmod:
			dump_elmod_fit(prefix + "elcal.txt", fit)
		if "tdump" in elmod:
			dump_elmod_tfit(prefix + "telcal.txt", fit)
	except ValueError: pass

def dump_elmod_fit(fname, fit):
	with open(fname, "w") as ofile:
		for di in range(len(fit.bands)):
			ofile.write("%s %s %8.5f %8.5f %8.5f %8.5f %4d %8.5f\n" % (
				fit.detids[di], fit.bands[di], fit.xieta[di,0]/utils.degree,
				fit.xieta[di,1]/utils.degree, fit.a[di]/1e6, fit.da[di]/1e6,
				fit.n, fit.gprime[di]))

def dump_elmod_tfit(fname, fit):
	np.savetxt(fname, fit.ba/1e6, fmt="%8.5f")

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
	ba    = slope/gprime # µK [nblock,ndet]
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

# Helpers

def mask2cut(tod, dev=None):
	if dev is None: dev = device.get_device()
	# I think the best way to do this is with np.nonzero followed by
	# converting that to cuts. I considered starting from np.diff, but
	# too many big arrays are involved
	dets, samps = dev.np.nonzero(tod)
	if len(dets) == 0:
		return socut.Simplecut(ndet=tod.shape[0], nsamp=tod.shape[1])
	# Now need to find offset and length of consecutive regions.
	# Cupy is being clunky here, so must construct this array
	one = dev.np.ones(1,samps.dtype)
	isstart = (dev.np.ediff1d(samps, to_begin=-one, to_end=-one)!=1)|(dev.np.ediff1d(dets, to_begin=one, to_end=one)!=0)
	edges   = dev.np.nonzero(isstart)[0]
	offs    = edges[:-1]
	lens    = dev.np.diff(edges)
	# Format as cuts. These are expected to be on the cpu
	cuts    = socut.Simplecut(dev.get(dets[offs]), dev.get(samps[offs]),
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
