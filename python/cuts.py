import numpy as np, os
from pixell import config, utils, enmap, ephem, pointsrcs
from . import pmat, tiling, device

config.default("object_cut",    "planets:30,asteroids:5")
config.default("planet_list",   "Mercury,Venus,Mars,Jupiter,Saturn,Uranus,Neptune", "What planets the 'planets' keyword in object_cut expands to")
# Vesta has a peak brightness of 1 Jy @f150. These asteroids get within 4% of that
# (40 mJy) at some point in their orbit by extrapolation. This is the 5σ forecasted
# depth-1 sensitivity at f150, and would be even weaker after dilution from multile
# exposures, so this should be a safe level without cutting too much.
config.default("asteroid_list", "Vesta,Ceres,Pallas,Juno,Eunomia,Hebe,Iris,Pluto,Eris,Amphitrite,Makemake,Hygiea,Herculina,Metis,Flora,Dembowska,Melpomene,Haumea,Psyche,Laetitia,Massalia", "What asteroids the 'asteroids' keyword in object_cut expands to")
# This should probably support variables like $SOPATH to be less system specific
# os.path.expandvars is good for this
config.default("asteroid_path", "/home/sigurdkn/project/so/ephemerides/objects")

def object_cut(obs, object_list=None, geo=None, down=4, base_res=0.5*utils.arcmin,
		dt=100, dr=1*utils.arcsec, dev=None):
	if dev is None: dev = device.get_device()
	setup_ephem()
	object_list = get_object_list(object_list)
	# Set up a low-resolution geometry, either by downgrading a given geometry
	# of by downgrading a fullsky geometry with the base_res resolution.
	# The advantage of passing an existing geometry is that one avoids partially
	# cut pixels, which can have very high noise.
	shape, wcs = geo if geo is not None else enmap.fullsky_geometry2(res=res)
	shape, wcs = enmap.downgrade_geometry(shape, wcs, down)
	# Get our time range
	t1, t2 = utils.minmax(obs.ctime)
	map    = ephem_map(shape, wcs, object_list, [t1,t2], dt=dt, dr=dr)
	# map2tod. ft buffer and the pointing buffers are free
	lmap = tiling.enmap2lmap(map, dev=dev)
	pmap = pmat.PmatMap(shape, wcs, obs.ctime, obs.boresight, obs.point_offset, obs.polangle, dev=dev)
	tod  = dev.pools["ft"].zeros(obs.tod.shape, obs.tod.dtype)
	pmap.forward(tod, lmap)
	# turn non-zero values into a cuts object
	# return result. User must merge with other cuts and gapfill as necessary

def mask2cut(tod, dev=None):
	if dev is None: dev = device.get_device()
	# I think the best way to do this is with np.nonzero followed by
	# converting that to cuts. I considered starting from np.diff, but
	# too many big arrays are involved
	dets, samps = dev.np.nonzero(tod)
	# Now need to find offset and length of consecutive regions
	isstart = (dev.np.ediff1d(samps, to_begin=-1, to_end=-1)!=1)|(dev.np.ediff1d(dets, to_begin=1, to_end=1)!=0)
	edges   = dev.np.nonzero(isstart)[0]
	offs    = edges[:-1]
	lens    = dev.np.diff(edges)
	# Format as cuts. These are expected to be on the cpu
	cuts    = np.zeros((len(offs),3),np.int32)
	cuts[:,0] = dev.get(dets[offs])
	cuts[:,1] = dev.get(samps[offs])
	cuts[:,2] = dev.get(lens)
	return cuts

def ephem_map(shape, wcs, object_list, trange, dt=1000, dr=1*utils.arcsec):
	"""Make a map where pixels hit by the objects in object_list [(name,rad),
	(name,rad),...], over the given ctime range trange are set to 1, and
	other pixels to zero. A pixel is hit if it's within each object's radius
	in object_list. The painting is done in steps of at most dt, so dt should
	be small enough that objects don't have time to move much relative to
	their masking radius over dt"""
	if len(object_list) == 0:
		return enmap.zeros((3,)+shape[-2:], wcs, np.float32)
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
	# Paint these on the map, with per-object max rad. Easiest to do this with
	# a 1/r-profile shared between all, an amplitude of rad, and a
	# shared vmin of 1
	rmax   = np.max(rads)
	prof_r = np.arange(0, rmax+2*dr, dr)
	prof_v = 1/np.maximum(prof_r, prof_r[1]/2)
	profile= np.array([prof_r,prof_v])
	amps   = np.broadcast_to(rads[:,None], poss.shape[1:])
	# flatten, since we only needed the 2d shape for broadcasting
	poss   = poss.reshape(2,-1)
	amps   = amps.reshape(-1)
	# Finally do the actual painting
	map    = enmap.zeros((3,)+shape[-2:], wcs, np.float32)
	pointsrcs.sim_objects(shape, wcs, poss[::-1], amps, profile, omap=map[0], vmin=1)
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

ephem_setup_done = False
def setup_ephem(asteroid_path=None):
	global ephem_setup_done
	if ephem_setup_done: return
	asteroid_path = os.path.expandvars(config.get("asteroid_path", asteroid_path))
	if not asteroid_path: return
	astephem = ephem.PrecompEphem(asteroid_path)
	ephem.add(astephem)
	ephem_setup_done = True

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
