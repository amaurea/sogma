# Things used by both sofast and soslow
import re, numpy as np, warnings
from sotodlib import core
from pixell import utils
from .. import device

def finish_query(res_db, pycode, sweeps=True, output="sogma"):
	if output not in ["sqlite", "resultset", "sogma"]:
		raise ValueError("Unrecognized output format '%s" % str(output))
	if output == "sqlite": return res_db
	info   = core.metadata.resultset.ResultSet.from_cursor(res_db.execute("select * from obs"))
	if output == "resultset" and not pycode: return info
	dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("baz","d"),("waz","d"),("bel","d"),("wel","d"),("r","d"),("sweep","d",(6,2))]
	obsinfo = np.zeros(len(info), dtype).view(np.recarray)
	obsinfo.id    = info["subobs_id"]
	obsinfo.ndet  = utils.dict_lookup(flavor_ndets_per_band, info["tube_flavor"])
	obsinfo.nsamp = info["n_samples"]
	obsinfo.ctime = info["start_time"]
	obsinfo.dur   = info["stop_time"]-info["start_time"]
	# Here come the parts that have to do with pointing.
	# These arrays have a nasty habit of being object dtype
	obsinfo.baz   = info["az_center"].astype(np.float64) * utils.degree
	obsinfo.bel   = info["el_center"].astype(np.float64) * utils.degree
	obsinfo.waz   = info["az_throw" ].astype(np.float64)*2 * utils.degree
	wafer_centers, obsinfo.r = wafer_info_multi(info["tube_slot"], info["wafer_slots_list"])
	if sweeps:
		obsinfo.sweep = make_sweep(obsinfo.ctime, obsinfo.baz, obsinfo.waz, obsinfo.bel, wafer_centers)
	# Evaluate pycode
	good = eval_pycode(pycode, obsinfo)
	if output == "resultset": return resultset_subset(info, np.where(good)[0])
	else: return obsinfo[good]

# Hard-coded raw wafer detector counts per band. Independent of
# telescope type, etc.
flavor_ndets_per_band = {"lf":118, "mf": 864, "uhf": 864}

wafer_pos_sat = [
	#      xi,      eta
	[  0.0000,   0.0000],
	[  0.0000, -12.6340],
	[-10.9624,  -6.4636],
	[-10.9624,   6.4636],
	[  0.0000,  12.6340],
	[ 10.9624,   6.4636],
	[ 10.9624,  -6.4636],
]

wafer_pos_lat = {
	"c1": [ [-0.3710,  0.0000], [ 0.1815,  0.3211], [ 0.1815, -0.3211] ],
	"i1": [ [-1.9112, -0.9052], [-1.3584, -0.5704], [-1.3587, -1.2133] ],
	"i2": [ [-0.3642, -1.7832], [ 0.1888, -1.4631], [ 0.1927, -2.1035] ],
	"i3": [ [ 1.1865, -0.8919], [ 1.7326, -0.5705], [ 1.7333, -1.2135] ],
	"i4": [ [ 1.1732,  0.9052], [ 1.7332,  1.2135], [ 1.7326,  0.5705] ],
	"i5": [ [-0.3655,  1.7833], [ 0.1879,  2.1045], [ 0.1867,  1.4620] ],
	"i6": [ [-1.9082,  0.8920], [-1.3577,  1.2133], [-1.3584,  0.5854] ],
	"o1": [ [-1.8959, -2.6746], [-1.3455, -2.3530], [-1.3392, -2.9954] ],
	"o2": [ [ 1.1876, -2.6747], [ 1.7447, -2.3537], [ 1.7505, -2.9965] ],
	"o3": [ [ 2.7302,  0.0000], [ 3.2929,  0.3220], [ 3.2929, -0.3219] ],
	"o4": [ [ 1.1876,  2.6747], [ 1.7505,  2.9965], [ 1.7447,  2.3537] ],
	"o5": [ [-1.8959,  2.6747], [-1.3392,  2.9955], [-1.3455,  2.3530] ],
	"o6": [ [-3.4369,  0.0000], [-2.8869,  0.3218], [-2.8869, -0.3218] ],
}

# Lowest possible sensitivity per detector in µK√s. Used for sanity checks.
# These are about half our forecast goal sensitivity
sens_limits = {"f030":120, "f040":80, "f090":100, "f150":140, "f220":300, "f750":750}

def wafer_info(tube_slot, wafer):
	# Allow both 0 and ws0
	if isinstance(wafer, str):
		wafer = int(wafer[2:])
	if re.match(r"stp\d", tube_slot):
		pos, rad = wafer_pos_sat[wafer], 6.0
	elif tube_slot in wafer_pos_lat:
		pos, rad = wafer_pos_lat[tube_slot][wafer], 0.3
	else: raise KeyError("tube %s wafer %d: no hardcoded position" % tube_slot, wafer)
	# Convert to radians
	pos = [p*utils.degree for p in pos]
	rad = rad*utils.degree
	return pos, rad

def wafer_info_multi(tubes, wafers, missing="warn"):
	"""Vectorized version of wafer info. Given tubes[nobs], wafers[nobs],
	returns poss[nobs,2], rads[nobs]"""
	nobs  = len(tubes)
	tag   = (tubes, wafers)
	label = utils.label_multi(tag)
	uvals, order, edges = utils.find_equal_groups_fast(label)
	poss  = np.zeros((nobs,2))
	rads  = np.zeros(nobs)
	for gi, uval in enumerate(uvals):
		inds = order[edges[gi]:edges[gi+1]]
		try:
			pos, rad = wafer_info(tubes[inds[0]], wafers[inds[0]])
		except KeyError as e:
			pos, rad = [0,0], 0
			if   missing == "warn": warnings.warn(str(e))
			elif missing == "ignore": pass
			else: raise
		poss[inds] = pos
		rads[inds] = rad
	return poss, rads

def sensitivity_cut(rms_uKrts, sens_lim, med_tol=0.2):
	ap  = device.anypy(rms_uKrts)
	# First reject detectors with unreasonably low noise
	good     = rms_uKrts >= sens_lim
	# Then reject outliers
	if ap.sum(good) == 0: return good
	ref      = ap.median(rms_uKrts[good])
	good    &= rms_uKrts > ref*med_tol
	good    &= rms_uKrts < ref/med_tol
	return good

def measure_rms(tod, dt=1, bsize=32, nblock=10):
	ap  = device.anypy(tod)
	tod = tod[:,:tod.shape[1]//bsize*bsize]
	tod = tod.reshape(tod.shape[0],-1,bsize)
	bstep = max(1,tod.shape[1]//nblock)
	tod = tod[:,::bstep,:][:,:nblock,:]
	rms = ap.median(ap.std(tod,-1),-1)
	# to µK√s units
	rms *= dt**0.5
	return rms

# This sweep isn't quite accurate. It's off by ~1°.
# Is something going wrong with the offset? The math
# looks good to me, and sign flips make things worse.
# The error isn't consistently in the same direction
# in horizontal coordinates. For now we'll just have to
# operate with a margin of error
def make_sweep(ctime, baz0, waz, bel0, off, npoint=6, nocross=True):
	import so3g
	from pixell import coordinates
	# given ctime,baz0,waz,bel [ntod], off[ntod,{xi,eta}], make
	# make sweeps[ntod,npoint,{ra,dec}]
	# so3g can't handle per-sample pointing offsets, so it would
	# force us to loop here. We therefore simply modify the pointing
	# offsets to we can simply add them to az,el
	az_off, el = coordinates.euler_rot((0.0, -bel0, 0.0), off.T)
	az1 = baz0+az_off-waz/2
	az2 = baz0+az_off+waz/2
	if nocross: az1, az2 = truncate_az_crossing(az1, az2)
	az  = az1[:,None] + (az2-az1)[:,None]*np.linspace(0,1,npoint)
	el  = el   [:,None] + az*0
	ts  = ctime[:,None] + az*0
	sightline = so3g.proj.coords.CelestialSightLine.az_el(
		ts.reshape(-1), az.reshape(-1), el.reshape(-1), site="so", weather="typical")
	pos_equ = np.asarray(sightline.coords()) # [ntot,{ra,dec,cos,sin}]
	sweep   = pos_equ.reshape(len(ctime),npoint,4)[:,:,:2]
	# Make sure we don't have any sudden jumps in ra
	sweep[:,:,0] = utils.unwind(sweep[:,:,0])
	return sweep

def truncate_az_crossing(az1, az2):
	# Which side of the sky are we on?
	amid = 0.5*(az1+az2)
	# Legal bounds
	leg1 = utils.floor(amid/np.pi)*np.pi
	leg2 = utils.ceil (amid/np.pi)*np.pi
	az1  = np.maximum(az1, leg1)
	az2  = np.minimum(az2, leg2)
	return az1, az2

# How to check if point is hit by observation?
# 1. Build interpol ra(dec) for sweep
# 2. If point not within sweep dec range padded by array rad, we're not hit
# 3. Evaluate sweep at dec of point, getting ra. Clip to valid dec range.
# 4. The sky rotates by -15°/hour in ra, which means that our coverage
#    rotates by +15°/hour in ra. So we're hit if ra_point inside
#    [ra-r,ra+15°/hour*dur+r]
def point_hit(point, sweep, dur, r, pad=1.0*utils.degree):
	"""Check if the given points are hit by an observation with the given sweep, duration and
	wafer radius. pad gives a safety margin, and is needed because sweep is a bit inaccurate
	for some reason. 1 degree should be enough to make up for this."""
	# point[nalt,:,{ra,dec}], point[:,{ra,dec}], or [{ra,dec}], sweep[:,npoint,{ra,dec}], dur[:], r[:]
	point, _ = np.broadcast_arrays(point, sweep[:,0,:])
	if point.ndim == 3:
		# Handle 3D case, where we have multiple points and want to know if we hit
		# any of them
		return np.any([point_hit(p, sweep, dur, r, pad=pad) for p in point],0)
	nobs, nsamp = sweep.shape[:2]
	# Our output array
	was_hit = np.zeros(nobs,bool)
	# 1. Check dec range
	dec1, dec2 = utils.minmax(sweep[:,:,1],-1)
	# 2. Don't try if dec range is too short
	good = dec2-dec1 > 0.1*utils.degree
	ra   = poly_interpol(point[good,1], sweep[good,:,1], sweep[good,:,0])
	# Check if we're in bounds for the valid obss
	pra, pdec = point[good].T
	eff_rad   = r[good]/np.cos(pdec)
	speed     = 15*utils.degree/utils.hour
	dec_hit   = (pdec>dec1[good]-r[good]-pad)&(pdec<dec2[good]+r[good]+pad)
	ra_hit    = (pra>ra-eff_rad-pad)&(pra<ra+speed*dur[good]+eff_rad+pad)
	was_hit[good] = ra_hit & dec_hit
	return was_hit

def poly_interpol(x, xp, yp):
	nobs, nsamp = xp.shape
	xmin, xmax = utils.minmax(xp,-1)
	# Build polynomial interpol. Scipy spline not vectoriced enough
	def normalize(x): return (2*(x.T-xmin.T)/(xmax.T-xmin.T)).T
	xnorm= normalize(xp)
	B    = np.array([xnorm**i for i in range(nsamp)]) # [order,nobj,nsamp]
	rhs  = np.einsum("anp,np->na", B, yp)
	div  = np.einsum("anp,bnp->nab", B, B)
	amp  = np.einsum("nab,nb->na", np.linalg.inv(div), rhs)
	# Evaluate at requested position
	xnorm= normalize(x)
	B    = np.array([xnorm**i for i in range(nsamp)]) # [order,nobj]
	y    = np.einsum("na,an->n", amp, B)
	return y

########################################################
# Functions handling python code evaluation in queries #
########################################################

def eval_pycode(pycode, obsinfo):
	if not pycode: return ()
	def pycode_hits(ra, dec=None):
		if isinstance(ra, str): pos = planet_pos(ra, obsinfo)
		else: pos = np.array([ra,dec])*utils.degree
		return point_hit(pos, obsinfo.sweep, obsinfo.dur, obsinfo.r)
	def pycode_planet(name): return planet_pos(name, obsinfo)
	globs = {}
	# Make numpy available, both with and without np
	globs.update(**vars(np))
	globs["np"] = np
	# Register our function
	globs["hits"]   = pycode_hits
	globs["planet"] = pycode_planet
	return eval(pycode, globs)

planet_names = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "sun", "moon"]

def planet_pos(name, obsinfo):
	import astropy.time, astropy.coordinates
	name = name.lower()
	if name == "planet": names = planet_names
	else: names = name.split(",")
	poss = []
	for name in names:
		if name in astropy.coordinates.solar_system_ephemeris.bodies:
			t   = astropy.time.Time(obsinfo.ctime, format="unix")
			obj = astropy.coordinates.get_body(name, t)
			pos = np.array([obj.ra.radian, obj.dec.radian]).T
		else:
			raise ValueError("Unrecognized body '%s'" % str(name))
		poss.append(pos)
	return np.array(poss)

# Result-set workaround
def resultset_subset(resultset, inds):
	return core.metadata.resultset.ResultSet(resultset.keys, [resultset.rows[ind] for ind in inds])


