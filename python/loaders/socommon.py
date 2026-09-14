# Things used by both sofast and soslow
import re, numpy as np, warnings, os, yaml, contextlib, copy
from pixell import utils, sqlite, bunch, config, fft
from .. import device, gutils, socut
from . import minisotodlib

# Base loading classes

class Loader:
	def __init__(self, dev=None, pool_map={}, dtype=np.float32):
		self.dev      = dev or device.get_device()
		self.pool_map = pool_map
		self.dtype    = dtype
	def query(self, query=None):
		raise NotImplementedError
	def probe(self, linfo, id, dets=None, detids=None):
		raise NotImplementedError
	def load(self, linfo, id, dets=None, detids=None, samprange=None, pinfo=None):
		raise NotImplementedError
	# Grouping doesn't really belong here. What would make more sense would be
	# something like classify_obs, which would return a list of tags per obs.
	# Or maybe this should already be a part of what query handles. Instead of
	# parsing obsids, would be nice to have these things simply available as tags.
	# Oh well, leaving it for now.
	def group_obs(self, linfo, mode="obs"):
		raise NotImplementedError
	def prealloc(self, linfo):
		raise NotImplementedError
	def avoid_pools(self, names, suf="*"):
		# First register directly
		for name in names:
			self.pool_map[name] = name + suf
		# Then resolve chains
		self.pool_map = {name:recursive_lookup(self.pool_map,name) for name in self.pool_map}
	def pool(self, name):
		try: name = self.pool_map[name]
		except KeyError: pass
		return self.dev.pools[name]

class LoadInfo:
	"""Class representing a set of observations to load, and metadata needed to load it.
	Contains at least the .obsinfo member a numpy table of the observations and their properties,
	and provides the load() meathod for reading in an observation"""
	def __init__(self, loader, obsinfo, omap=None, dets=None, detids=None):
		self.loader  = loader
		self.obsinfo = obsinfo
		# Detector restriction
		self.dets    = dets
		self.detids  = detids
		if omap is None:
			omap = {id:oi for oi,id in enumerate(obsinfo.id)}
		self.omap    = omap
	@property
	def nobs(self): return len(self.obsinfo)
	def probe(self, id, dets=None, detids=None):
		return self.loader.probe(self, id, dets=dets, detids=detids)
	def load(self, id, dets=None, detids=None, samprange=None, pinfo=None):
		return self.loader.load(self, id, dets=dets, detids=detids, samprange=samprange, pinfo=pinfo)
	def prealloc(self):
		return self.loader.prealloc(self)
	def copy(self): return copy.copy(self)
	def __getitem__(self, sel):
		# Generic slice that should work for most subclasses
		res = self.copy()
		res.obsinfo = self.obsinfo[sel]
		res.omap    = {id:oi for oi,id in enumerate(res.obsinfo.id)}
		return res

class ProbeInfo:
	"""Class representing the result of probing an observation, which means doing a
	relatively light-weight partial read in order to determine the actually readable number
	of detectors, the number of samples and the absolute sample timing. May also include other
	information as needed by the individual loaders"""
	def __init__(self, ndet, nsamp, t1, srate, **kwargs):
		self.ndet, self.nsamp, self.t1, self.srate = ndet, nsamp, t1, srate
		self.__dict__.update(kwargs)

def recursive_lookup(imap, name):
	while name in imap:
		name = imap[name]
	return name

def srange_suffix(id, srange):
	if srange is None: return id
	else: return id + "," + "%d:%d" % tuple(srange)

def srange_trunc(srange, dev):
	srange = np.array(srange)
	srange[...,1] = srange[...,0] + dev.goodlen(srange[...,1]-srange[...,0])
	return srange

def srange_expand(srange, factor=1):
	return utils.nint(srange*factor)

def srange_chain(srange1, srange2):
	"""If srange2 is a sub-srange to srange1, what absolute sample range does it actually cover?"""
	res = srange1[...,0,None] + srange2
	res[...,1] = np.minimum(res[...,1], srange1[...,1])
	return res

########################
# Contexts and configs #
########################

def find_so(): return os.environ["SOPATH"]
#def find_cdir(telescope): return find_so() + "/metadata/%s/contexts" % telescope
# Temporarily switch to my own context directory, since the official one isn't being
# kept up to date
def find_cdir(telescope): return find_so() + "/users/sigurdkn/contexts/%s" % telescope
def find_context(path_or_name, type="preprocess"):
	if not re.match(r"^\w+$", path_or_name):
		# Treat it as a path if it contains non-word characters
		return path_or_name
	else:
		# Otherwise, treat it as a telescope name
		cdir = find_cdir(path_or_name)
		for name in [type + "_local", type]:
			path = "%s/%s.yaml" % (cdir, name)
			if os.path.exists(path): return path
		raise FileNotFoundError

def expand_context(context):
	tags = context["tags"]
	return _expand_context_helper(context, tags)

def _expand_context_helper(obj, tags):
	if isinstance(obj, dict):
		return {key:_expand_context_helper(obj[key], tags) for key in obj}
	elif isinstance(obj, list):
		return [_expand_context_helper(val, tags) for val in obj]
	elif isinstance(obj, str):
		return obj.format(**tags)
	else:
		return obj

def find_label(cmeta, name, whole=False):
	for entry in cmeta:
		for key in ["label", "name"]: # both not standardized?
			if key in entry and entry[key] == name:
				return entry["db"] if not whole else entry
def cmeta_lookup(context, name):
	val = find_label(context["metadata"], name)
	return val.format(**context["tags"]) if val else None
def read_yaml(fname):
	with open(fname, "r") as ifile:
		return yaml.safe_load(ifile)

def get_expanded_context(context_or_config_or_name):
	"""Given either a config, a context or a telescope name, returns
	the info sofast actually needs, which is a context object with
	a preprocess archive entry, and {} tags expanded"""
	if re.match(r"^\w+$", context_or_config_or_name):
		# A plain word. Treat as telescope name
		cpath = find_context(context_or_config_or_name)
	else:
		cpath = context_or_config_or_name
	# Read the yaml file
	context = read_yaml(cpath)
	# Is it a config file? If so, it will have a context_file entry
	if "context_file" in context:
		cpath = os.path.join(os.path.dirname(cpath), context["context_file"])
		ppath = os.path.join(os.path.dirname(cpath), context["archive"]["index"])
		context = read_yaml(cpath)
	else: ppath = None
	# Add cdir to tags, so we can have context-dir relative paths
	context["tags"]["cdir"] = os.path.dirname(cpath)
	# Ok, by now we have a context dict. Expand curly braces in it
	context = expand_context(context)
	# Set the preprocess path if we have one from config
	if ppath:
		entry = find_label(context["metadata"], "preprocess", whole=True)
		if entry: entry["db"] = ppath
		else: context["metadata"].append({"db":ppath, "label":"preprocess", "unpack":"preprocess"})
	# Check that we actually have a preprocess entry in the end
	if not find_label(context["metadata"], "preprocess"):
		raise ValueError("Could not infer preprocess archive from '%s'" % (context_or_config_or_name))
	return context

def group_obs(obsinfo, mode="wafer"):
	if len(obsinfo) == 0:
		return bunch.Bunch(names=[], groups=[], bands=[], nullbands=[], joint=mode!="none", sampranges=None)
	if mode == "obs":
		# Group by the obs-id. For the LAT, this will group the 3 wafers in a tube,
		# but keep the tubes separate
		key = np.char.partition(obsinfo.id, ":")[:,0]
		groups, names = _group_obs_simple(key)
	elif mode == "oband":
		# Group by obs-band
		key = astr_tok_inds(obsinfo.id, ":", [0, 2])
		groups, names = _group_obs_simple(key)
	elif mode == "wafer":
		key = astr_tok_range(obsinfo.id, ":", 0, 2)
		groups, names = _group_obs_simple(key)
	elif mode == "none":
		key = obsinfo.id
		groups, names = _group_obs_simple(key)
	elif mode == "wband":
		# Group by wafer-band. Usually not necessary, as one
		# usually either selects a single band or splits by band anyway
		key = astr_tok_range(obsinfo.id, ":", 0, 3)
		groups, names = _group_obs_simple(key)
	elif mode == "full":
		groups, names = _group_obs_tol(obsinfo)
	else: raise ValueError("Unrecognized subid grouping mode '%s'" % str(mode))
	full_bands = np.unique(astr_tok_range(obsinfo.id, ":", 2, 4))
	is_null    = np.array([(":" in band and not band.endswith(":OPTC")) for band in full_bands])
	joint = bunch.Bunch(names=names, groups=groups,
		bands=full_bands[~is_null], nullbands=full_bands[is_null],
		joint=mode!="none", sampranges=None)
	return joint

def astr_tok_inds(astr, sep, inds):
	# np.char lacks substr, so just loop
	res = []
	for word in astr:
		toks = word.split(sep)
		res.append(sep.join([toks[i] for i in inds]))
	return np.array(res)

def astr_tok_range(astr, sep, start, end):
	# np.char lacks substr, so just loop
	return np.array([sep.join(word.split(sep)[start:end]) for word in astr])

def _group_obs_simple(key):
	names, order, edges = utils.find_equal_groups_fast(key)
	groups = [order[edges[i]:edges[i+1]] for i in range(len(edges)-1)]
	return groups, names

def _group_obs_tol(obsinfo, tol=100):
	groups = utils.find_equal_groups(obsinfo.ctime, tol=tol)
	names  = np.array(["_".join(obsinfo.id[g[0]].split("_")[:2]) for g in groups])
	return groups, names

def find_scanning(az, down=10, tol=0.01, pad=1):
	"""Find the first and last sample where the telescope is scanning in az"""
	# Downgrade to reduce noise
	baz  = gutils.downgrade(az, down)
	# Measure the speed
	v     = np.abs(np.gradient(baz))
	vtyp  = np.mean(v)
	moving= np.where(v>vtyp*tol)[0]
	if len(moving) == 0: return 0, 0
	i1   = max((moving[ 0]-1-pad)*down, 0)
	i2   = min((moving[-1]+1+pad)*down+1, az.size)
	return i1, i2

# Lowest possible sensitivity per detector in µK√s. Used for sanity checks.
# These are about half our forecast goal sensitivity
sens_limits = {"f030":120, "f040":80, "f090":100, "f150":140, "f220":300, "f280":750}

def sensitivity_cut(rms_uKrts, sens_lim, med_tol=0.2, max_lim=10000):
	ap  = device.anypy(rms_uKrts)
	# First reject detectors with unreasonably low noise
	good     = rms_uKrts >= sens_lim
	# Also reject far too noisy detectors
	good    &= rms_uKrts <  sens_lim*max_lim
	# Then reject outliers
	if ap.sum(good) == 0: return good
	ref      = ap.median(rms_uKrts[good])
	good    &= rms_uKrts > ref*med_tol
	good    &= rms_uKrts < ref/med_tol
	return good

_rms_der_norm = [1,2,6,20,70,252]
def measure_rms_der(tod, dt=1, nder=3, bsize=32, nblock=10):
	ap  = device.anypy(tod)
	tod = tod[:,:tod.shape[1]//bsize*bsize]
	tod = tod.reshape(tod.shape[0],-1,bsize)
	bstep = max(1,tod.shape[1]//nblock)
	tod = tod[:,::bstep,:][:,:nblock,:]
	# Take the nder'th derivative, to effectively highpass filter.
	# This will put our focus on only the highest freqs, which may not
	# be representative of the practical white noise floor, but it will
	# make us robust to any hwp
	tod  = ap.diff(tod, n=nder, axis=-1)
	rms  = ap.median(ap.std(tod,-1),-1)
	rms /= _rms_der_norm[nder]**0.5
	# to µK√s units
	rms *= dt**0.5
	return rms

# This one is not robust to bright signals like planets.
# Trying to map a planet would see the detectors that see it
# disqualified for being too noisy. The good thing about
# this version is that it's robust to the hwp
def measure_rms_ft(ftod, dt=1, fmin=30, fmax=100):
	fnyq = 0.5/dt
	imin = utils.ceil (ftod.shape[-1] * fmin/fnyq)
	imax = utils.floor(ftod.shape[-1] * fmax/fnyq)
	fsub = ftod[:,imin:imax]
	rms  = np.std(fsub,-1)*(dt/fsub.shape[-1])**0.5 * ftod.shape[-1]
	return rms

# This one is robust to planets, but fails in the
# presence of a hwp, since all the blocks would be
# impacted.
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

def det_intersect(dets1, dets2):
	if dets1 is None: return dets2
	if dets2 is None: return dets1
	return np.unique(np.concatenate([dets1,dets2]))
