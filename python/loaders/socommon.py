# Things used by both sofast and soslow
import re, numpy as np, warnings, os, yaml, contextlib
from pixell import utils, sqlite, bunch, config
from .. import device, gutils

def eval_query(obsdb, simple_query, predb=None, default_good=True, cols=None, tags=None, pre_cols=None, subobs=True, _obslist=None):
	"""Given an obsdb SQL and a Simple Query, evaluate the
	query and return a new SQL object (pointing to a temporary
	file) containing the resulting observations."""
	if cols is None: cols = sqlite.columns(obsdb, "obs")
	if tags is None: tags = get_tags(obsdb)
	if pre_cols is None and predb is not None: pre_cols = sqlite.columns(predb, "map")
	if not simple_query: simple_query = "1"
	# Check if we have a subobsdb or not
	is_subobs = "subobs_id" in cols
	# Main parse of the simple query
	qinfo = parse_query(simple_query, cols, tags, pre_cols=pre_cols, default_good=default_good)
	qjoin = ""
	# By default, order by obs_id or subobs id
	if is_subobs: qsort = "order by subobs_id"
	else:         qsort = "order by obs_id"
	# Create database we will put the result in, and attach
	# it temporarily to obsdb so we can write to it
	res_db  = sqlite.SQL()
	with sqlite.attach(obsdb, res_db, "res", mode="rw"):
		# Attach preprocess db if availabl
		with sqlite.attach(obsdb, predb, "pre", mode="r") if predb else contextlib.nullcontext():
			if qinfo.idfile:
				# Ok, an obslist was passed. The _oblist stuff is
				# so we can reuse an already read in obslist in the
				# subobs pass
				if _obslist is None: _obslist = load_obslist(qinfo.idfile)
				if is_subobs and _obslist["subobs"] is not None:
					# Yes, obsdb, but indirectly res_db. Can't use res_db directly since
					# we're working on a copy of it due to how attach works
					add_list(obsdb, "res.targ", _obslist["subobs"])
					qjoin = "join res.targ on obs.subobs_id = res.targ.id"
				else:
					add_list(obsdb, "res.targ", _obslist["obs"])
					qjoin = "join res.targ on obs.obs_id = res.targ.id"
				qsort = "order by res.targ.ind"
			qjoin += qinfo.join
			# Build the full query
			query = "select obs.* from obs %s where %s %s" % (qjoin, qinfo.where, qsort)
			# Use it to build the output table
			obsdb.execute("create table res.obs as %s" % query)
			# Also get the tags
			obsdb.execute("create table res.tags as select tags.* from tags where obs_id in (select obs_id from res.obs)")
			if qinfo.idfile:
				# Drop the obslist table we created
				obsdb.execute("drop table res.targ")
	# Ok, by this we're detached again, and res_db contains the
	# resulting obs and tags tables. But we may need a second pass
	# if we're still at the obsdb level, but need a subobs db
	if not subobs: return res_db, qinfo.pycode, qinfo.slices # didn't ask for subobs
	if is_subobs:  return res_db, qinfo.pycode, qinfo.slices # already subobs
	# Ok, do the subobs expansion
	with subobs_expansion(res_db) as subobs_db:
		res_db.close() # don't need this one any more
		return eval_query(subobs_db, simple_query, predb=predb, default_good=default_good, tags=tags, pre_cols=pre_cols, _obslist=_obslist)

def parse_query(simple_query, cols, tags, pre_cols=None, default_good=True):
	"""Gven a Simple Query, return an sqlite selection for an obsdb.
	When cols contains 'band' and wafer_slots_list only has one entry,
	as in a subobsdb, it will fully handle selections like ws0,f090.
	Otherwise, as in obsdb, it will select any obs that partially matches.

	Returns:
		* query: the part of the full SQL select statement that comes after where.
		  Typically the full query would then be "select obs.* from obs where %s" % query,
		  but it's up to the user to complete it like that.
		* idfile: The path to a file with a plain list of obsids or subids, or None if
		  the simple_query didn't refer to one.
	"""
	# A Simple Query is a comma-separated list of constraints.
	# Each constraint can be just a tag name, or a full expression
	# 1. Translate ,-separation into ()and()
	toks   = utils.split_outside(simple_query, ",", start="([{'\"", end=")]}'\"")
	# 2. Inital pass through syntax that's only supported at top-level
	idfile = None
	otoks  = []
	pycode = []
	slices = []
	det_type_set = False
	for tok in toks:
		# Direct tod list. Can be combined with other constraints, but
		# only to further restrict. Multiple @ per query not supported -
		# only the last one will be respected. @ must be followed by
		# the path to a text file with lines starting with an obsid
		# or subid each. Anything after the obsid is ignored. The file
		# must be either all-obsid or all-subid, so you can't mix them.
		# The results will be in the same order as this file. Any entres
		# not in the database will be silently discarded.
		if tok.startswith("@"):
			idfile = tok[1:]
			# Hack: Count det type to be set if we read in from a file, since
			# the det type is part of the obs-db. If we don't do this, then
			# dark detectors will be ignored in the idfile unless +dark is given,
			# which is confusing. Ideally this would be handled separately for the
			# file and any other constraints, but this will do for now.
			det_type_set = True
		elif is_slice(tok):
			slices.append(utils.parse_slice("["+tok+"]"))
		# Very hacky! (See notes above on future directions)
		# Look for specific function calls that must be implemented in python.
		# If present, the whole top-level segment will be treated in python!
		elif contains_pyfuncs(tok):
			pycode.append(tok)
		else:
			otoks.append(tok)
	toks = otoks
	# Translate the rest into a general query, so it can all be processed
	# homogeneously
	query = " and ".join(["(%s)" % tok for tok in toks])
	# Translate python code chunks into a single statement
	pycode = " & ".join(["(%s)" % tok for tok in pycode])
	# This helper function handles more specific band selection if available
	if "band" in cols: bandsel = lambda ftag, flavor: "(band = '%s')" % ftag
	else:              bandsel = lambda ftag, flavor: "(tube_flavor = '%s')" % flavor
	# 3. Loop through all referenced fields, and expand them if necessary.
	otoks = []
	prep_sel = {"set": False, "good": True, "bad": False}
	for tok, isname in fieldname_iter(query):
		# Split out unary op
		if len(tok) > 0 and tok[0] in "+-~":
			op, tok = tok[0], tok[1:]
		else: op = " "
		# Handle our formats
		if not isname: pass
		# Restrict as much as we can given any subobs constraints.
		# This isn't very much, since obsdb doesn't operate on the subobs level
		elif re.match(r"ws\d", tok):
			tok = tag_op("instr(wafer_slots_list, '%s')" % tok, op)
		elif tok in ["f030","f040"]: tok = tag_op(bandsel(tok, 'lf'),  op)
		elif tok in ["f090","f150"]: tok = tag_op(bandsel(tok, 'mf'),  op)
		elif tok in ["f220","f280"]: tok = tag_op(bandsel(tok, 'uhf'), op)
		# Optics tube
		elif tok in ["c1", "i1", "i2", "i3", "i4", "i5", "i6", "o1", "o2", "o3", "o4", "o5", "o6"]:
			tok = tag_op("(tube_slot = '%s')" % tok, op)
		# Pseudo-tags
		elif tok == "obs": tok = tag_op("(type='obs')", op)
		elif tok == "cmb": tok = tag_op("(type='obs' and subtype='cmb')", op)
		elif tok == "night": tok = tag_op("(mod(timestamp/3600,24) not between 11 and 23)", op)
		elif tok == "day": tok = tag_op("(mod(timestamp/3600,24) between 11 and 23)", op)
		elif tok.upper() in ["DARK","OPTC"]:
			if "det_type" in cols: tok = tag_op("(det_type = '%s')" % tok.upper(), op)
			else:                  tok = "1"
			det_type_set = True
		# Aliases
		elif tok == "t":   tok = "timestamp"
		elif tok == "baz": tok = "az_center"
		elif tok == "bel": tok = "el_center"
		elif tok == "roll": tok = "roll_center"
		elif tok == "waz": tok = "az_throw"
		elif tok == "wel": tok = "el_throw"
		elif tok == "dur": tok = "duration"
		elif tok == "nsamp": tok = "n_samples"
		# Constants
		elif tok == "tcorot": tok = "1749513600"
		elif tok == "tfoc2":  tok = "1756684800"
		# Don't interpret columns as tags if they conflict
		elif tok in cols: pass
		# Actual tags
		elif tok in tags:
			tok = tag_op("(obs_id in (select obs_id from tags where (tag = '%s')))" % tok, op)
		# Planet pseudo-tag
		elif tok == "planet":
			# The or 0 handles the case where there are no planets defined
			sel = "(" + " or ".join(["tag = '%s'" % planet for planet in planets if planet in tags]) + " or 0)"
			tok = tag_op("(obs_id in (select obs_id from tags where %s))" % sel, op)
		else:
			# Stuff that doesn't fit in. These are more limited than the others,
			# and can't be parts of complicated expressions.
			eq = "!=" if op in "-~" else "="
			if tok == "good":
				if   op == " ": prep_sel["good"], prep_sel["bad"] = True, False
				elif op == "+": prep_sel["good"] = True
				elif op == "-": prep_sel["good"] = False
				elif op == "~": prep_sel["good"], prep_sel["bad"] = False, True
				prep_sel["set"]   = True
			elif tok == "bad":
				if   op == " ": prep_sel["bad"], prep_sel["good"] = True, False
				elif op == "+": prep_sel["bad"] = True
				elif op == "-": prep_sel["bad"] = False
				elif op == "~": prep_sel["bad"], prep_sel["good"] = False, True
				prep_sel["set"]   = True
			# Unknown tag
			else:
				raise ValueError("Name '%s' not a recognized tag or obs table column!" % str(tok))
			# Handled separately, so just replace with a 1
			tok = "1"
		otoks.append(tok)
	query = "".join(otoks) if len(otoks) > 0 else "1"
	# Add default OPTC (non-dark) detector selection
	if not det_type_set and "det_type" in cols:
		query += " and (det_type = 'OPTC')"
	# Filter on valid preprocess
	pjoin = ""
	if not prep_sel["set"]: prep_sel["good"] = True
	if prep_sel["good"] and prep_sel["bad"]:
		# Want both good and bad, so no prep restriction required
		pass
	elif prep_sel["good"] and not prep_sel["bad"]:
		if "band" in cols:
			# band is present if we have a subobsdb
			pjoin = " join pre.map on obs.obs_id = pre.map.[obs:obs_id]"
			if "dets:wafer_slot" in pre_cols:
				pjoin += " and instr(obs.wafer_slots_list, pre.map.[dets:wafer_slot])"
			if "dets:wafer.bandpass" in pre_cols:
				pjoin += " and obs.band = pre.map.[dets:wafer.bandpass]"
		else:
			query += "  and obs.obs_id in (select [obs:obs_id] from pre.map)"
	elif prep_sel["bad"] and not prep_sel["good"]:
		# Selecting only bad was cumbersome to implement...
		raise ValueError("Selecting only bad observations not supported yet")
	else:
		# Neither good nor bad. So disqalify all obs
		query += " and 0"
	return bunch.Bunch(where=query, join=pjoin, idfile=idfile, pycode=pycode, slices=slices)

def tag_op(expr, op):
	if   op in " ": return expr # standard
	elif op in "+": return "1"  # + enables, but everything enabled by default
	elif op in "-~": return "not " + expr # - removes from set, ~ complements. Same for boolean
	else: raise ValueError("Invalid tag op '%s'" % str(op))

# FIXME: when splitting bands, we must remember that this
# also reduces ndet. Otherwise we will end up overestimating
# memory use later

def subobs_expansion(obsdb, tags=True):
	"""Given an obsdb SQL, return a subobsdb SQL using looping in SQL"""
	# Will keep all columns, except that wafer_slots_list will be replaced with
	# a single slot
	cols = [r[1] for r in obsdb.execute("PRAGMA table_info('obs')")]
	cols.remove("wafer_slots_list")
	# wafer slot expansion
	query = """with recursive
split(obs_id, slot, rest) as (
	select
		obs_id,
		substr(wafer_slots_list, 1, instr(wafer_slots_list||',', ',')-1),
		substr(wafer_slots_list || ',', instr(wafer_slots_list||',', ',')+1)
	from obs
	union all
	select
		obs_id,
		substr(rest, 1, instr(rest,',')-1),
		substr(rest, instr(rest,',')+1)
	from split
	where rest <> ''
),"""
	# band expansion
	case    = "case obs.tube_flavor"
	flavors = list(flavor_bands.keys())
	for flavor in flavors:
		case += " when '%s' then '" % flavor + ",".join(flavor_bands[flavor])+",'"
	case += " else 'f???,' end"
	query += """
list_bands(obs_id, bands) as (
	select
		obs_id,
		%s
	from obs
),
split_bands(obs_id, band, bands) as (
	select
		obs_id,
		substr(bands, 1, instr(bands, ',')-1),
		substr(bands, instr(bands, ',')+1)
	from list_bands
	union all
	select
		obs_id,
		substr(bands, 1, instr(bands, ',')-1),
		substr(bands, instr(bands, ',')+1)
	from split_bands
	where bands <> ''
)""" % case
	# dark expansion. Want "" and ":DARK" for each entry. Could do ":OPTC" instead of "",
	# but that makes the common case needlessly verbose and confusing. "" is slightly
	# confusing too, though, since one might think that "" would mean "all types".
	# But we don't support wildcard subids. A subid refers to a specific set of detectors.
	# the rest
	det_types = ["OPTC", "DARK"]
	for i, det_type in enumerate(det_types):
		idsuf = "" if det_type == "OPTC" else " || ':%s'" % det_type
		union = " union all " if i > 0 else ""
		query += """
%s select obs.obs_id || ':' || slot || ':' || band%s as subobs_id, '%s' as det_type, %s, slot as wafer_slots_list, band from obs join split on obs.obs_id = split.obs_id join split_bands on split_bands.obs_id = obs.obs_id where slot <> '' and band <> ''
""" % (union, idsuf, det_type, ", ".join(["obs.%s" % col for col in cols]))
	subobsdb = sqlite.SQL()
	with obsdb.attach(subobsdb, mode="rw"):
		obsdb.execute("create table other.obs as %s" % query)
		obsdb.execute("create table other.tags as select tags.* from tags where obs_id in (select obs_id from other.obs)")
	return subobsdb

# Have to hard-code this, I think
flavor_bands = {"lf":["f030","f040","DARK"], "mf":["f090","f150","DARK"], "uhf":["f220","f280","DARK"]}
planets = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "moon", "sun"]

##### Helpers #####

def get_tags(db): return [row[0] for row in db.execute("select distinct tag from tags")]

def fieldname_iter(query, quote="'\""):
	toks = utils.split_by_group(query, start=quote, end=quote)
	#fmt  = re.compile(r"~?\b[a-zA-Z]\w*\b")
	# word optionally preceded by ~, or +/-, but only if the latter can't be interpreted
	# as a binary operation. I had to use two groups to capture what amounts to the same
	# thin here because of limitations with lookbehind length variability
	fmt = re.compile(r"(?:(?:^|[\[({,])([+-]?\b[a-zA-Z]\w*\b))|(~?\b[a-zA-Z]\w*\b)") # urk
	current = ""
	for tok in toks:
		if len(tok) == 0: continue
		elif tok[0] in quote: current += tok
		else:
			# Ok, we're in a non-quoted section. Look for identifiers
			pos = 0
			while m := fmt.search(tok, pos):
				# Find which of the groups matched
				mind = [i for i,g in enumerate(m.groups()) if g is not None][0]+1 # yuck
				# pos:m.start() will be new, non-matching stuff
				current += tok[pos:m.start(mind)]
				# m.start():m.end() will be either a name or a keyword
				fieldname = m[mind]
				if fieldname.lower() in recognized_sql:
					current += fieldname
				else:
					# Ok, we actually have a fieldname
					if len(current) > 0:
						yield current, False
						current = ""
					yield m[mind], True
				pos = m.end(mind)
			# Handle what's left of the token
			current += tok[pos:]
	# Yield left-over stuff if present
	if len(current) > 0:
		yield current, False

# all is only used in select all, which is the default, and we
# need it for other things
recognized_sql = sqlite.keywords - set(["all"])

def add_list(db, table, vals):
	rows = [(val,ind) for ind, val in enumerate(vals)]
	db.execute("create table %s (id text, ind integer)" % table)
	db.executemany("insert into %s (id, ind) values (?,?)" % table, rows)
	db.execute("commit")

def load_obslist(fname):
	idlist = []
	# Read in the first column
	with open(fname, "r") as ifile:
		for line in ifile:
			toks = line.split()
			if len(toks) > 0: idlist.append(toks[0])
	# Check first entry to see if these are obsids or not
	ncolon = len(idlist[0].split(":")) if len(idlist) > 0 else 0
	if ncolon == 0:
		# We have a plain obslist
		obss    = idlist
		subobss = None
	else:
		# A subobs list
		subobss = idlist
		obss    = np.array([subobs.split(":")[0] for subobs in subobss])
		# Get rid of duplicates while preserving order
		uobss, inds = np.unique(obss, return_index=True)
		obss    = obss[np.sort(inds)]
	return {"obs":obss, "subobs":subobss}


# Python functions we will recognize
_pyfuncs = ["hits","maxel","minel"]
def contains_pyfuncs(s):
	for pyfunc in _pyfuncs:
		if pyfunc + "(" in s:
			return True
	return False

def finish_query(res_db, pycode, slices=[], sweeps=True, output="sogma"):
	from sotodlib import core
	if output not in ["sqlite", "resultset", "sogma"]:
		raise ValueError("Unrecognized output format '%s" % str(output))
	if output == "sqlite":
		# This format does not support pycode or slices
		return res_db
	info   = core.metadata.resultset.ResultSet.from_cursor(res_db.execute("select * from obs"))
	if output == "resultset" and not pycode:
		# Skip obsinfo construction if we don't need pycode
		for sel in slices: info = info[sel]
		return info
	# Estimate number of detectors. This will be an overestimate,
	# since some detectors will be cut. This assumes optc and dark
	# are the only possibilities, and uses hardcoded values.
	ndet = np.where(info["det_type"] == "OPTC",
		utils.dict_lookup(flavor_noptc_per_band, info["tube_flavor"]),
		utils.dict_lookup(flavor_ndark_per_band, info["tube_flavor"]),
	)
	dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("baz","d"),("waz","d"),("bel","d"),("wel","d"),("r","d"),("sweep","d",(6,2))]
	obsinfo = np.zeros(len(info), dtype).view(np.recarray)
	obsinfo.id    = info["subobs_id"]
	obsinfo.ndet  = ndet
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
	# Apply slices
	inds = np.where(good)[0]
	for sel in slices: inds = inds[sel]
	# And return in the requested format
	if output == "resultset": return resultset_subset(info, inds)
	else: return obsinfo[inds]

def add_tags_column(idb):
	resdb = sqlite.SQL()
	with resdb.attach(idb):
		# Copy over the tags table
		resdb.execute("create table tags as select * from other.tags")
		# Copy over obs table while adding a tags column
		resdb.execute("create table obs as select *, grouped.tagcol as tags from other.obs left join (select obs_id, group_concat(tag,',') as tagcol from other.tags group by tags.obs_id) as grouped on obs.obs_id = grouped.obs_id")
	return resdb

# Hard-coded raw wafer detector counts per band. Independent of
# telescope type, etc.
flavor_noptc_per_band = {"lf":118, "mf": 864, "uhf": 864}
flavor_ndark_per_band = {"lf":  4, "mf":  18, "uhf":  18}

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
sens_limits = {"f030":120, "f040":80, "f090":100, "f150":140, "f220":300, "f280":750}

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

def sensitivity_cut(rms_uKrts, sens_lim, med_tol=0.2, max_lim=100):
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

def gal_hit(sweep, dur, r, minlat=None, pad=1*utils.degree):
	# sweep[:,npoint,{ra,dec}]
	if minlat is None: minlat = 5*utils.degree
	from pixell import coordinates
	speed= 15*utils.degree/utils.hour
	ras, decs = sweep.T # [npoint,ntod]
	icoord = np.array([[ras,ras+dur*speed],[decs,decs]]) # [{ra,dec},2,npoint,ntod]
	lats   = coordinates.transform("equ", "gal", icoord)[1]
	# We hit the galaxy if lats either crosses zero, or if min(abs(lat))+r+pad < minlat
	lat1,lat2 = utils.minmax(lats,(0,1))
	hits = (lat1*lat2 < 0)|(np.min(np.abs(lats),(0,1))+r+pad < minlat)
	return hits

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
	if not pycode: return np.ones(len(obsinfo), bool)
	def pycode_hits(ra, dec=None, r=1):
		if isinstance(ra, str):
			# Special case: 'gal'. The optional second argument gives the min
			# galactic latitude
			if ra == "gal": return gal_hit(obsinfo.sweep, obsinfo.dur, obsinfo.r, minlat=dec, pad=r*utils.degree)
			# For a planet name, we look up the planet position and continue on to the
			# standard point hit
			else: pos = planet_pos(ra, obsinfo)
		else: pos = np.array([ra,dec])*utils.degree
		return point_hit(pos, obsinfo.sweep, obsinfo.dur, obsinfo.r, pad=r*utils.degree)
	def pycode_planet(name): return planet_pos(name, obsinfo)
	def pycode_maxel(name, dec=None):
		return np.max(hor_helper(name, obsinfo, dec=dec).el, (0,1))
	def pycode_minel(name, dec=None):
		return np.min(hor_helper(name, obsinfo, dec=dec).el, (0,1))
	# Would be nice to be able to do just hor(planet(name))[1]>0 to get tods
	# where some object is above the horizon, but they move too quickly
	# in these coordinates. Will instead make the more specialized maxel and minel
	globs = {}
	# Make numpy available, both with and without np
	globs.update(**vars(np))
	globs["np"] = np
	# Register our function
	globs["hits"]   = pycode_hits
	globs["planet"] = pycode_planet
	globs["maxel"]  = pycode_maxel
	globs["minel"]  = pycode_minel
	return eval(pycode, globs)

def planet_pos(name, obsinfo):
	from pixell import ephem
	name = name.lower()
	if   name == "planet":   name = config.get("planet_list")
	elif name == "asteroid": name = config.get("asteroid_list")
	names = name.split(",")
	poss = []
	for name in names:
		pos = ephem.eval(name, obsinfo.ctime)[0]
		poss.append(pos)
	return np.array(poss) # [nplanet,nobs,{ra,dec}]

def hor_helper(name, obsinfo, dec=None):
	from pixell import coordsys
	if isinstance(name, str):
		pos = planet_pos(name, obsinfo) # [nplanet,nobs,{ra,dec}]
	else:
		ra  = float(name)*utils.degree
		dec = float(dec)*utils.degree
		pos = np.array([ra,dec])[None,None]
	pos  = coordsys.Coords(ra=pos[...,0], dec=pos[...,1])
	times = np.array([obsinfo.ctime,obsinfo.ctime+obsinfo.dur])[:,None,:] # [2,*,nobs]
	hpos = coordsys.transform("equ", "hor", pos, ctime=times)
	return hpos # [2,*,nobs]

# Result-set workaround
def resultset_subset(resultset, inds):
	from sotodlib import core
	return core.metadata.resultset.ResultSet(resultset.keys, [resultset.rows[ind] for ind in inds])

def is_slice(s):
	return re.match(r"^[+-]?\d*:[+-]?\d*:?[+-]?\d*$", s) is not None

########################
# Contexts and configs #
########################

def find_so(): return os.environ["SOPATH"]
def find_cdir(telescope): return find_so() + "/metadata/%s/contexts" % telescope
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
	return find_label(context["metadata"], name).format(**context["tags"])
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
