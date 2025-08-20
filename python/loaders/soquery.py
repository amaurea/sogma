# Simplified obs query interface, similar to the one in ACT
# We want to be able to simply list keywords to match against,
# or artihmetic expressions. We should be able to select at
# sub-id level, but also access the corresponding query() data.
#
# The current approach is simple but hacky. It relies on rewriting
# the query into an SQL query without actually knowing SQL syntax.
# This architecture makes supporting python-level functions like
# hits() diffcult to shoehorn in.
#
# A more well-defined approach would be to requre that it's instead
# transformable into *python* syntax, and then evaluating that on
# abstract objects that simply capture what operations are being done
# on colums, tags or constants. Any purely-sql thing that that remains
# at the top-level "and" would be transformed into SQL and executed.
# What's left would be evaled in python based on the result of the
# SQL. Implementing this would be a lot of work though, so for now
# I'm sticking with the simple approach.
#
# Can be done with these steps:
# 1. separate out the sub-obs parts of the query. These are the wafer
#    slot and band. Maybe others. These won't have a special syntax,
#    because part of the goal of this module is to harmonize the interface,
#    but will instead simply be recognized by their names.
# 2. Get the full list of built-in tags with a query. Technically
#    optional, but lets us catch errors earlier and give more sensible error
#    messages.
# 3. translate non-subobs part of simple query to a normal obsdb query/tag expression
# 4. perform the query, getting a query result
# 5. expand obsids to subids
# 6. provide a mapping from subid to the query result

import re
import numpy as np
from pixell import sqlite
# only needed for split_outside and split_by_group
# speed up loading by skipping?
from pixell import utils

def eval_query(obsdb, simple_query, cols=None, tags=None, subobs=True, _obslist=None):
	"""Given an obsdb SQL and a Simple Query, evaluate the
	query and return a new SQL object (pointing to a temporary
	file) containing the resulting observations."""
	if cols is None: cols = sqlite.columns(obsdb, "obs")
	if tags is None: tags = get_tags(obsdb)
	if not simple_query: simple_query = "1"
	# Check if we have a subobsdb or not
	is_subobs = "subobs_id" in cols
	# Main parse of the simple query
	qwhere, idfile, pycode = parse_query(simple_query, cols, tags)
	qjoin = ""
	qsort = ""
	# Create database we will put the result in, and attach
	# it temporarily to obsdb so we can write to it
	res_db  = sqlite.SQL()
	with sqlite.attach(obsdb, res_db, "res", mode="rw"):
		if idfile:
			# Ok, an obslist was passed. The _oblist stuff is
			# so we can reuse an already read in obslist in the
			# subobs pass
			if _obslist is None: _obslist = load_obslist(idfile)
			if is_subobs and _obslist["subobs"] is not None:
				# Yes, obsdb, but indirectly res_db. Can't use res_db directly since
				# we're working on a copy of it due to how attach works
				add_list(obsdb, "res.targ", _obslist["subobs"])
				qjoin = "join res.targ on obs.subobs_id = res.targ.id"
			else:
				add_list(obsdb, "res.targ", _obslist["obs"])
				qjoin = "join res.targ on obs.obs_id = res.targ.id"
			qsort = "order by res.targ.ind"
		# Build the full query
		query = "select obs.* from obs %s where %s %s" % (qjoin, qwhere, qsort)
		# Use it to build the output table
		obsdb.execute("create table res.obs as %s" % query)
		# Also get the tags
		obsdb.execute("create table res.tags as select tags.* from tags where obs_id in (select obs_id from res.obs)")
		if idfile:
			# Drop the obslist table we created
			obsdb.execute("drop table res.targ")
	# Ok, by this we're detached again, and res_db contains the
	# resulting obs and tags tables. But we may need a second pass
	# if we're still at the obsdb level, but need a subobs db
	if not subobs: return res_db, pycode # didn't ask for subobs
	if is_subobs:  return res_db, pycode # already subobs
	# Ok, do the subobs expansion
	with subobs_expansion(res_db) as subobs_db:
		res_db.close() # don't need this one any more
		return eval_query(subobs_db, simple_query, tags=tags, _obslist=_obslist)

def parse_query(simple_query, cols, tags):
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
	qtags = []
	for tok, isname in fieldname_iter(query):
		# Handle our formats
		if not isname: pass
		# Restrict as much as we can given any subobs constraints.
		# This isn't very much, since obsdb doesn't operate on the subobs level
		elif re.match(r"ws\d", tok):
			tok = "instr(wafer_slots_list, '%s')" % tok
		elif tok in ["f030","f040"]: tok = bandsel(tok, 'lf')
		elif tok in ["f090","f150"]: tok = bandsel(tok, 'mf')
		elif tok in ["f220","f280"]: tok = bandsel(tok, 'uhf')
		# Optics tube
		elif tok in ["c1", "i1", "i2", "i3", "i4", "i5", "i6", "o1", "o2", "o3", "o4", "o5", "o6"]:
			tok = "(tube_slot = '%s')" % tok
		# Pseudo-tags
		elif tok == "cmb": tok = "(type='obs' and subtype='cmb')"
		elif tok == "night": tok = "(mod(timestamp/3600,24) not between 11 and 23)"
		elif tok == "day": tok = "(mod(timestamp/3600,24) between 11 and 23)"
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
		# Don't interpret columns as tags if they conflict
		elif tok in cols: pass
		# Finally handle tags. These must be alone, not part of
		# value comparisons, but they can be negated.
		else:
			# negation. This required a hack in the fieldname iterator,
			# making it treat the ~ as part of the name itself. Ugly.
			if tok.startswith("~"): tok, eq = tok[1:], "!="
			else: eq = "="
			# Handle normal tags
			if tok in tags:
				qtags.append("tag %s '%s'" % (eq, tok))
			# Handle some pseudo-tags
			elif tok == "planet":
				# The or 0 handles the case where there are no planets defined
				sel = "(" + " or ".join(["tag = '%s'" % planet for planet in planets if planet in tags]) + " or 0)"
				if eq == "!=": sel = "not %s" % sel
				qtags.append(sel)
			# Unknown tag
			else:
				raise ValueError("Name '%s' not a recognized tag or obs table column!" % str(tok))
			# Handled separately, so just replace with a 1
			tok = "1"
		otoks.append(tok)
	query = "".join(otoks) if len(otoks) > 0 else "1"
	# Filter on tags
	if len(qtags) > 0:
		# I tried exists too, but it was much slower
		query += " and obs_id in (select obs_id from tags where " + " and ".join(qtags) + ")"
	return query, idfile, pycode

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
	# the rest
	query += """
select obs.obs_id || ':' || slot || ':' || band as subobs_id, %s, slot as wafer_slots_list, band from obs join split on obs.obs_id = split.obs_id join split_bands on split_bands.obs_id = obs.obs_id where slot <> '' and band <> ''
""" % ", ".join(["obs.%s" % col for col in cols])
	subobsdb = sqlite.SQL()
	with obsdb.attach(subobsdb, mode="rw"):
		obsdb.execute("create table other.obs as %s" % query)
		obsdb.execute("create table other.tags as select tags.* from tags where obs_id in (select obs_id from other.obs)")
	return subobsdb

# Have to hard-code this, I think
flavor_bands = {"lf":["f030","f040"], "mf":["f090","f150"], "uhf":["f220","f280"]}
planets = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "moon", "sun"]

##### Helpers #####

def get_tags(db): return [row[0] for row in db.execute("select distinct tag from tags")]

def fieldname_iter(query, quote="'\""):
	toks = utils.split_by_group(query, start=quote, end=quote)
	fmt  = re.compile(r"~?\b[a-zA-Z]\w*\b")
	current = ""
	for tok in toks:
		if len(tok) == 0: continue
		elif tok[0] in quote: current += tok
		else:
			# Ok, we're in a non-quoted section. Look for identifiers
			pos = 0
			while m := fmt.search(tok, pos):
				# pos:m.start() will be new, non-matching stuff
				current += tok[pos:m.start()]
				# m.start():m.end() will be either a name or a keyword
				fieldname = m[0]
				if fieldname.lower() in sqlite.keywords:
					current += fieldname
				else:
					# Ok, we actually have a fieldname
					if len(current) > 0:
						yield current, False
						current = ""
					yield m[0], True
				pos = m.end()
			# Handle what's left of the token
			current += tok[pos:]
	# Yield left-over stuff if present
	if len(current) > 0:
		yield current, False

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
_pyfuncs = ["hits"]
def contains_pyfuncs(s):
	for pyfunc in _pyfuncs:
		if pyfunc + "(" in s:
			return True
	return False
