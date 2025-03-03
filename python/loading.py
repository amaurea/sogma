from pixell import utils
import numpy as np
import cupy

# Loading of obs lists, metadata and data. The interface needs at least these:
# 1. get a list of observations
# 2. get scanpat info for each observation
# 3. load a calibrated observation

# The last part would ideally be a standard axismanager, but for now it will just
# be a bunch, so the rest of the code doesn't need to be changed

# What should be an "observation"? At least for the sat, files are 10 minutes of 1 wafer,
# but this 10-min-division is treated as an implementation detail, and I can't rely on
# this being available directly in the future. The basic unit is around one hour.

# Getting the observations and getting the info often involves loading the same
# database, so having them as independent functions is wasteful. Let's make a
# class that can be queried.

def Loader(dbfile, type="auto", mul=32):
	if type == "auto":
		if dbfile.endswith(".yaml"): type = "sofast"
		else: type = "simple"
	if type == "simple":
		from .loaders.simple import SimpleLoader
		return SimpleLoader(dbfile, mul=mul)
	elif type == "sofast":
		from .loaders.sofast import SoFastLoader
		return SoFastLoader(dbfile, mul=mul)
	elif type == "soslow":
		from .loaders.soslow import SotodlibLoader
		return SotodlibLoader(dbfile, mul=mul)
	else: raise ValueError("Unrecognized loader type '%s'" % str(type))

def get_filelist(ifiles):
	fnames = []
	for gfile in ifiles:
		for ifile in sorted(utils.glob(gfile)):
			if ifile.startswith("@"):
				with open(ifile[1:],"r") as f:
					for line in f:
						fnames.append(line.strip())
			else:
				fnames.append(ifile)
	return fnames

def check_data_requirements(data):
	# The fields that have strict type/contig requirements
	fields = [
		["tod", cupy.ndarray, 2, np.float32],
	]
	for fname, typ, ndim, dtype in fields:
		d = data[fname]
		if not isinstance(d, typ):
			raise ValueError("Field %s type %s is not a subtype of %s" % (fname, str(type(d)), str(typ)))
		if d.ndim != ndim:
			raise ValueError("Field %s ndim %d != %d" % (fname, d.ndim, ndim))
		if d.dtype != dtype:
			raise ValueError("Field %s dtype %s != %s" % (fname, str(d.dtype), str(dtype)))
