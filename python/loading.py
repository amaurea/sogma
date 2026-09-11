from pixell import utils, config
import numpy as np
from . import device

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

def get_loader(dbfile, type="auto", group="obs", split=None, tsplit=None, dev=None, dtype=np.float32):
	plain = type.endswith("-")
	if plain: type = type[:-1]
	if type == "auto":
		if dbfile.endswith("npy"): type = "simple"
		else: type = "sofast"
	if type == "simple":
		from .loaders.simple import SimpleLoader
		loader = SimpleLoader(dbfile, dev=dev, dtype=dtype)
	elif type == "sofast":
		from .loaders.sofast import SoFastLoader
		loader = SoFastLoader(dbfile, dev=dev, dtype=dtype)
	elif type = "soslow":
		raise NotImplementedError
	else:
		raise ValueError("Unrecognized loader type '%s'" % str(type))
	# Wrap in post-loader unless plain
	if not plain:
		from .loaders.post import PostLoader
		loader = PostLoader(loader, group=group, split=split, tsplit=tsplit, dev=dev)
	return loader
