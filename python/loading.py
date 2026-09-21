from pixell import utils, config
import numpy as np
from . import device
# Would prefer to not load these unconditionally, but the config system needs them.
# Should find a better solution...
from .loaders import simple, post, sofast

# Loading of obs lists, metadata and data

def get_loader(dbfile, type="auto", group="obs", split=None, tsplit=None, fast_fail=False, dev=None, dtype=np.float32):
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
	elif type == "soslow":
		raise NotImplementedError
	else:
		raise ValueError("Unrecognized loader type '%s'" % str(type))
	# Wrap in post-loader unless plain
	if not plain:
		from .loaders.post import PostLoader
		loader = PostLoader(loader, group=group, split=split, tsplit=tsplit, fast_fail=fast_fail, dev=dev)
	return loader
