from pixell import utils, config, colors, bunch
import numpy as np, os
from . import device, errors
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

class LoadHelper:
	def __init__(self, fname_fun=None, empty_fun=None):
		# Must import before ArgumentParser is initialized to import args from config
		import numpy as np, os, time, sys
		from pixell import config, utils, mpi, colors, bench
		from . import logging, mapmaking, loading, device, gutils
		# Have to do all imports that could contain configuration before
		# building config.ArgumentParser because these may define variables.
		# This means we can't bail out early when called with wrong arguments;
		# we have to wait for potentially slow imports first. Oh well.
		parser = config.ArgumentParser("sogma", conflict_handler="resolve")
		parser.add_argument(      "--odir",    type=str, default=".")
		parser.add_argument("-C", "--context", type=str, default="lat", help="Either a config.yaml, context.yaml or a telescope name, like lat, satp1, etc.. The most general is config.yaml, which works with both the sofast and soslow loaders. Using just a context only works with sofast, and must be a context that includes the preprocess database. Telescope names expand to the default preprocess context using $SOPATH, which should point to the SO top-level directory (the one that contains the metadata directory).")
		parser.add_argument("-Q", "--query",   type=str, default=None)
		parser.add_argument("-p", "--prefix",  type=str, default=None)
		parser.add_argument("-v", "--verbose", action="count", default=1)
		parser.add_argument("-q", "--quiet",   action="count", default=0)
		parser.add_argument("-L", "--loader",  type=str, default="auto")
		parser.add_argument("-D", "--device",  type=str, default="auto")
		parser.add_argument(      "--ignore",  type=str, default="recover")
		parser.add_argument("-j", "--joint",   type=str, default="oband", help="full, obs, oband, wafer, wband or none")
		parser.add_argument("-s", "--split",   type=float, default=0.75, help="Split obs-groups in time to keep them smaller than this many giga-samples. Larger groups use more memory. Small groups have more edge effects and may be slower to read in.")
		parser.add_argument(      "--tsplit",  type=float, default=None, help="Split obs-groups in time to keep them shorter than this many seconds. See also --split")
		parser.add_argument(      "--sel",     type=str, default=":")
		parser.add_argument(      "--dets",    type=str, default=None, help="Path to file with white-list of channel ids, e.g. sch_ufm_mv21_1755145838_3_338. See --detids if you want to use detector ids instead")
		parser.add_argument(      "--detids",  type=str, default=None, help="Path to file with white-list of detector ids, e.g. Mv21_f090_Ar00c02A. One per line. See --dets if you want to use channel ids instead")
		parser.add_argument("-c", "--cont",    action="store_true")
		self.parser = parser
		self.comm   = mpi.COMM_WORLD
		self.fname_fun = fname_fun
		self.empty_fun = empty_fun
	def add_argument(self, *args, **kwargs):
		self.parser.add_argument(*args, **kwargs)
	def init(self):
		import numpy as np, os, time, sys
		from pixell import utils, mpi, colors, bench
		from . import logging, mapmaking, loading, device, gutils, soeph
		args = self.parser.parse_args()
		mpi.install_abort_hook()
		dev  = device.get_device(args.device)
		# Set up our logging
		verbosity  = args.verbose-args.quiet
		L = logging.Logger(dev, id=self.comm.rank, level=verbosity, fmt="{id:3d} {t:8.3f} {mem:5.2f} {dmem_pools:5.2f} {dmem_rest:5.2f} {dmem_unknown:5.2f} {msg:s}").setdefault()
		bench.set_verbose(verbosity >= 4)
		L.print("Init", level=0, id=0, color=colors.lgreen)
		# Benchmarking
		bench.set_tfun(dev.time)
		# Optional detector restriction
		dets   = gutils.read_detnames(args.dets)   if args.dets   is not None else None
		detids = gutils.read_detnames(args.detids) if args.detids is not None else None
		# Ephemerides
		soeph.setup()
		# Set up our data loader
		split  = args.split*1e9 if args.split else None
		fast_fail = args.ignore == "none"
		loader = loading.get_loader(args.context, type=args.loader, split=split, tsplit=args.tsplit, dev=dev, fast_fail=fast_fail, group=args.joint)
		linfo  = loader.query(args.query, dets=dets, detids=detids)
		linfo  = linfo[gutils.parse_slice(args.sel)]
		if linfo.nobs == 0:
			L.print("No tods selected", level=0, id=0, color=colors.red)
			raise errors.NothingLeft("No data selected")
		L.print("Processing %d tods with %d mpi tasks" % (linfo.nobs, self.comm.size), level=0, id=0, color=colors.lgreen)
		linfo.prealloc()
		prefix = args.odir + "/"
		if args.prefix: prefix += args.prefix + "_"
		utils.mkdir(args.odir)
		# Write out our arguments
		if self.comm.rank == 0:
			with open(prefix + "args.txt", "w") as ofile:
				ofile.write(" ".join(sys.argv) + "\n")
			config.save(prefix + "config.txt")
		# Useful to have in log-files too
		L.print(" ".join(sys.argv), level=0, id=0, color=colors.lgreen)
		if dev is None: dev = device.get_device()
		# Register interface
		locs = locals()
		# What to catch during iteration
		if args.ignore == "none": self.catch = ()
		catch_map = {
				"none":     (),
				"load":     (errors.LoadError,),
				"recover":  (errors.RecoverableError,),
				"expected": (errors.ExpectedError,),
				"all":      (Exception,),
		}
		self.catch = catch_map[args.ignore]
		for name in ["args", "dev", "linfo", "L", "verbosity", "dets", "detids", "prefix"]:
			setattr(self, name, locs[name])
	def get_obs(self, ind=0, id=None):
		if id is None: id = self.linfo.obsinfo.id[ind]
		return self.linfo.load(id, dets=self.dets, detids=self.detids)
	def obs_iter(self, inds=None, fname_fun=None, empty_fun=None):
		from . import tiling
		if fname_fun is None: fname_fun = self.fname_fun
		if empty_fun is None: empty_fun = self.empty_fun
		if empty_fun is None: empty_fun = lambda self,id: fname_fun(self,id).replace(".txt", ".empty")
		if inds is None: inds = range(self.comm.rank, self.linfo.nobs, self.comm.size)
		for ind in inds:
			id = self.linfo.obsinfo.id[ind]
			subpre = self.prefix + id.replace(":","_") + "_"
			if self.args.cont and fname_fun and (os.path.isfile(fname_fun(self, id)) or os.path.isfile(empty_fun(self, id))):
				self.L.print("Skipped %s: done" % (id), level=2, color=colors.gray)
				continue
			try:
				obs = self.get_obs(ind)
				yield bunch.Bunch(obs=obs, ind=ind, id=id, subpre=subpre)
				del obs
			except self.catch as e:
				if empty_fun: utils.touch(empty_fun(self, id))
				self.L.print("Skipped %s: %s" % (id, str(e)), level=2, color=colors.red)
