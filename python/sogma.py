# Should this be moved to bin?
# Should this be turned into a specialized SO mapmaker class?

if __name__ == "__main__":
	# Do argparse first to give quick feedback on wrong arguments
	# without waiting for slow imports
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("ifiles", nargs="+")
	parser.add_argument("area")
	parser.add_argument("odir")
	parser.add_argument("-p", "--prefix",  type=str, default=None)
	parser.add_argument("-v", "--verbose", action="count", default=1)
	parser.add_argument("-q", "--quiet",   action="count", default=0)
	parser.add_argument("-n", "--maxfiles",type=int, default=0)
	args = parser.parse_args()

import numpy as np, os, time
from pixell import enmap, utils, mpi, colors
import cupy
import gpu_mm
from . import gutils, logging, gmem
from .gmem import scratch

if __name__ == "__main__":
	ifiles = files.get_filelist(args.ifiles)
	if args.maxfiles: ifiles = ifiles[:args.maxfiles]
	nfile  = len(ifiles)
	comm   = mpi.COMM_WORLD
	dtype_tod  = np.float32
	dtype_map  = np.float32
	shape, wcs = enmap.read_map_geometry(args.area)
	logging.L = L = Logger(id=comm.rank, level=args.verbose-args.quiet)
	L.print("Init", level=0, id=0, color=colors.lgreen)
	# Set up our gpu scratch memory. These will be used for intermediate calculations.
	# We do this instead of letting cupy allocate automatically because cupy garbage collection
	# is too slow. We pre-allocate 6 GB of gpu ram. We also allocate some gpu ram in the plan_cache
	scratch.tod      = CuBuffer(1000*250000*4,     name="tod")              # 1 GB
	scratch.ft       = CuBuffer(1000*250000*4,     name="ft")               # 1 GB
	scratch.pointing = CuBuffer(3*1000*250000*4,   name="pointing")         # 3 GB
	scratch.map      = CuBuffer(3*shape[-2]*(shape[-1]+512)*4, name="map")  # 0.75 GB
	scratch.nmat_work= CuBuffer(40*1000*1000*4,    name="nmat_work")        # 0.15 GB
	scratch.cut      = CuBuffer(1000*750000*4,     name="cut")              # 0.3  GB (up to 30% cut)
	# Disable the cufft cache. It uses too much gpu memory
	cupy.fft.config.get_plan_cache().set_memsize(0)
	L.print("Mapping %d tods with %d mpi tasks" % (nfile, comm.size), level=0, id=0, color=colors.lgreen)
	prefix = args.odir + "/"
	if args.prefix: prefix += args.prefix + "_"
	utils.mkdir(args.odir)
	# Set up the signals we will solve for
	#signal_map = SignalMap(shape, wcs, comm, dtype=dtype_map)
	signal_map = mapmaking.SignalMapGpu(shape, wcs, comm, dtype=np.float32)
	# FIXME: poly cuts need a better preconditioner or better basis.
	# The problem is probably that the legendre polynomials lose their orthogonality
	# with the truncated block approach used here
	signal_cut = mapmaking.SignalCutPolyGpu(comm, precon="var")
	signal_cut = mapmaking.SignalCutFullGpu(comm)
	# Set up the mapmaker
	mapmaker = mapmaking.MLMapmaker(signals=[signal_cut,signal_map], dtype=dtype_tod, verbose=True, noise_model=NmatDetvecsGpu())
	# Add our observations
	for ind in range(comm.rank, nfile, comm.size):
		ifile = ifiles[ind]
		id    = ".".join(os.path.basename(ifile).split(".")[:-1])
		t1    = time.time()
		data  = read_tod(ifile)
		t2    = time.time()
		with gmem.leakcheck("mapmaker.add_obs"):
			mapmaker.add_obs(id, data, deslope=False)
			gmem.gpu_garbage_collect()
		del data
		t3    = time.time()
		L.print("Processed %s in %6.3f. Read %6.3f Add %6.3f" % (id, t3-t1, t2-t1, t3-t2))
	# Solve the equation system
	for step in mapmaker.solve():
		L.print("CG %4d %15.7e (%6.3f s)" % (step.i, step.err, step.t), id=0, level=1, color=colors.lgreen)
		if step.i % 10 == 0:
			for signal, val in zip(mapmaker.signals, step.x):
				if signal.output:
					signal.write(prefix, "map%04d" % step.i, val)
