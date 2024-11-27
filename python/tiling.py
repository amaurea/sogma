import numpy as np
from pixell import wcsutils, enmap, utils, bunch
from scipy import spatial, optimize
import gpu_mm
from . import gmem

# Tiled map support. With this all the maps will be tiled, even
# when not using distributed maps, since that's what the GPU wants.
# We need to distinguish between 3-4 different layouts:
# 1. The target map geometry. Some subset of the full sky
# 2. The equivalent full sky geometry. Determines wrapping
# 3. The tiling. Tiles the full sky geometry, but can
#    overshoot to allow for deferred wrapping. Given by a tile
#    size and the number of global y and x tiles. Only
#    a subset of these tiles are ever actually used, so having
#    a big overshoot isn't expensive. 2x overshoot in x
#    should be a very conservative choice.
# 4. The tile ownership. A two-way list of tiles "owned" by
#    this mpi task. Used to avoid double-counting during
#    CG and reductions, since the same tile will be in
#    several mpi tasks' workspaces.
# 5. The local wrapped tiles. A two-way list of tiles
#    workspace tiles after wrapping
# 6. The lowal workspace tiles. A two-way list of the tiles
#    we will actually work on.
# We also want to precompute transfer information, including
# 1. global buffer size, offsets and indices into owned tile list.
#    a) buffer size and offsets in byte units. Requires us to know
#       the number of pre-dims and the bit depth.
#    b) agnostic. size and offsets multiplied by npre*nbyte_per_num
#       on-the-fly
#    Let's go with b), since the multiplication will take microseconds
#    for these short arrays.
# 2. local buffer size and indices into wrapped tiles
# 3. information needed for translating between work tiles and
#    wrapped tiles, if necessary.
# We will store this in the class Tiling, which won't concern
# itself with the actual data, just how it's laid out.

# Building this structure takes some prerequisites.
# 1. Given the map geometry, derive the fullsky geometry
# 2. Read in a list of tods. For each tod we should have at least a
#    ra,dec bounding box as part of the tod database.
# 3. Translate the bounding boxes to pixels. Add multiples of wrap
#    until min(x) >= 1
# 4. Define the global tiling
# 5. Distribute TOD ownership to mpi tasks using the bounding boxes and
#    per-tod weights like nsamp. This could be done by calculating the
#    mid-point of each bounding box, and then splitting by [ra], [ra,dec]
#    or [rise/set,ra,dec]. This step is hard to do optimally. We want
#    each to be as local as possible
# 6. Distribute tile ownership. This could be done blindly, but that
#    will require more communication than necessary. A Better version
#    would be to base it on the tod ownership and bounding boxes.
#    For very wide scans, it may be worth it to separate rising and
#    setting, since the tiles would be in a narrow curved parallelogram,
#    but this requires better bounding boxes than just a rectangle for the
#    local workspaces. We can add this later. The function that does this
#    would need the tiling, tod list and tod metdata, and tod ownerships.
#    It may be best to have the same function do both #5 and #6
# 7. Determine the local work tiles. This can be as simple as getting
#    the bounding box of the bounding boxes of our tods, and turning that
#    into tile lists.
# 8. Determine the wrapped local tiles from the local work tiles and
#    tiling and wrap. The number of wrapped tiles may be different from
#    the work tiles.
# 9. Build the communication buffer info

# Actually, the local tile ownership will be determined on-the-fly
# when the RHS is built, so we must split the construction into two parts
#
# Problem, I assume that cell_offsets is in cell units, but it's in
# pixel number units, so 12288 times larger. Need to decide which to use.
# In any case useful to have tsize=12288 stored somewhere.

# Better terminology: local → work, global → ?

# Let's assume w can wait to construct the TileDistribution until
# LocalPixelization is ready. Then the process would go
#
# 1. target geo → fullsky geo. This is necessary to avoid negative y
#    pixels, which I don't think gpu_mm will like
# 2. distribute tods (uses fullsky geo)
# 3. build LocalPixelization (uses fullsky geo)
# 4. build global tiling using LocalPixelization
# 5. build mpi info using LocalPixelization and global
#
# It would be nice if we could construct all this without
# needing to scan through all the tods, but we've already given
# up on that. So can just let SignalMap take care of these steps
#
# It's cumbersome to use LocalMap and LocalPixelization
# directly, since they miss variables like number of
# active cells and cell size, and LocalMap is flat.
# Could consider encapsulating them, but would have to
# make sure we still work with DynamicMap

class TileDistribution:
	def __init__(self, shape, wcs, local_pixelization, comm):
		# hardcoded constants
		self.tshape = (64,64)
		self.ncomp  = 3
		self.tsize  = self.ncomp*np.prod(self.tshape)
		self.shape, self.wcs = shape, wcs
		# Set up the observation workspace tiling. Would be nice if the
		# latter could be part of 
		lp = local_pixelization
		self.obs  = bunch.Bunch(lp=lp)
		self.obs.ntile = np.sum(lp.cell_offsets_cpu>=0)
		self.obs.size  = self.obs.ntile*self.tsize
		self.obs.cell_offsets = lp.cell_offsets_cpu # alias to avoid annoying name
		self.obs.cell_inds    = self.obs.cell_offsets//self.tsize
		# Set up the distributed tile tiling
		downer    = distribute_global_tiles_exposed_simple(self.obs.cell_inds, comm)
		self.dist = build_dist_tiling(downer, comm, tsize=self.tsize)
		# Set up the mpi communication info
		self.mpi  = build_mpi_info(self.dist.cell_inds, self.obs.cell_inds, comm)
	def dmap2gomap(self, dmap, gomap=None, buf=None):
		omap = self.dmap2omap(dmap)
		if gomap is None:
			gomap = self.gomap(buf=buf, dtype=omap.dtype)
		gmem.copy(omap, gomap.arr)
		return gomap
	def gomap2dmap(self, gomap, dmap=None):
		return self.omap2dmap(gomap.arr.get(), dmap=dmap)
	def dmap2omap(self, dmap, omap=None):
		"""Distributed to observational tile transfer on the CPU."""
		# dmap and omap should be numpy arrays with shape [ntile,:,tshape[0],tshape[1]]
		if omap is None: omap = np.zeros(self.oshape, dmap.dtype)
		omap  = omap.reshape(self.oshape) # to 4d in case 1d form is passed
		sbuf  = dmap[self.mpi.d2o_sinds]
		rbuf  = np.zeros(omap.shape, omap.dtype)
		sinfo = (self.mpi.d2o_scount*self.tsize, self.mpi.d2o_soffs*self.tsize)
		rinfo = (self.mpi.d2o_rcount*self.tsize, self.mpi.d2o_roffs*self.tsize)
		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (rbuf.reshape(-1),rinfo))
		omap[self.mpi.d2o_rinds] = rbuf
		return omap
	def omap2dmap(self, omap, dmap=None):
		"""Observational to distributed transfer and reduction on the CPU"""
		omap= omap.reshape(self.oshape) # to 4D in case 1D form is passed
		if dmap is None: dmap = self.dmap(dtype=omap.dtype)
		tshape= dmap.shape[1:]
		rbuf  = np.zeros((self.mpi.d2o_stot,)+tshape, dmap.dtype)
		sbuf  = np.ascontiguousarray(omap[self.mpi.d2o_rinds])
		rinfo = (self.mpi.d2o_scount*self.tsize, self.mpi.d2o_soffs*self.tsize)
		sinfo = (self.mpi.d2o_rcount*self.tsize, self.mpi.d2o_roffs*self.tsize)
		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (rbuf.reshape(-1),rinfo))
		# At this point we will have multiple contributions to each of our
		# distal tiles, which we need to reduce. Loop in python for now.
		# This performs 3*64*64 = 12k operations per iteration, so overhead
		# might not be too bad. But could try numba otherwise.
		dmap[:] = 0
		for i, gi in enumerate(self.mpi.d2o_sinds):
			dmap[gi] += rbuf[i]
		return dmap
	@property
	def oshape(self): return (self.obs .ntile,self.ncomp,self.tshape[0],self.tshape[1])
	@property
	def dshape(self): return (self.dist.ntile,self.ncomp,self.tshape[0],self.tshape[1])
	def dmap(self, dtype=np.float32, ncomp=3): return np.zeros(self.dshape, dtype)
	def omap(self, dtype=np.float32, ncomp=3): return np.zeros(self.oshape, dtype)
	def gomap(self, buf=None, dtype=np.float32):
		if buf: arr = buf .zeros(self.obs.size, dtype)
		else:   arr = cupy.zeros(self.obs.size, dtype)
		return gpu_mm.LocalMap(self.obs.lp, arr)

def build_dist_tiling(dist_owner, comm, tsize=1):
	dist = bunch.Bunch(owner=dist_owner)
	dist.mine  = dist_owner == comm.rank
	dist.ntile = np.sum(dist.mine)
	dist.size  = dist.ntile*tsize
	dist.cell_inds = np.full(dist_owner.shape, -1, dist_owner.dtype)
	dist.cell_inds[dist.mine] = np.arange(dist.ntile)
	dist.cell_offsets = np.full(dist_owner.shape, -1, dist_owner.dtype)
	dist.cell_offsets[dist.mine] = np.arange(dist.ntile)*tsize
	return dist

def distoff_owners(dmine, comm):
	nhit = utils.allreduce(dmine.astype(int), comm)
	assert np.max(nhit) <= 1, "Invalid distributed cell ownership: Cell owned by more than one task"
	owners = utils.allreduce(comm.rank*dmine, comm)
	owners[nhit==0] = -1
	return owners

def build_mpi_info(dist_inds, obs_inds, comm):
	"""Set up information needed for transfer between dist and obs tilings.
	Everything is in tile units. Multiply by tsize to get indices into flattened
	arrays."""
	# Build owners
	mpi   = bunch.Bunch(comm=comm)
	dmine = dist_inds >= 0
	omine = obs_inds  >= 0
	owners= distoff_owners(dmine, comm)

	# We order tiles in buffers in send order
	# We assume that there aren't any missing tiles in dist

	# Part 1: What to sent to each, for a dist→obs operation
	# My dist inds for all tiles everybody wants
	my_oty,  my_otx  = np.ascontiguousarray(np.nonzero(omine)) # [{ty,tx},nobs]
	all_oty          = utils.allgatherv(my_oty, comm)
	all_otx          = utils.allgatherv(my_otx, comm)
	all_on           = utils.allgather(np.sum(omine), comm)
	all_ranks        = np.repeat(np.arange(comm.size), all_on)
	mpi.d2o_sinds    = dist_inds[all_oty,all_otx]
	# Which of these are actually valid
	svalid           = mpi.d2o_sinds >= 0
	mpi.d2o_sinds    = mpi.d2o_sinds[svalid]
	# Corresponding send counts
	mpi.d2o_scount   = np.bincount(all_ranks, svalid, minlength=comm.size).astype(int)
	mpi.d2o_stot     = np.sum(mpi.d2o_scount)
	# The offset in the send array for each
	mpi.d2o_soffs    = utils.cumsum(mpi.d2o_scount)

	# Part 2: What to receive from each, for a dist→obs operation
	# Which of my obs tiles everybody has
	my_owners = owners[my_oty,my_otx]
	rvalid    = my_owners >= 0
	# How much to receive from each
	mpi.d2o_rcount = np.bincount(my_owners[rvalid], minlength=comm.size).astype(int)
	mpi.d2o_roffs  = utils.cumsum(mpi.d2o_rcount)
	# Mapping from our obs inds to the buffer
	# Need to know which tiles everybody will send to us
	#  [y,x for rank in size for y,x in zip(my_oty,my_otx) if owners[y,x]==rank]
	# How do I write this efficiently? Can collapse yx
	#  [yx for rank in size for yx in yxs if owners[yx]==rank]
	#  for yx in yxs: subs[owners[yx]].append(yx)
	# Could be done with an alltoallv, but that's overkill
	yxlist = [[] for i in range(comm.size)]
	for ty,tx in zip(my_oty,my_otx):
		if owners[ty,tx] >= 0:
			yxlist[owners[ty,tx]].append((ty,tx))
	recv_y, recv_x = np.concatenate(yxlist, 0).T
	mpi.d2o_rinds  = obs_inds[recv_y,recv_x]

	return mpi


#class TileDistribution:
#	def __init__(self, target, glob, mpi, loc=None):
#		self.target = target
#		self.glob   = glob
#		self.mpi    = mpi
#		self.loc    = loc
#	@staticmethod
#	def build(shape, wcs, obsinfo, comm, local_pixelization=None):
#		tshape = (64,64)
#		shape  = shape[-2:]
#		fshape, fwcs, off = infer_fullsky_geometry(shape, wcs)
#		# Ownership of the active global tiles
#		xperiod  = utils.nint(360/np.abs(wcs.wcs.cdelt[0]))
#		nty, ntx = utils.ceil_div(np.asanyarray(fshape[-2:]), tshape)
#		# TOD distribution
#		tinfo    = distribute_tods_semibrute(fshape, fwcs, obsinfo, comm.size)
#		# Find my active tiles
#		tiling= TileDistribution(
#			target=bunch.Bunch(shape=shape,  wcs=wcs,  off=off),
#			glob=bunch.Bunch(shape=fshape, wcs=fwcs, nty=nty, ntx=ntx, tshape=tshape, periodic=True,
#				ncomp=3)
#			mpi=bunch.Bunch(comm=comm), # will contain buffer info later
#		)
#		if local_pixelization:
#			tiling.finish(local_pixelization)
#		return tiling
#	def finalize(self, local_pixelization):
#		"""Finalize a tiling info givin an initalized one and a gpu_mm.LocalPixelization"""
#		self.loc  = local_pixelization
#		# Set up global stuff
#		owner     = distribute_global_tiles_exposed_simple(self.loc.cell_offsets_cpu, self.mpi.comm)
#		print("owner")
#		print(owner)
#		gmine     = owner == self.mpi.comm.rank
#		self.glob.owners       = owner
#		self.glob.ncell        = np.sum(gmine)
#		self.glob.cell_offsets[gmine] = np.arange(self.glob.ncell)
#		# Build the mpi buffers. We will be performing
#		# local→global reductions and global→local broadcasts
#		# In both cases we need to know how much data to send
#		# and receive from each task. Let's start with global→local.
#		# 1. Find out how many cells to receive from each
#		lmine     = self.loc.cell_offsets_cpu >= 0
#		self.nloc = np.sum(lmine)
#		self.locsize = self.loc.npix*self.glob.ncomp
#		sources   = self.glob.owners[lmine]
#		self.mpi.g2l_rcount = np.bincount(sources, minlength=self.mpi.comm.size)
#		self.mpi.g2l_roffs  = utils.cumsum(self.mpi.g2l_rcount)
#		# We don't need g2l_rinds, since local storage is already flat
#		# 2. Find out how many cells to send to each. Here we need some
#		# communication, since we don't know all the others' local
#		# ownership
#		all_nloc      = utils.allgather (np.sum(lmine), self.mpi.comm)
#		my_lcells_2d  = np.array(np.where(lmine)).T # [nloc,{ty,tx}]
#		all_lcells_2d = utils.allgatherv(my_lcells_2d, self.mpi.comm) # [sum(nloc),{ty,tx}]
#		all_ranks     = np.repeat(np.arange(self.mpi.comm.size), all_nloc)
#		# What glob tiles I would send if I had everything. This will contain
#		# -1 for ones I don't actually own, and so shouldn't send. With this,
#		# the send buffer can be constructed as gtiles[g2l_sinds],
#		# how much to send to each will be g2l_scount*tsize,
#		# and the offset to start at will be at g2l_soffs*tsize
#		print("self.glob")
#		print(self.glob)
#		print("self.loc.cell_offsets_cpu")
#		print(self.loc.cell_offsets_cpu)
#
#		g2l_sinds   = self.glob.cell_offsets[all_lcells_2d[:,0],all_lcells_2d[:,1]]
#		print("glob.cell_offsets", self.glob.cell_offsets)
#		print("all_lcells_2d", all_lcells_2d)
#		print("g2l_sinds", g2l_sinds)
#		self.mpi.g2l_scount  = np.bincount(all_ranks, g2l_sinds>=0, minlength=self.mpi.comm.size).astype(int)
#		self.mpi.g2l_sinds   = g2l_sinds[g2l_sinds>=0]
#		self.mpi.g2l_soffs   = utils.cumsum(self.mpi.g2l_scount)
#		self.mpi.g2l_stot    = np.sum(self.mpi.g2l_scount)
#	def glob2loc_gpu(self, gmap, glmap=None, buf=None):
#		lmap = self.glob2loc(gmap)
#		if glmap is None: glmap = self.glmap(buf, dtype=lmap.dtype)
#		gmem.copy(lmap, glmap.arr)
#		return glmap
#	def loc2glob_gpu(self, glmap, gmap=None):
#		return self.loc2glob(glmap.arr.get(), gmap=gmap)
#	def glob2loc(self, gmap, lmap=None):
#		"""Global to local tile transfer on the CPU."""
#		# gmap and lmap should be numpy arrays with shape [ntile,:,tshape[0],tshape[1]]
#		if lmap is None:
#			lmap = np.zeros((self.nloc,)+gmap.shape[1:], gmap.dtype)
#		lmap= self.lunzip(lmap) # to 4D in case 1D form is passed
#		tsize = np.prod(gmap.shape[1:])
#		sbuf  = gmap[self.mpi.g2l_sinds]
#		sinfo = (self.mpi.g2l_scount*tsize, self.mpi.g2l_soffs*tsize)
#		rinfo = (self.mpi.g2l_rcount*tsize, self.mpi.g2l_roffs*tsize)
#		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (lmap.reshape(-1),rinfo))
#		return lmap
#	def loc2glob(self, lmap, gmap=None):
#		"""Local to global transfer and reduction on the CPU"""
#		lmap= self.lunzip(lmap) # to 4D in case 1D form is passed
#		if gmap is None: gmap = self.gmap(dtype=lmap.dtype)
#		tshape= gmap.shape[1:]
#		tsize = np.prod(tshape)
#		rbuf  = np.zeros((self.mpi.g2l_stot,)+tshape, gmap.dtype)
#		sbuf  = lmap
#		rinfo = (self.mpi.g2l_scount*tsize, self.mpi.g2l_soffs*tsize)
#		sinfo = (self.mpi.g2l_rcount*tsize, self.mpi.g2l_roffs*tsize)
#		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (rbuf.reshape(-1),rinfo))
#		# At this point we will have multiple contributions to each of our
#		# global tiles, which we need to reduce. Loop in python for now.
#		# This performs 3*64*64 = 12k operations per iteration, so overhead
#		# might not be too bad. But could try numba otherwise.
#		gmap[:] = 0
#		for i, gi in enumerate(self.mpi.g2l_sinds):
#			gmap[gi] += rbuf[i]
#		return gmap
#	def lunzip(self, lmap):
#		"""Reshape a 1d local tile representation to [ntile,ncomp,tshape[0],tshape[1]].
#		lmap must already have this ordering!"""
#		return lmap.reshape(self.nloc,-1,self.glob.tshape[0],self.glob.tshape[1])
#	def gmap(self, dtype=np.float32, ncomp=3):
#		return np.zeros((self.glob.ncell,ncomp,self.glob.tshape[0],self.glob.tshape[1]), dtype)
#	def lmap(self, dtype=np.float32, ncomp=3):
#		return np.zeros((self.nloc,ncomp,self.glob.tshape[0],self.glob.tshape[1]), dtype)
#	def glmap(self, buf, dtype=np.float32):
#		return gpu_mm.LocalMap(self.loc, buf.zeros(self.locsize, dtype))

# We want to distribute both tod ownership and tile ownership.
# We also need to decide which tiles should exist at all, so we
# don't use more memory than necessary.
# Alternatives:
#  1. global tiling = full sky, but only tiles overlapping with
#     target geometry have ownership
#  2. global tiling = target geometry, but with whole tiles
# In either case, need to figure out how to handle tods that
# go outside the global tiling.
#  1. don't allocate these tiles in the work tiling either. Only
#     possible if pointing matrix handles skipping missing tiles.
#     This seems to be the case
#  2. allocate them in work, but skip them when reducing
# Let's go with global tiling = target geometry and skipping
# irrelevant local tiles.

def infer_fullsky_geometry(shape, wcs):
	"""Given a cylindrical geometry shape,wcs, return the fullsky geometry
	it is embedded in, as well as the pixel offset of the map relative
	to this."""
	owcs = wcs.deepcopy()
	nx   = utils.nint(360/np.abs(wcs.wcs.cdelt[0]))
	y1,y2= enmap.sky2pix(shape, wcs, np.array([[-np.pi/2,np.pi/2],[0,0]]))[0]
	# We want every valid coordinate to have a non-negative y pixel value,
	# so we don't get anything chopped off or wrapped in this direction.
	# We also need room for the interpolation context. If the south pole
	# is at y1:
	#  * nn:     iy1 = nint(y1)    ; iy2 = nint(y2)
	#  * lin:    iy1 = floor(y1)   ; iy2 = floor(y2)+1
	iy1 = utils.floor(y1)
	iy2 = utils.floor(y2)+1
	ny  = iy2+1-iy1
	owcs.wcs.crpix[1] -= iy1
	off = utils.nint(owcs.wcs.crpix-wcs.wcs.crpix)[::-1]
	return (ny,nx), owcs, off

def distribute_global_tiles_bands(fshape, fwcs, off, shape, tshape, nsplit):
	"""Sets up the global tile ownership. This determines which
	mpi task should own each of the global tiles *that needs to
	be stored*. Returns owner[nty,ntx], where the value in each
	cell is which mpi task owns that cell, or -1 if the cell is not used.
	Arguments:
	* fshape, fwcs: Geometry of the full sky
	* off: (y,x) pixel offset for the start of the target area.
	    Tiles that don't overlap with this will not be stored
	* shape: (ny,nx) shape of the target area
	* tshape: (y,x) tile shape
	* nsplit: Number of mpi tasks to split across"""
	# Full number of tiles
	nty, ntx = utils.ceil_div(np.asanyarray(fshape[-2:]), tshape)
	# Determine ownership
	off = np.asarray(off)
	t1  = off//tshape
	t2  = (off+shape-1)//tshape+1
	owner = np.full((nty,ntx),-1,int)
	nacty, nactx = t2-t1
	owner[t1[0]:t2[0],t1[1]:t2[1]] = np.arange(nactx*nsplit//nactx)
	return owner

def distribute_global_tiles_exposed_simple(loc_cells, comm):
	"""Sets up the global tile ownership based on local workspace
	cells. This variant does the simplest possible thing, and doesn't
	care about maximizing overlap between global and local cells."""
	hits  = utils.allreduce((loc_cells>=0).astype(int), comm)
	mask  = hits>0
	nhit  = np.sum(mask)
	owner = np.full(hits.shape, -1, int)
	owner[mask] = np.arange(nhit)*comm.size//nhit
	return owner

def distribute_tods_semibrute(shape, wcs, obsinfo, nsplit, npass=2, niter=10, verbose=False):
	"""Split into nsplit groups such that the groups are
	are approximately maximally-separated in state-space,
	and each group has approximately the same weight.
	This function is pretty inefficient, but it does its job. It's probably still
	fast enough. It scales as nobs*nsplit, and takes 280 ms for 21k tods split over 100
	mpi tasks. We can improve it later if necessary. It does a reasonably good job at
	optimizing both the sample volume and compactness per mpi task."""
	ntod  = len(obsinfo)
	# Handle special cases
	if nsplit == 1: return [np.arange(ntod)]
	if ntod <= nsplit: return [[i] for i in range(ntod)]+[[] for i in range(ntod,nsplit)]
	# Then the main case
	pix   = enmap.sky2pix(shape, wcs, [obsinfo.sweep[:,:,1],obsinfo.sweep[:,:,0]]) # [{y,x},ntod,npoint]
	state = np.moveaxis(pix,1,0).reshape(ntod,-1) # [ntod,ncoord]
	# TODO: Do I need to unwind state here?
	weight= obsinfo.ndet*obsinfo.nsamp
	# Start from the point with the lowest state-sub
	refs, dists  = find_refs_rimwise(state, nsplit)
	gid, gweight = assign_to_group_semibrute(dists, weight, nsplit, niter=niter, state=state, verbose=verbose)
	for ipass in range(1, npass):
		# Build groups given current refs
		# Then update refs to be close to the center of each group
		refs  = find_central_element(state, gid)
		dists = calc_dist(state[refs,None],state)
		gid, gweight = assign_to_group_semibrute(dists, weight, nsplit, niter=niter, state=state, verbose=verbose)
	return bunch.Bunch(owner=gid, weights=gweight)

###########
# Helpers #
###########

def calc_dist(a,b): return np.sum((a-b)**2,-1)**0.5
def find_refs_rimwise(state, nsplit, sort=True):
	refs  = np.zeros(nsplit,int)
	dists = np.zeros((nsplit,len(state)))
	refs[0] = np.argmin(np.sum(state,-1))
	dists[0]= calc_dist(state, state[refs[0]])
	# Find the other refs as the point furthest away from all the others
	for i in range(1, nsplit):
		refs[i]  = np.argmax(np.min(dists[:i],0))
		dists[i] = calc_dist(state, state[refs[i]])
	# Sort the groups from left to right
	if sort:
		order = np.argsort(np.mean(state[refs],1))
		refs  = refs[order]
		dists = dists[order]
	return refs, dists
def assign_to_group_semibrute(dists, weight, nsplit, niter=10, state=None, verbose=False):
	# The scale iteration here is inefficient. Example for 209 tods split 15-wise,
	# after two passes:
	#  niter  evenness   compactness
	#      1      1.66           164
	#      2      2.11           155
	#      5      1.39           172
	#     10      1.21           170
	#     20      1.20           169
	#     50      1.17           217
	#    100      1.16           215
	#    200      1.06           174
	#    500      1.09           265
	#   1000      1.11           173
	#   2000      1.06           197
	# Some good solutions are found, e.g. the one at 200 steps, but progress is uneven
	# and slow. However, it still does better than a non-linear search
	scales = np.ones(nsplit)
	target = np.sum(weight)/nsplit
	for it in range(niter):
		gid     = np.argmin(dists*scales[:,None],0)
		gweight = np.bincount(gid, weight)
		if verbose:
			compactness = np.mean(calc_compactness(state, gid))
			balance     = np.max(gweight/target)
			np.savetxt("/dev/stdout", np.array([balance,compactness])[None], fmt="%10.7f")
		scales *= (gweight/target)**(1/(it+1))
	return gid, gweight
def find_central_element(state, gid):
	uid, order, edges = utils.find_equal_groups_fast(gid)
	refs = []
	for gi in range(len(uid)):
		sub    = state[order[edges[gi]:edges[gi+1]]]
		center = np.mean(sub,0)
		dists  = calc_dist(sub, center)
		ref    = order[edges[gi]+np.argmin(dists)]
		refs.append(ref)
	return refs
def calc_compactness(state, gid):
	nhit    = np.bincount(gid)
	centers = (utils.bincount(gid, state.T)/nhit).T
	# This method is a bit slower, but more numerically stable
	offs    = state-centers[gid]
	comps   = (utils.bincount(gid, offs.T**2)/nhit)**0.5
	return comps
