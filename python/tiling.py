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
	def __init__(self, shape, wcs, local_pixelization, comm, pixbox=None):
		# hardcoded constants
		self.tshape = (64,64)
		self.ncomp  = 3
		self.tsize  = self.ncomp*np.prod(self.tshape)
		# Our geometry
		shape = shape[-2:]
		self.shape, self.wcs = shape, wcs
		# Sub-rectangle we're actually interested in.
		# Used when reducing the map
		self.pixbox = pixbox
		if pixbox is not None:
			_, self.pwcs = enmap.crop_geometry(shape, wcs, pixbox=pixbox)
		else: self.pwcs = wcs
		# Set up the work tiling. Would be nice if the
		# latter could be part of 
		lp = local_pixelization
		self.work  = bunch.Bunch(lp=lp)
		self.work.ntile = np.sum(lp.cell_offsets_cpu>=0)
		self.work.size  = self.work.ntile*self.tsize
		self.work.cell_offsets = lp.cell_offsets_cpu # alias to avoid annoying name
		self.work.cell_inds    = self.work.cell_offsets//self.tsize
		# Set up the distributed tile tiling
		downer    = distribute_global_tiles_exposed_simple(self.work.cell_inds, comm)
		self.dist = build_dist_tiling(downer, comm, tsize=self.tsize)
		# Set up the mpi communication info
		self.mpi  = build_mpi_info(self.dist.cell_inds, self.work.cell_inds, comm)
		self.ompi = build_omap_mpi(shape, wcs, self.tshape, self.dist.cell_inds, comm, pixbox=pixbox)

	def dmap2gwmap(self, dmap, gwmap=None, buf=None):
		wmap = self.dmap2wmap(dmap)
		if gwmap is None:
			gwmap = self.gwmap(buf=buf, dtype=wmap.dtype)
		gmem.copy(wmap, gwmap.arr)
		return gwmap
	def gwmap2dmap(self, gwmap, dmap=None):
		return self.wmap2dmap(gwmap.arr.get(), dmap=dmap)
	def dmap2wmap(self, dmap, wmap=None):
		"""Distributed to observational tile transfer on the CPU."""
		# dmap and wmap should be numpy arrays with shape [ntile,:,tshape[0],tshape[1]]
		if wmap is None: wmap = np.zeros(self.oshape, dmap.dtype)
		wmap  = wmap.reshape(self.oshape) # to 4d in case 1d form is passed
		sbuf  = dmap[self.mpi.d2w_sinds]
		rbuf  = np.zeros(wmap.shape, wmap.dtype)
		sinfo = (self.mpi.d2w_scount*self.tsize, self.mpi.d2w_soffs*self.tsize)
		rinfo = (self.mpi.d2w_rcount*self.tsize, self.mpi.d2w_roffs*self.tsize)
		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (rbuf.reshape(-1),rinfo))
		wmap[self.mpi.d2w_rinds] = rbuf
		return wmap
	def wmap2dmap(self, wmap, dmap=None):
		"""Observational to distributed transfer and reduction on the CPU"""
		wmap= wmap.reshape(self.oshape) # to 4D in case 1D form is passed
		if dmap is None: dmap = self.dmap(dtype=wmap.dtype)
		tshape= dmap.shape[1:]
		rbuf  = np.zeros((self.mpi.d2w_stot,)+tshape, dmap.dtype)
		sbuf  = np.ascontiguousarray(wmap[self.mpi.d2w_rinds])
		rinfo = (self.mpi.d2w_scount*self.tsize, self.mpi.d2w_soffs*self.tsize)
		sinfo = (self.mpi.d2w_rcount*self.tsize, self.mpi.d2w_roffs*self.tsize)
		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (rbuf.reshape(-1),rinfo))
		# At this point we will have multiple contributions to each of our
		# distal tiles, which we need to reduce. Loop in python for now.
		# This performs 3*64*64 = 12k operations per iteration, so overhead
		# might not be too bad. But could try numba otherwise.
		dmap[:] = 0
		for i, gi in enumerate(self.mpi.d2w_sinds):
			dmap[gi] += rbuf[i]
		return dmap
	def dmap2omap(self, dmap, root=0):
		"""Turn the distributed tile map dmap into a full enmap
		on the mpi rank root (defaults to 0). If the pixbox
		argument was passed to the constructor, then this subset
		of the full map will be output."""
		sbuf = np.ascontiguousarray(dmap[self.ompi.sinds].reshape(-1))
		rbuf = np.zeros(self.ompi.rtot*self.tsize, dmap.dtype) if self.ompi.comm.rank == 0 else None
		rinfo= (self.ompi.rcount*self.tsize, self.ompi.roffs*self.tsize)
		self.ompi.comm.Gatherv(sbuf, (rbuf, rinfo))
		if self.ompi.comm.rank == 0:
			rbuf = rbuf.reshape(-1,self.ncomp,self.tshape[0],self.tshape[1])
			omap = np.zeros((self.ompi.onty,self.ompi.ontx,self.ncomp,self.tshape[0],self.tshape[1]),dmap.dtype)
			omap[self.ompi.rinds[0],self.ompi.rinds[1]] = rbuf
			omap = np.moveaxis(omap, (2,3,4), (0,2,4))
			omap = omap.reshape(omap.shape[0],omap.shape[1]*omap.shape[2],omap.shape[3]*omap.shape[4])
			print(self.ompi.obox)
			# self.mpi.obox will not have negative values or wrapping issues the way we have
			# constructed things here
			print(self.ompi.obox)
			(y1,x1),(y2,x2) = self.ompi.obox
			omap = omap[...,y1:y2,x1:x2]
			omap = enmap.ndmap(omap, self.pwcs)
			print(omap.box()/utils.degree)
			return omap
	@property
	def oshape(self): return (self.work .ntile,self.ncomp,self.tshape[0],self.tshape[1])
	@property
	def dshape(self): return (self.dist.ntile,self.ncomp,self.tshape[0],self.tshape[1])
	def dmap(self, dtype=np.float32, ncomp=3): return np.zeros(self.dshape, dtype)
	def wmap(self, dtype=np.float32, ncomp=3): return np.zeros(self.oshape, dtype)
	def gwmap(self, buf=None, dtype=np.float32):
		if buf: arr = buf .zeros(self.work.size, dtype)
		else:   arr = cupy.zeros(self.work.size, dtype)
		return gpu_mm.LocalMap(self.work.lp, arr)

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

def build_mpi_info(dist_inds, work_inds, comm):
	"""Set up information needed for transfer between dist and work tilings.
	Everything is in tile units. Multiply by tsize to get indices into flattened
	arrays."""
	# Build owners
	mpi   = bunch.Bunch(comm=comm)
	dmine = dist_inds >= 0
	wmine = work_inds  >= 0
	owners= distoff_owners(dmine, comm)

	# We order tiles in buffers in send order
	# We assume that there aren't any missing tiles in dist

	# Part 1: What to sent to each, for a dist→work operation
	# My dist inds for all tiles everybody wants
	my_oty,  my_otx  = np.ascontiguousarray(np.nonzero(wmine)) # [{ty,tx},nobs]
	all_oty          = utils.allgatherv(my_oty, comm)
	all_otx          = utils.allgatherv(my_otx, comm)
	all_on           = utils.allgather(np.sum(wmine), comm)
	all_ranks        = np.repeat(np.arange(comm.size), all_on)
	mpi.d2w_sinds    = dist_inds[all_oty,all_otx]
	# Which of these are actually valid
	svalid           = mpi.d2w_sinds >= 0
	mpi.d2w_sinds    = mpi.d2w_sinds[svalid]
	# Corresponding send counts
	mpi.d2w_scount   = np.bincount(all_ranks, svalid, minlength=comm.size).astype(int)
	mpi.d2w_stot     = np.sum(mpi.d2w_scount)
	# The offset in the send array for each
	mpi.d2w_soffs    = utils.cumsum(mpi.d2w_scount)

	# Part 2: What to receive from each, for a dist→work operation
	# Which of my work tiles everybody has
	my_owners = owners[my_oty,my_otx]
	rvalid    = my_owners >= 0
	# How much to receive from each
	mpi.d2w_rcount = np.bincount(my_owners[rvalid], minlength=comm.size).astype(int)
	mpi.d2w_roffs  = utils.cumsum(mpi.d2w_rcount)
	# Mapping from our work inds to the buffer
	recv_y, recv_x = owners2rtiles(owners, my_oty, my_otx)
	mpi.d2w_rinds  = work_inds[recv_y,recv_x]

	return mpi

def build_omap_mpi(shape, wcs, tshape, dist_inds, comm, pixbox=None):
	"""Build the information needed to go from the distributed
	maps to a single enmap on the root node"""
	if pixbox is None: pixbox = [[0,0],list(shape[-2:])]
	# 1. Split pixbox into tile fragments consisting of
	#    [tyx,subpixbox]
	tinfo    = tile_pixbox(pixbox, tshape)
	# 2. Wrap tyx to be within nty,ntx
	tinds    = tinfo.tinds % dist_inds.shape
	# Flatten tinds to make it a bit easier to work with
	oty, otx = tinds.reshape(-1,2).T
	# 3. Find which parts we own, building the send info
	out_inds = dist_inds[oty,otx].reshape(tinfo.nty,tinfo.ntx)
	mine     = out_inds >= 0
	sinds    = out_inds[mine]
	scount   = np.sum(sinds)
	# 4. Build the receive info. This is only relevant for the
	#    root. Find out which cells everybody will send to root.
	owners   = distoff_owners(dist_inds>=0, comm)
	rinds    = np.array(owners2rtiles(owners, oty, otx))
	# Just because a tile is owned doesn't mean we want it.
	oowners  = owners[oty,otx]
	rcount   = np.bincount(oowners[oowners>=0], minlength=comm.size).astype(int)
	roffs    = utils.cumsum(rcount)
	# Make rinds relative to output tiles
	rinds   -= (tinfo.t1 % dist_inds.shape)[:,None]
	rtot     = np.sum(rcount)
	# Infer the output geometry we will get after slicing
	return bunch.Bunch(onty=tinfo.nty, ontx=tinfo.ntx, obox=tinfo.obox, sinds=sinds, scount=scount, rinds=rinds, rcount=rcount, roffs=roffs, rtot=rtot, comm=comm)

def tile_pixbox(pixbox, tshape):
	pixbox   = np.asarray(pixbox)
	t1       = pixbox[0]    //tshape
	t2       = (pixbox[1]-1)//tshape+1
	nty,ntx  = t2-t1
	obox     = pixbox - t1*tshape
	# Build full list of indices
	tinds        = np.zeros((nty,ntx,2),int)
	tinds[:,:,0] = np.arange(t1[0],t2[0])[:,None]
	tinds[:,:,1] = np.arange(t1[1],t2[1])[None,:]
	return bunch.Bunch(t1=t1, t2=t2, nty=nty, ntx=ntx, obox=obox, tinds=tinds)

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
	off    = utils.nint(owcs.wcs.crpix-wcs.wcs.crpix)[::-1]
	pixbox = np.array([off,off+shape[-2:]])
	return (ny,nx), owcs, pixbox

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

def owners2rtiles(owners, tys, txs):
	# Mapping from our work inds to the buffer
	# Need to know which tiles everybody will send to us
	#  [y,x for rank in size for y,x in zip(my_oty,my_otx) if owners[y,x]==rank]
	# How do I write this efficiently? Can collapse yx
	#  [yx for rank in size for yx in yxs if owners[yx]==rank]
	#  for yx in yxs: subs[owners[yx]].append(yx)
	# Could be done with an alltoallv, but that's overkill
	n = max(np.max(owners)+1,1)
	yxlist = [[] for i in range(n)]
	for ty,tx in zip(tys,txs):
		if owners[ty,tx] >= 0:
			yxlist[owners[ty,tx]].append((ty,tx))
	rty, rtx = np.concatenate(yxlist, 0).T
	return rty, rtx

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
