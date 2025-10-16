import numpy as np
from pixell import wcsutils, enmap, utils, bunch
from scipy import spatial, optimize
from . import device

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

# FIXME: This currently hardcodes ncomp, which breaks output
# for hits and div, and in general makes these hacky to deal with.
# In theory this would just involve replacing self.tsize with the actual
# tsize, but a complication is that local_pixelization and LocalMap
# assume ncomp = 3, so some functions will still not be general.
# Should make it explicit which functions work

class TileDistribution:
	def __init__(self, shape, wcs, local_pixelization, comm, pixbox=None, dev=None):
		# hardcoded constants
		self.tshape = (64,64)
		self.ncomp  = 3
		self.tsize  = self.ncomp*np.prod(self.tshape)
		self.dev    = dev or device.get_device()
		# Shortcut, since we'll be referring to it a lot
		lp = local_pixelization
		# Our geometry
		shape = shape[-2:]
		self.shape, self.wcs = shape, wcs
		# Sub-rectangle we're actually interested in.
		# Used when reducing the map.
		if pixbox is not None and utils.streq(pixbox,"auto"):
			pixbox = get_pixbounds(lp, comm)
		self.pixbox = pixbox
		if pixbox is not None:
			_, self.pwcs = enmap.crop_geometry(shape, wcs, pixbox=pixbox, recenter=True)
		else: self.pwcs = wcs
		# Set up the work tiling. Would be nice if the
		# latter could be part of
		self.work  = bunch.Bunch(lp=lp)
		self.work.cell_inds = lp.cell_offsets_cpu//self.tsize
		self.work.ntile     = np.sum(self.work.cell_inds>=0)
		self.work.shape     = (self.work.ntile,self.ncomp)+self.tshape
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
		self.dev.copy(wmap, gwmap.arr)
		return gwmap
	def gwmap2dmap(self, gwmap, dmap=None):
		return self.wmap2dmap(self.dev.get(gwmap.arr), dmap=dmap)
	def dmap2wmap(self, dmap, wmap=None):
		"""Distributed to observational tile transfer on the CPU.
		Does not assume pre = (3,)."""
		# dmap and wmap should be numpy arrays with shape [ntile,...,tshape[0],tshape[1]]
		oshape = (self.work.ntile,)+dmap.shape[1:-2]+(self.tshape[0],self.tshape[1])
		tsize  = np.prod(oshape[1:])
		if wmap is None: wmap = np.zeros(oshape, dmap.dtype)
		wmap  = wmap.reshape(oshape) # to 4d in case 1d form is passed
		sbuf  = dmap[self.mpi.d2w_sinds]
		rbuf  = np.zeros(wmap.shape, wmap.dtype)
		sinfo = (self.mpi.d2w_scount*tsize, self.mpi.d2w_soffs*tsize)
		rinfo = (self.mpi.d2w_rcount*tsize, self.mpi.d2w_roffs*tsize)
		self.mpi.comm.Alltoallv((sbuf.reshape(-1),sinfo), (rbuf.reshape(-1),rinfo))
		wmap[self.mpi.d2w_rinds] = rbuf
		return wmap
	def wmap2dmap(self, wmap, dmap=None):
		"""Observational to distributed transfer and reduction on the CPU.
		Assumes pre = (3,)."""
		wmap= wmap.reshape(self.oshape) # to 4D in case 1D form is passed
		if dmap is None: dmap = self.dmap(dtype=wmap.dtype)
		tshape= dmap.shape[1:]
		rbuf  = np.zeros((self.mpi.d2w_stot,)+tshape, dmap.dtype)
		# Problem: d2w_rinds is somehow zero
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
		of the full map will be output.
		Does not assume pre = (3,)."""
		# Set up our output map, which we will copy stuff into
		pre   = dmap.shape[1:-2]
		tsize = np.prod(dmap.shape[1:])
		if self.ompi.comm.rank == 0:
			omap = enmap.zeros(pre+self.ompi.oshape, self.pwcs, dmap.dtype)
		# Now loop over individual output slabs, receive each, and insert into
		# the output
		for i, sub in enumerate(self.ompi.subs):
			sbuf = np.ascontiguousarray(dmap[sub.sinds].reshape(-1))
			rbuf = np.zeros(sub.rtot*tsize, dmap.dtype) if sub.comm.rank == 0 else None
			rinfo= (sub.rcount*tsize, sub.roffs*tsize)
			sub.comm.Gatherv(sbuf, (rbuf, rinfo), root=root)
			if self.ompi.comm.rank == 0:
				rbuf   = rbuf.reshape((-1,)+dmap.shape[1:])
				submap = np.zeros((sub.onty,sub.ontx)+dmap.shape[1:],dmap.dtype)
				submap[sub.rinds[0],sub.rinds[1]] = rbuf
				# We have [nty,ntx,...,y,x] and want [...,nty,y,ntx,x]
				submap = np.moveaxis(submap, (0,1), (-4,-2))
				submap = submap.reshape(submap.shape[:-4]+(submap.shape[-4]*submap.shape[-3],submap.shape[-2]*submap.shape[-1]))
				# sub.obox will not have negative values or wrapping issues the way we have
				# constructed things here
				(iy1,ix1),(iy2,ix2) = sub.ibox
				(oy1,ox1),(oy2,ox2) = sub.obox
				# Copy over to output map
				omap[...,oy1:oy2,ox1:ox2] += submap[...,iy1:iy2,ix1:ix2]
		if self.ompi.comm.rank == 0:
			return omap
	@property
	def oshape(self): return (self.work.ntile,self.ncomp,self.tshape[0],self.tshape[1])
	@property
	def dshape(self): return (self.dist.ntile,self.ncomp,self.tshape[0],self.tshape[1])
	def dmap(self, dtype=np.float32, ncomp=3): return np.zeros(self.dshape, dtype)
	def wmap(self, dtype=np.float32, ncomp=3): return np.zeros(self.oshape, dtype)
	def gwmap(self, buf=None, dtype=np.float32):
		if buf: arr = buf.zeros(self.work.shape, dtype)
		else:   arr = self.dev.np.zeros(self.work.shape, dtype)
		return self.dev.lib.LocalMap(self.work.lp, arr)

def build_dist_tiling(dist_owner, comm, tsize=1):
	dist = bunch.Bunch(owner=dist_owner)
	dist.mine  = dist_owner == comm.rank
	dist.ntile = np.sum(dist.mine)
	dist.size  = dist.ntile*tsize
	dist.cell_inds = np.full(dist_owner.shape, -1, dist_owner.dtype)
	dist.cell_inds[dist.mine] = np.arange(dist.ntile)
	return dist

def distoff_owners(dmine, comm):
	"""dmine: boolean mask of tiles owned by us. Returns who owns each cell (-1 for unowned)"""
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
	wmine = work_inds >= 0
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

# What to do when pixbox straddles the x wrap? Tiles no lonager align then.
# Might be simplest to split into 2 sub-pixboxes in that case, each with
# its own info. Maybe simplest to just call build_omap_mpi multiple times


def build_omap_mpi(shape, wcs, tshape, dist_inds, comm, pixbox=None):
	if pixbox is None: pixbox = [[0,0],list(shape[-2:])]
	# * sboxes = split but not wrapped pixboxes. These are directly
	#   comparable to the input pixbox
	# * wboxes = also wrapped. These are directly comparable to the full
	#   pixelization, and will lie inside it
	sboxes, wboxes = wrap_pixbox(pixbox, shape)
	ompi = bunch.Bunch(oshape=tuple(pixbox[1]-pixbox[0]), subs=[], comm=comm)
	for sbox, wbox in zip(sboxes, wboxes):
		sub = _build_omap_mpi_single(shape, wcs, tshape, dist_inds, comm, pixbox=wbox)
		sub.obox = sbox-pixbox[0]
		ompi.subs.append(sub)
	return ompi

def wrap_pixbox(pixbox, shape):
	# Find how many wraps we have in each direction
	sboxes, wboxes = [], []
	t1 = pixbox[0]//shape
	t2 = (pixbox[1]-1)//shape+1
	for ty in range(t1[0],t2[0]):
		y1 = max(ty*shape[0],pixbox[0,0])
		y2 = min((ty+1)*shape[0],pixbox[1,0])
		for tx in range(t1[1],t2[1]):
			x1 = max(tx*shape[1],pixbox[0,1])
			x2 = min((tx+1)*shape[1],pixbox[1,1])
			sbox = np.array([[y1,x1],[y2,x2]])
			wbox = sbox - [ty*shape[0],tx*shape[1]]
			sboxes.append(sbox)
			wboxes.append(wbox)
	return sboxes, wboxes

def _build_omap_mpi_single(shape, wcs, tshape, dist_inds, comm, pixbox=None):
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
	scount   = len(sinds)
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
	return bunch.Bunch(onty=tinfo.nty, ontx=tinfo.ntx, ibox=tinfo.ibox, sinds=sinds, scount=scount, rinds=rinds, rcount=rcount, roffs=roffs, rtot=rtot, comm=comm)

def tile_pixbox(pixbox, tshape):
	pixbox   = np.asarray(pixbox)
	t1       = pixbox[0]    //tshape
	t2       = (pixbox[1]-1)//tshape+1
	nty,ntx  = t2-t1
	ibox     = pixbox - t1*tshape
	# Build full list of indices
	tinds        = np.zeros((nty,ntx,2),int)
	tinds[:,:,0] = np.arange(t1[0],t2[0])[:,None]
	tinds[:,:,1] = np.arange(t1[1],t2[1])[None,:]
	return bunch.Bunch(t1=t1, t2=t2, nty=nty, ntx=ntx, ibox=ibox, tinds=tinds)

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

def distribute_tods_semibrute(obsinfo, nsplit, npass=2, niter=10, verbose=False):
	"""Split into nsplit groups such that the groups are
	are approximately maximally-separated in state-space,
	and each group has approximately the same weight.
	This function is pretty inefficient, but it does its job. It's probably still
	fast enough. It scales as nobs*nsplit, and takes 280 ms for 21k tods split over 100
	mpi tasks. We can improve it later if necessary. It does a reasonably good job at
	optimizing both the sample volume and compactness per mpi task."""
	ntod  = len(obsinfo)
	# Handle special cases
	if nsplit == 1: return bunch.Bunch(owner=np.zeros(ntod,int), weight=np.ones(ntod))
	if ntod <= nsplit: return bunch.Bunch(owner=np.arange(ntod), weight=np.ones(ntod))
	# Then the main case
	# This used to use pixels, but the geometry dependency was annoying, and I don't
	# see why raw coordinates wouldn't work
	#pix   = enmap.sky2pix(shape, wcs, [obsinfo.sweep[:,:,1],obsinfo.sweep[:,:,0]]) # [{y,x},ntod,npoint]
	pos   = np.array([obsinfo.sweep[:,:,1],obsinfo.sweep[:,:,0]]) # [{y,x},ntod,npoint]
	state = np.moveaxis(pos,1,0).reshape(ntod,-1) # [ntod,ncoord]
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
	"""owners: 2d grid of who owns each cell (-1 for unowned)
	tys: y tile index of the cells we want
	txs: x tile index of the cells we want
	Returns
	  rty: concatenated list of tile y indices to receive from each task
	  rtx: concatenated list of tile x indices to receive from each task
	"""
	# Mapping from our work inds to the buffer
	# Need to know which tiles everybody will send to us
	#  [y,x for rank in size for y,x in zip(my_oty,my_otx) if owners[y,x]==rank]
	# How do I write this efficiently? Can collapse yx
	#  [yx for rank in size for yx in yxs if owners[yx]==rank]
	#  for yx in yxs: subs[owners[yx]].append(yx)
	# Could be done with an alltoallv, but that's overkill
	# 1. Infer how many mpi tasks we have. Should really get it from comm,
	#    since some tasks could in theory not own anything. In practice,
	#    those tasks won't contribute to the send offset either, so it should work
	n = max(np.max(owners)+1,1)
	# 2. Build a list of which tiles to receive from each
	yxlist = [[] for i in range(n)]
	for ty,tx in zip(tys,txs):
		if owners[ty,tx] >= 0:
			yxlist[owners[ty,tx]].append((ty,tx))
	# We could end up with some tasks owning nothing.
	# This isn't ideal in general, but we should at least
	# make sure it doesn't crash by turning them into correct-
	# shaped 0-size arrays
	yxlist = [np.array(a, dtype=np.int32).reshape(-1,2) for a in yxlist]
	# 3. Flatten
	rty, rtx = np.concatenate(yxlist, 0).reshape(-1,2).astype(np.int32).T
	return rty, rtx

def get_pixbounds(lp, comm):
	"""Return a pixbox bounding the active cells in a LocalPixelization"""
	hit  = lp.cell_offsets_cpu >= 0
	hit  = utils.allreduce(hit.astype(int), comm) > 0
	# The y bounds are simple, since there's no wrapping there
	ycells = np.where(np.sum(hit,1)>0)[0]
	# Nothing hit?!
	if len(ycells) == 0: return np.array([[0,0],[1,1]])
	y1 = ycells[0]*64
	y2 = np.minimum((ycells[-1]+1)*64, lp.nypix_global)
	# The x bounds are harder. We don't want a huge stripe across the
	# sky just because our exposed area straddles the wrapping point.
	# Start by getting the x-range each x-cell covers
	xcells  = np.where(np.sum(hit,0)>0)[0]
	if len(xcells) == 0: return np.array([[0,0],[1,1]])
	xranges = np.array([xcells*64,(xcells+1)*64]).T
	xranges = np.minimum(xranges, lp.nxpix_global)
	if len(xranges) > 1:
		# Find the biggest jump between cell starts
		gaps    = xranges[1:,0]-xranges[:-1,1]
		ijump   = np.argmax(gaps)
		# If this jump is bigger than half the sky, then
		# it will be more efficient to rewrap
		if gaps[ijump] >= lp.nxpix_global//2:
			xranges[:ijump+1] += lp.nxpix_global
	# Ok, it should be safe now
	x1 = np.min(xranges[:,0])
	x2 = np.max(xranges[:,1])
	return np.array([[y1,x1],[y2,x2]])

def enmap2lmap(map, tshape=(64,64), ncomp=3, dev=None):
	"""Given an enmap, return a LocalMap that can be used in
	map2tod operations. All tiles covering the emap will be
	active, so there's no distribution happening here. This
	is useful for cases like projecting a per-obs mask to
	time domain in order to construct cuts."""
	if dev is None: dev = device.get_device()
	fshape, fwcs, pixbox = infer_fullsky_geometry(map.shape, map.wcs)
	tinfo = tile_pixbox(pixbox, tshape)
	# Allocate our tiles
	ncell = tinfo.nty*tinfo.ntx
	cell_inds = np.arange(ncell).reshape(tinfo.nty,tinfo.ntx).astype(np.int32)
	cell_offs = cell_inds*(ncomp*tshape[0]*tshape[1])
	# Make a buffer that covers our map and which we can reshape into
	# our local-map cells later
	buf = np.zeros((ncomp,tinfo.nty,tshape[0],tinfo.ntx,tshape[1]),map.dtype)
	buf[:,tinfo.ibox[0,0]:tinfo.ibox[1,0],tinfo.ibox[0,1]:tinfo.ibox[1,1]] = map
	# Reshape and then move it to the device
	buf = buf.reshape(ncell,ncomp,tshape[0],tshape[1])
	buf = dev.np.array(buf, order="C")
	# Finally build a LocalMap from it
	lp  = dev.lib.LocalPixelization(fshape[-2], fshape[-1], cell_offsets=cell_offs)
	lmap= dev.lib.LocalMap(lp, buf)
	return lmap

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
