import numpy as np
from pixell import utils, bunch
from . import gutils, device
from .logging import L

class Nmat:
	def __init__(self, dev=None):
		"""Initialize the noise model. In subclasses this will typically set up parameters, but not
		build the details that depend on the actual time-ordered data"""
		self.dev   = dev or device.get_device()
		self.ivar  = dev.np.ones(1, dtype=np.float32)
		self.ready = True
	def build(self, tod, **kwargs):
		"""Measure the noise properties of the given time-ordered data tod[ndet,nsamp], and
		return a noise model object tailored for that specific tod. The returned object
		needs to provide the .apply(tod) method, which multiplies the tod by the inverse noise
		covariance matrix. Usually the returned object will be of the same class as the one
		we call .build(tod) on, just with more of the internal state initialized."""
		return self
	def apply(self, tod):
		"""Multiply the time-ordered data tod[ndet,nsamp] by the inverse noise covariance matrix.
		This is done in-pace, but the result is also returned."""
		return tod.copy()
	def white(self, tod):
		"""Like apply, but without detector or time correlations"""
		return tod.copy()
	def write(self, fname):
		bunch.write(fname, bunch.Bunch(type="Nmat"))
	@staticmethod
	def from_bunch(data): return Nmat()
	def check_ready(self):
		if not self.ready:
			raise ValueError("Attempt to use partially constructed %s. Typically one gets a fully constructed one from the return value of nmat.build(tod)" % type(self).__name__)

# TODO: Update these notes to reflect what's actually done

# Our model will be:
#  ps = ps_detcov * Δperdet * Δmean * Δtweak
# * ps_detcov is a detvecs noise model. Low frequency resolution
# * Δperdet is a medium-resolution per-detector correction factor
# * Δmean and Δtweak are high-resolution overall correction factors.
#   They can probably be merged. Δmean ensures mean(ps) gets the
#   right value at high resolution. Good for capturing thin spikes.
#   Δtweak modifies this to give ps(mean) the right value instead.
#   It downweights scales where there's correlations in the real
#   data that's not captured by ps_detcov, so these scales get the
#   correct overall weight, even if phases aren't captured the way
#   they would have been in ps_detcov.
#
# The noise can have large spikes, especially due to the strong scan-synchronous
# sgnal. The uncorrelated part of this will be captures by ps_mean and ps_perdet,
# but we don't have the resolution to capture this in the detector-correlated model.
# We will therefore do spike detection, and do separate mode finding in the spike
# regions and atmospheric regions. Will also measure eigenvalues separately per
# spike. If necessary can measure eigenvalues across multiple spikes at once if
# there's not enough statistics.
#
# The overall measurement procedure will be:
# 1. Measure mean(ps)
# 2. Detect spikes in mean(ps). This only needs to capture the most important spikes.
#    Use these to define spike bins. One find-bin across all
# 3. Build atmospheric bins and find-bins
# 4. Find eigenmodes in the find-bins
# 5. Measure eigvals and white in each bin
# 7. Predict ps_perdet. Define Δperdet as ps_perdet_true/ps_perdet_pred
# 8. Predict mean(ps) using 6 and 7. Define Δmean as ps_mean_true/ps_mean_pred
# 9. Predict ps(mean) using 6,7,8. Compare to true ps(mean). Define Δtweak
#
# Given the order we've measured things in, our model is actually
#  N = Δtweak * Δmean * Δperdet * N_detcov
# where N_detcov does not commute with Δperdet. Is this symmetric?
# It can't be...


# How to fix this? Could divide out the simple parts first (mean, perdet),
# but this would leave all freqs equally important. I fear that would make
# the eigenvectors it finds suboptimal, since t wouldn't know to prioritize
# higher covariance, just higher correlation. But this can be fixed with
# some care in the find-bins.


# 1. Multiply by ips_mean * ips_tweak
# 2. Multiply by ips_perdet
# 3. Multiply by ips_detcorr


# Fourier normalization
# Define tod = F" ftod and Nf = <ftod ftod'>
# Then N = <tod tod'> = F" <ftod ftod'> F"' = F" Nf F"'
# and  N"= F' Nf" F
# So we don't want the actual normalized inverse in apply().
# But we can still choose our Fourier units to avoid overflow.
# Currently we have F = rfft, but could also choose F = rfft/nsamp
# to give us some breathing room.


class NmatAdaptive(Nmat):
	def __init__(self, eig_lim=16, single_lim=0.55, window=2, sampvar=1e-3,
			bsmooth=50, atm_res=2**(1/4), dev=None,
			hbmps=None, hbrel=None, bsize_mean=None, bsize_per=None,
			bins=None, iD=None, iE=None, V=None, Kh=None, ivar=None, nwin=None,
			normexp=-1):
		self.dev = dev or device.get_device()
		self.eig_lim    = eig_lim
		self.single_lim = single_lim
		self.window     = window
		self.sampvar    = sampvar
		self.bsmooth    = bsmooth
		self.atm_res    = atm_res
		self.normexp    = normexp
		# These are usually computed in build()
		self.bsize_mean = bsize_mean
		self.bsize_per  = bsize_per
		self.bins       = bins
		self.nwin       = nwin
		self.ready = all([a is not None for a in [bsize_mean,bsize_per,hbmps,hbrel,bins,iD,iE,V,Kh,ivar,nwin]])
		if self.ready:
			self.iD   = dev.np.ascontiguousarray(iD)
			self.ivar = dev.np.ascontiguousarray(ivar)
			self.hbmps= dev.np.ascontiguousarray(hbmps)
			self.hbrel= dev.np.ascontiguousarray(hbrel)
			self.V    = [dev.np.ascontiguousarray(v) for v in V]
			self.Kh   = [dev.np.ascontiguousarray(kh) for kh in Kh]
			self.iE   = iE
			self.maxbin = np.max(self.bins[:,1]-self.bins[:,0])
		self.dev.garbage_collect()
	def build(self, tod, srate, extra=False, **kwargs):
		# Improve on NmatDetvecs in two ways:
		# 1. Factorize out per-detector power
		# 2. Use the shape of the power spectrum to define
		#    both where to measure the eigenvectors, and
		#    the bins for the eigenvalues.
		nwin = utils.nint(self.window*srate)
		ndet, nsamp = tod.shape
		nfreq       = nsamp//2+1
		# Apply window in-place, to prefpare for fft
		gutils.apply_window(tod, nwin)
		ftod = self.dev.pools["fft"].empty((ndet,nfreq), utils.complex_dtype(tod.dtype))
		self.dev.lib.rfft(tod, ftod)
		# Normalize to avoid 32-bit overflow in atmospheric region.
		# Make sure apply and ivar are consistent
		ftod *= nsamp**self.normexp
		# Undo window
		gutils.apply_window(tod, nwin, -1)
		return self.build_fourier(ftod, srate, nsamp)
	def build_fourier(self, ftod, srate, nsamp):
		ndet, nfreq = ftod.shape
		dtype = utils.real_dtype(ftod.dtype)
		nwin  = utils.nint(self.window*srate)

		# [Step 1]: Build a detector-uncorrelated noise model
		# 1a. Measure power spectra
		ps  = self.dev.np.abs(ftod)**2  # full-res ps
		mps = self.dev.np.mean(ps,0)    # mean det power
		mtod= self.dev.np.mean(ftod,0)
		psm = self.dev.np.abs(mtod)**2  # power of common mode
		freqs = np.linspace(0, srate, nfreq)
		#np.savetxt("test_mps.txt", np.array([self.dev.get(freqs), self.dev.get(mps), self.dev.get(psm)]).T, fmt="%15.7e")
		# The relative sample variance is 2/nsamp, which we want to be better than self.sampvar.
		# This requires nsamp > 2/self.sampvar
		bsize_mean = max(1,utils.ceil(2/self.sampvar/ndet))
		bsize_per  = max(1,utils.ceil(2/self.sampvar))
		# Simpler if the bin sizes are multiples of each other
		bsize_per  = utils.ceil(bsize_per, bsize_mean)
		bmps       = gutils.downgrade(mps, bsize_mean)
		brel       = gutils.downgrade(ps, bsize_per) / gutils.downgrade(mps, bsize_per)
		print("B01", utils.minmax(self.dev.get(gutils.downgrade(ps, bsize_per))))
		print("B02", utils.minmax(self.dev.get(gutils.downgrade(mps, bsize_per))))
		print("B03", utils.minmax(self.dev.get(bmps)))
		print("B04", utils.minmax(self.dev.get(brel)))
		print("B05", self.dev.np.where(ps==0))
		# Precompute whitening versions of these
		hbmps      = bmps**-0.5
		hbrel      = brel**-0.5
		# 1b. Divide out these from the tod. After this ftod is whitened, except
		# for the detector correlations
		print("B1", self.dev.np.std(ftod))
		gutils.block_scale(ftod, hbmps, bsize=bsize_mean, inplace=True)
		print("B2", self.dev.np.std(ftod))
		gutils.block_scale(ftod, hbrel, bsize=bsize_per,  inplace=True)
		print("B3", self.dev.np.std(ftod))
		print("B4", utils.minmax(self.dev.get(hbmps)))
		print("B5", utils.minmax(self.dev.get(hbrel)))

		# [Step 2]: Build the frequency bins
		# 2a. Measure the smooth background behind the peaks. The bin size should
		# be wider then any spike we want to ignore. A good size would be the 1/scan_period,
		# but we don't have az here. Can't use too wide bins, or we won't find the
		# first spikes
		bsize_smooth = self.bsmooth
		smooth_bps   = gutils.downgrade(mps, bsize_smooth, op=self.dev.np.median)
		smooth_ps    = gutils.logint(smooth_bps, self.dev.np.arange(nfreq)/bsize_smooth)
		# 2b. Find our spikes
		rel_ps       = mps/smooth_ps
		bins_spike   = find_spikes(self.dev.get(rel_ps))
		# If we detect a spike at the very beginning, in the extrapolated region, then it's
		# unreliable
		if bins_spike[0,0] == 0: bins_spike = bins_spike[1:]
		# 2c. Find atmospheric regions using smooth_bps
		bins_atm     = find_atm_bins(smooth_bps, bsize=bsize_smooth, step=self.atm_res)
		# Add a final all-the-rest bin
		bins_atm     = np.concatenate([bins_atm,[[bins_atm[-1,1],nfreq]]],0)

		# Want to exclude the spikes from the atmospheric model.
		# Since we model the freq-bins as independent and zero-mean,
		# we can just zero out the spike regions after measuring
		# them, and then compensate for the loss of power aftewards.
		# This lets us avoid splitting the atm-bins.
		# At the end (before the model can be applied), the freq-bins
		# and atm-bins *will* need to be harmonized into a single set
		# of non-overlapping bins, though.

		# 3a. Find modes in spike bins
		spike_data = self.dev.np.concatenate([ftod[:,b1:b2] for b1,b2 in bins_spike],1)
		vecs_spike = find_modes_jon(spike_data, [[0,spike_data.shape[1]]],
			eig_lim=self.eig_lim, single_lim=self.single_lim, skip_mean=True, dtype=dtype)
		# 3b. Measure mode power per spike. Spikes are narrow, so we probably
		# don't have enough statistics to measure all the modes we found.
		# Have ndet*bsize knowns. Will measure nmode+ndet unknowns. Need
		# nmode < bsize*(ndet-1)
		spike_power = measure_detvecs_ortho(ftod, bins_spike, vecs_spike)
		# 3c. Find atm modes
		mask = self.dev.np.ones(ftod.shape[-1], np.int32)
		for bi, (b1,b2) in enumerate(bins_spike):
			mask[b1:b2] = 0
		vecs_atm = find_modes_jon(ftod, bins_atm, eig_lim=self.eig_lim,
			single_lim=self.single_lim, skip_mean=True, dtype=dtype)
		# 3d. Measure atm and residual power
		atm_power = measure_detvecs_ortho(ftod, bins_atm, vecs_atm, mask=mask)
		# 4. Interleave bins
		bins, srcs, sinds = override_bins([bins_atm, bins_spike])
		bins  = np.array(bins)
		power = bunch.Bunch()
		for key in atm_power:
			power[key] = pick_data([atm_power[key], spike_power[key]], srcs, sinds)
		# 5. Precompute Kh and iD
		nbin = len(bins)
		iD = 1/self.dev.np.array(power.Ds) # [nbin,ndet]
		iE = [1/E for E in power.Es]       # [nbin][nmode]
		# Precompute Kh = (E" + V'D"V)**-0.5. [nbin][nmode,nmode]
		Kh = []
		for bi in range(nbin):
			V = power.Vs[bi]
			iK = self.dev.np.diag(iE[bi]) + V.T.dot(iD[bi,:,None] * V)
			Kh.append(np.linalg.cholesky(self.dev.np.linalg.inv(iK)))
		# 6. Calculate the chisquare per mode. We will use this to
		#    adjust bmps, interpreting any excess chisquare as
		#    unmodelled uncorrelated noise.
		iNftod = ftod.copy()

		moomoo = ftod.copy()

		rft    = iNftod.view(dtype)
		maxnmode = max([V.shape[1] for V in power.Vs])
		maxbin   = np.max(bins[:,1]-bins[:,0])
		with self.dev.pools["pointing"].as_allocator():
			# Tmp must be big enough to hold a full bin's worth of data
			tmp    = self.dev.np.empty([maxnmode,2*maxbin],dtype=dtype)
			vtmp   = self.dev.np.empty([ndet,maxnmode], dtype=dtype)
			divtmp = self.dev.np.empty([ndet,maxnmode], dtype=dtype)
			apply_vecs2(rft, iD, power.Vs, Kh, bins, tmp, vtmp, divtmp, dev=self.dev, out=rft)
		with self.dev.pools["pointing"].as_allocator():
			iNftod = self.dev.np.conj(iNftod)
			ftod  *= iNftod

		moo  = self.dev.np.mean(ftod,0)
		if np.any(moo<0):
			from pixell import colors
			print(colors.lred + "writing test_negchisq.hdf" + colors.reset)
			bi = 1
			b1,b2 = bins[bi]
			print(moo[b1:b2])
			bunch.write("test_negchisq.hdf", bunch.Bunch(
				b1=b1, b2=b2, ftod=ftod[:,b1:b2], iD=iD[bi], V=power.Vs[bi], iE=iE[bi], Kh=Kh[bi],
				chisq=moo[b1:b2]))

		1/0

		chisq  = self.dev.np.mean(ftod.real,0)
		bchisq = gutils.downgrade(chisq, bsize_mean)
		# 7. Update hmbps using bchisq
		hbmps *= bchisq**-0.5
		# 7b. nsamp normalization of hbmps
		hbmps *= nsamp**self.normexp
		# 8. Also compute a representative white noise level
		bsize = self.dev.np.array(bins[:,1]-bins[:,0])
		ivar  = self.dev.np.mean(hbmps**2)
		ivar *= nsamp
		self.dev.garbage_collect()
		# 6. Construct the full noise model
		return NmatAdaptive(
			eig_lim=self.eig_lim, single_lim=self.single_lim, window=self.window,
			sampvar=self.sampvar, bsmooth=self.bsmooth, atm_res=self.atm_res, dev=self.dev,
			hbmps=hbmps, hbrel=hbrel, bsize_mean=bsize_mean, bsize_per=bsize_per,
			bins=bins, iD=iD, iE=iE, V=power.Vs, Kh=Kh, ivar=ivar, nwin=nwin, normexp=self.normexp)


		print("atm_power")
		print(atm_power)


		print("bins_atm")
		print(bins_atm)
		print("ps.shape", ps.shape)
		print("vecs_atm")
		print(vecs_atm.shape)

		bleh = mps*0
		for b1,b2 in bins_spike: bleh[b1:b2] = 1
		bleh2 = mps*0+1
		for b1,b2 in bins_atm:
			bleh2[b1] = bleh2[min(b2,len(bleh)-1)] = 0
		np.savetxt("test_relps.txt", np.array([freqs, rel_ps, smooth_ps, mps, bleh, bleh2]).T, fmt="%15.7e")
		1/0
	def apply(self, tod, inplace=True, nofft=False):
		self.check_ready()
		t1 = self.dev.time()
		if not inplace: tod = tod.copy()
		if not nofft: gutils.apply_window(tod, self.nwin)
		t2 = self.dev.time()
		if not nofft:
			ft = self.dev.pools["ft"].empty((tod.shape[0],tod.shape[1]//2+1),utils.complex_dtype(tod.dtype))
			self.dev.lib.rfft(tod, ft)
		else: ft = tod
		# If we don't cast to real here, we get the same result but much slower
		rft = ft.view(tod.dtype)
		t3  = self.dev.time()
		# Apply the high-resolution, non-detector-correlated part of the model.
		# The *2 compensates for the cast to real
		gutils.block_scale(rft, self.hbmps, bsize=self.bsize_mean*2, inplace=True)
		gutils.block_scale(rft, self.hbrel, bsize=self.bsize_per *2, inplace=True)
		t4  = self.dev.time()
		# Then handle the detector-correlation part.
		# First set up work arrays. Safe to overwrite tod array here,
		# since we'll overwrite it with the ifft afterwards anyway
		ndet     = len(tod)
		maxnmode = max([V.shape[1] for V in self.V])
		nbin     = len(self.bins)
		with self.dev.pools["tod"].as_allocator():
			# Tmp must be big enough to hold a full bin's worth of data
			tmp    = self.dev.np.empty([maxnmode,2*self.maxbin],dtype=rft.dtype)
			vtmp   = self.dev.np.empty([ndet,maxnmode],         dtype=rft.dtype)
			divtmp = self.dev.np.empty([ndet,maxnmode],         dtype=rft.dtype)
			apply_vecs2(rft, self.iD, self.V, self.Kh, self.bins, tmp, vtmp, divtmp, dev=self.dev, out=rft)
		self.dev.synchronize()
		t5 = self.dev.time()
		# Second half of high-resolution part
		gutils.block_scale(rft, self.hbmps, bsize=self.bsize_mean*2, inplace=True)
		gutils.block_scale(rft, self.hbrel, bsize=self.bsize_per *2, inplace=True)
		t6 = self.dev.time()
		# And finish
		if not nofft: self.dev.lib.irfft(ft, tod)
		t7 =self.dev.time()
		if not nofft: gutils.apply_window(tod, self.nwin)
		t8 =self.dev.time()
		L.print("iN sub win %6.4f fft %6.4f s1 %6.4f mats %6.4f s2 %6.4f ifft %6.4f win %6.4f ndet %3d nsamp %5d nmode %2d nbin %2d" % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t7-t6,t8-t7, tod.shape[0], tod.shape[1], maxnmode, len(self.bins)), level=3)
		return tod
	def white(self, gtod, inplace=True):
		self.check_ready()
		if not inplace: gtod.copy()
		gutils.apply_window(gtod, self.nwin)
		gtod *= self.ivar[:,None]
		gutils.apply_window(gtod, self.nwin)
		return gtod

def pick_data(datas, srcs, sinds):
	return [datas[src][sind] for src, sind in zip(srcs, sinds)]

#def override_bins(bins_base, bins_override):
#	ni = len(bins_base)
#	nj = len(bins_override)
#	i  = 0
#	j  = 0
#	obins, ofrom, oinds = [], [], []
#	b1, b2 = bins_base[0]
#	while True:
#		# Advance past any override bins that start before us.
#		# These will have already been dealt with
#		while j < nj and bins_override[j][0] <= b1: j += 1
#		# Handle everything that overlaps with us and
#		# starts at or after our start
#		while j < nj and bins_override[j][0] < b2:
#			# Part of us before override
#			if b1 < bins_override[j][0]:
#				obins.append((b1,bins_override[j][0]))
#				ofrom.append(0)
#				oinds.append(i)
#			# The override
#			obins.append(bins_override[j])
#			ofrom.append(1)
#			oinds.append(j)
#			b1 = bins_override[j][1]
#			j += 1
#		# Advance to next
#		if i >= ni-1: break
#		i += 1
#		b1 = max(b1, bins_base[i][0])
#		b2 = bins_base[i][1]
#	# Handle any left-over
#	if b1 < b2:
#		obins.append((b1,b2))
#		ofrom.append(0)
#		oinds.append(i)
#	return obins, ofrom, oinds

def override_bins(binss):
	# keep track of end of previous bin and progress
	# into each list. We will maintain the invariant that
	# the current bin in each list ends after prev_end
	# and starts on or before it.
	binss    = [bins.copy() for bins in binss]
	nlist    = len(binss)
	inds     = [0]*nlist
	active   = list(range(nlist))
	prev_end = 0
	obins, ofrom, oinds = [], [], []
	# Helper functions
	def get(li): return binss[li][inds[li]]
	def isort(startend):
		vals = [(get(li)[startend],li) for li in active]
		vals = sorted(vals, key=lambda a:a[0])
		return [a[1] for a in vals]
	while len(active) > 0:
		# 1. Find the one with the earliest start
		order  = isort(0)
		istart = order[0]
		vstart = get(istart)[0]
		vstart = max(vstart, prev_end)
		# 2. Find the first interruption. This is ether
		#    our bin's end, or the start of another bin
		vend   = get(istart)[1]
		interrupt = len(order) > 1 and get(order[1])[0] < vend
		if interrupt:
			# Other interrupts us
			vend = get(order[1])[0]
		# 3. Output bin if non-empty
		if vstart < vend:
			obins.append((vstart, vend))
			ofrom.append(istart)
			oinds.append(inds[istart])
		# 4. Update our state
		if interrupt:
			# We're not done, but update our starting point
			get(istart)[0] = min(get(order[1])[1],get(istart)[1])
		else:
			# We ended. Advance us
			inds[istart] += 1
			# Remove list from active if done with it
			if inds[istart] >= len(binss[istart]):
				active.remove(istart)
		prev_end = vend
	return obins, ofrom, oinds

def find_spikes(rel_ps, tol=10, pad=2):
	"""Find narrow spikes tol above 1 in rel_ps. Cpu"""
	mask       = np.zeros(len(rel_ps)+2, bool)
	mask[1:-1] = rel_ps > tol
	edges      = np.where(np.diff(mask))[0]
	bins       = np.array([edges[0::2],edges[1::2]]).T
	bins       = utils.pad_bins(bins, pad, min=0, max=len(rel_ps))
	bins       = np.asarray(utils.merge_bins(bins))
	return bins

def measure_detvecs_ortho(ftod, bins, vecs, mask=None, minper=2):
	# vecs: [ndet,nmode]
	ndet = len(ftod)
	ap   = device.anypy(ftod)
	Vs, Es, Ds = [], [], []
	for bi, (b1,b2) in enumerate(bins):
		ft   = ftod[:,b1:b2].copy()
		if mask is not None:
			ft   *= mask[b1:b2]
			nbad = (b2-b1)-int(np.sum(mask[b1:b2]))
		else: nbad = 0
		# Find how many modes we can afford to measure
		bsize= (b2-b1-nbad)*2 # complex
		nmax = utils.ceil((ndet*bsize-ndet)/minper)
		# Keep the first nmax vectors
		V    = vecs[:,:nmax] # [ndet,nmode]
		amps = V.T.dot(ft)   # [nmode,nf]
		E    = np.var(amps,1,ddof=nbad)
		order= np.argsort(E)[::-1]
		E, V, amps = E[order], V[:,order], amps[order]
		## Drop ones with too low power. But I don't have
		# a good criterion
		#keep = E>np.max(E)*1e-3
		#E, V, amps = E[keep], vecs[:,keep], amps[keep]
		# Project these out from the tod
		ft   = ft - V.dot(amps)
		D    = ap.var(ft,1,ddof=nbad)
		Vs.append(V)
		Es.append(E)
		Ds.append(D)
	return bunch.Bunch(Vs=Vs, Es=Es, Ds=Ds)

def find_atm_bins(ps, bsize=1, nmin=10, step=2, off=2):
	ifloor = len(ps)/off
	# translate to full-resolution indices
	ifloor *= bsize
	# Make log-spaced bins
	edges   = []
	while ifloor >= nmin:
		edges.append(utils.floor(ifloor))
		ifloor /= step
	edges.append(0)
	edges = edges[::-1]
	bins  = utils.edges2bins(edges)
	return bins

def find_floor(ps, tol=10, off=2):
	# Ignore any area before peak
	imax = max(1,np.argmax(ps))
	ps   = ps[imax:]
	# Floor is mean of values lower than tol times
	# min power
	vmin = np.min(ps)
	ifloor = utils.floor(find_first_below(ps, [vmin*tol])*off)[0]
	vfloor = np.mean(ps[ifloor:])
	# Add back imax offset
	ifloor += imax
	return ifloor, vfloor

def find_first_below(arr, vals):
	# Can't think of an efficient implementation here. Could
	# go to lower-level if necessary. The current implementation
	# makes only one pass through arr and vals, which would be
	# good if it weren't a python loop.
	vi   = 0
	inds = np.full(len(vals),-1)
	for ai, av in enumerate(arr):
		if vi >= len(vals): break
		if av < vals[vi]:
			inds[vi] = ai
			vi += 1
	return inds

class NmatDetvecs(Nmat):
	def __init__(self, bin_edges=None, eig_lim=16, single_lim=0.55, mode_bins=[0.25,4.0,20],
			downweight=[], window=2, nwin=None, verbose=False, bins=None, iD=None, V=None,
			Kh=None, iE=None, ivar=None, dev=None):
		# Variables used for building the noise model
		if bin_edges is None: bin_edges = np.array([
			0.16, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
			1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 3.80,
			4.60, 5.00, 5.50, 6.00, 6.50, 7.00, 8.00, 9.00, 10.0, 11.0,
			12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 22.0,
			24.0, 26.0, 28.0, 30.0, 32.0, 36.5, 41.0,
			45.0, 50.0, 55.0, 65.0, 70.0, 80.0, 90.0,
			100., 110., 120., 130., 140., 150., 160., 170.,
			180., 190.
		])
		self.dev       = dev or device.get_device()
		self.bin_edges = np.array(bin_edges)
		self.mode_bins = np.array(mode_bins)
		self.eig_lim   = eig_lim
		self.single_lim= single_lim
		self.verbose   = verbose
		self.downweight= downweight
		# Variables used for applying the noise model
		self.bins      = bins
		self.window    = window
		self.nwin      = nwin
		self.iD, self.V, self.Kh, self.ivar = iD, V, Kh, ivar
		self.iE        = iE # not necessary
		self.ready      = all([a is not None for a in [iD, V, Kh, ivar]])
		if self.ready:
			self.iD, self.V, self.Kh, self.ivar = [dev.np.asarray(a) for a in [iD, V, Kh, ivar]]
			self.maxbin = np.max(self.bins[:,1]-self.bins[:,0])
	def build(self, tod, srate, extra=False, **kwargs):
		# Apply window before measuring noise model
		dtype = tod.dtype
		nwin  = utils.nint(self.window*srate)
		ndet, nsamp = tod.shape
		nfreq = nsamp//2+1
		tod   = self.dev.np.asarray(tod)
		gutils.apply_window(tod, nwin)
		#bunch.write("test_sogma_full.hdf", bunch.Bunch(tod=tod, srate=srate))
		ft    = self.dev.lib.rfft(tod)
		# Unapply window again
		gutils.apply_window(tod, nwin, -1)
		#del tod
		# First build our set of eigenvectors in two bins. The first goes from
		# 0.25 to 4 Hz the second from 4Hz and up
		mode_bins = makebins(self.mode_bins, srate, nfreq, 1000, rfun=np.round)[1:]
		if np.any(np.diff(mode_bins) < 0):
			raise RuntimeError(f"At least one of the frequency bins has a negative range: \n{mode_bins}")
		# Then use these to get our set of basis vectors
		V = find_modes_jon(ft, mode_bins, eig_lim=self.eig_lim, single_lim=self.single_lim, verbose=self.verbose, dtype=dtype)
		nmode= V.shape[1]
		if V.size == 0: raise errors.ModelError("Could not find any noise modes")
		# Cut bins that extend beyond our max frequency
		bin_edges = self.bin_edges[self.bin_edges < srate/2 * 0.99]
		bins      = makebins(bin_edges, srate, nfreq, nmin=2*nmode, rfun=np.round)
		nbin      = len(bins)
		# Now measure the power of each basis vector in each bin. The residual
		# noise will be modeled as uncorrelated
		E  = self.dev.np.zeros([nbin,nmode],dtype)
		D  = self.dev.np.zeros([nbin,ndet],dtype)
		Nd = self.dev.np.zeros([nbin,ndet],dtype)
		for bi, b in enumerate(bins):
			# Skip the DC mode, since it's it's unmeasurable and filtered away
			b = np.maximum(1,b)
			E[bi], D[bi], Nd[bi] = measure_detvecs(ft[:,b[0]:b[1]], V)
		del Nd, ft
		# Optionally downweight the lowest frequency bins
		if self.downweight != None and len(self.downweight) > 0:
			D[:len(self.downweight)] /= self.dev.np.array(self.downweight)[:,None]
		# Also compute a representative white noise level
		bsize = self.dev.np.array(bins[:,1]-bins[:,0])
		ivar  = self.dev.np.sum(1/D*bsize[:,None],0)/self.dev.np.sum(bsize)
		ivar *= nsamp
		# We need D", not D
		iD, iE = 1/D, 1/E
		# Precompute Kh = (E" + V'D"V)**-0.5
		Kh = self.dev.np.zeros([nbin,nmode,nmode],dtype)
		for bi in range(nbin):
			iK = self.dev.np.diag(iE[bi]) + V.T.dot(iD[bi,:,None] * V)
			Kh[bi] = np.linalg.cholesky(self.dev.np.linalg.inv(iK))
		# Construct a fully initialized noise matrix
		nmat = NmatDetvecs(bin_edges=self.bin_edges, eig_lim=self.eig_lim, single_lim=self.single_lim,
				window=self.window, nwin=nwin, downweight=self.downweight, verbose=self.verbose,
				bins=bins, iD=iD, V=V, Kh=Kh, iE=iE, ivar=ivar, dev=self.dev)

		#model = np.zeros((tod.shape[0], tod.shape[1]//2+1), tod.dtype)
		#for bi, b in enumerate(nmat.bins):
		#	model[:,b[0]:b[1]] = D[bi,:,None] + np.sum(V**2*E[bi,None,:],1)[:,None]
		#bunch.write("test_sogma2.hdf", bunch.Bunch(ps=model[:,:model.shape[1]//100]))



		return nmat
	def apply(self, gtod, inplace=True):
		self.check_ready()
		t1 =self.dev.time()
		if not inplace: gtod = gtod.copy()
		gutils.apply_window(gtod, self.nwin)
		t2 =self.dev.time()
		ft = self.dev.pools["ft"].empty((gtod.shape[0],gtod.shape[1]//2+1),utils.complex_dtype(gtod.dtype))
		self.dev.lib.rfft(gtod, ft)
		# If we don't cast to real here, we get the same result but much slower
		rft = ft.view(gtod.dtype)
		t3 =self.dev.time()
		# Work arrays. Safe to overwrite tod array here, since we'll overwrite it with the ifft afterwards anyway
		ndet, nmode = self.V.shape
		nbin        = len(self.bins)
		with self.dev.pools["tod"].as_allocator():
			# Tmp must be big enough to hold a full bin's worth of data
			tmp    = self.dev.np.empty([nmode,2*self.maxbin],dtype=rft.dtype)
			vtmp   = self.dev.np.empty([ndet,nmode],         dtype=rft.dtype)
			divtmp = self.dev.np.empty([ndet,nmode],         dtype=rft.dtype)
			apply_vecs(rft, self.iD, self.V, self.Kh, self.bins, tmp, vtmp, divtmp, dev=self.dev, out=rft)
		self.dev.synchronize()
		t4 =self.dev.time()
		self.dev.lib.irfft(ft, gtod)
		t5 =self.dev.time()
		gutils.apply_window(gtod, self.nwin)
		t6 =self.dev.time()
		L.print("iN sub win %6.4f fft %6.4f mats %6.4f ifft %6.4f win %6.4f ndet %3d nsamp %5d nmode %2d nbin %2d" % (t2-t1,t3-t2,t4-t3,t5-t4,t6-t5, gtod.shape[0], gtod.shape[1], self.V.shape[1], len(self.bins)), level=3)
		return gtod
	def white(self, gtod, inplace=True):
		self.check_ready()
		if not inplace: gtod.copy()
		gutils.apply_window(gtod, self.nwin)
		gtod *= self.ivar[:,None]
		gutils.apply_window(gtod, self.nwin)
		return gtod
	def write(self, fname):
		data = bunch.Bunch(type="NmatDetvecs")
		for field in ["bin_edges", "eig_lim", "single_lim", "window", "nwin", "downweight",
				"bins", "D", "V", "E", "ivar"]:
			data[field] = getattr(self, field)
		bunch.write(fname, data)
	@staticmethod
	def from_bunch(data, dev=None):
		return NmatDetvecs(bin_edges=data.bin_edges, eig_lim=data.eig_lim, single_lim=data.single_lim,
				window=data.window, nwin=data.nwin, downweight=data.downweight,
				bins=data.bins, D=data.D, V=data.V, E=data.E, ivar=data.ivar, dev=dev)

def apply_vecs(ftod, iD, V, Kh, bins, tmp, vtmp, divtmp, dev=None, out=None):
	"""Jon's core for the noise matrix. Does not allocate any memory itself. Takes the work
	memory as arguments instead.

	ftod: The fourier-transform of the TOD, cast to float32. [ndet,nfreq]
	iD:   The inverse white noise variance. [nbin,ndet]
	V:    The eigenvectors. [ndet,nmode]
	Kh:   The square root of the Woodbury kernel (E"+V'DV)**-0.5. [nbin,nmode,nmode]
	bins: The frequency ranges for each bin. [nbin,{from,to}]
	tmp, vtmp, divtmp: Work arrays
	"""
	if out    is None: out = dat
	if dev    is None: dev = device.get_device()
	ndet, nmode = V.shape
	nfreq = ftod.shape[1]
	for bi, (i1,i2) in enumerate(2*bins):
		bsize = i2-i1
		# We want to perform out = iD ftod - (iD V Kh)(iD V Kh)' ftod
		# 1. divtmp = iD V      [ndet,nmode]
		# Cublas is column-major though, so to it we're doing divtmp = V iD [nmode,ndet]. OK
		dev.lib.sdgmm("R", nmode, ndet, V, nmode, iD[bi], 1, divtmp, nmode)
		# 2. vtmp   = iD V Kh   [ndet,nmode] -> vtmp = Kh divtmp [nmode,ndet]. OK
		dev.lib.sgemm("N", "N", nmode, ndet, nmode, 1, Kh[bi:bi+1], nmode, divtmp, nmode, 0, vtmp, nmode)
		# 3. tmp    = (iD V Kh)' ftod  [nmode,bsize] -> tmp = ftod vtmp.T [bsize,nmode]. OK
		dev.lib.sgemm("N", "T", bsize, nmode, ndet, 1, ftod[:,i1:i2], nfreq, vtmp, nmode, 0, tmp[:,:bsize], tmp.shape[1])
		# 4. out    = iD ftod  [ndet,bsize] -> out = ftod iD [bsize,ndet]. OK
		dev.lib.sdgmm("R", bsize, ndet, ftod[:,i1:i2], nfreq, iD[bi], 1, out[:,i1:i2], nfreq)
		# 5. out    = iD ftod - (iD V Kh)(iD V Kh)' ftod [ndet,bsize] -> out = ftod iD - ftod vtmp.T vtmp [bsize,ndet]. OK
		dev.lib.sgemm("N", "N", bsize, ndet, nmode, -1, tmp[:,:bsize], tmp.shape[1], vtmp, nmode, 1, out[:,i1:i2], nfreq)


def apply_vecs2(ftod, iD, V, Kh, bins, tmp, vtmp, divtmp, dev=None, out=None):
	"""Jon's core for the noise matrix. Does not allocate any memory itself. Takes the work
	memory as arguments instead. This is like apply_vecs, but Kh can be jagged and
	there's a separate V per bin.

	ftod: The fourier-transform of the TOD, cast to float32. [ndet,nfreq]
	iD:   The inverse white noise variance. [nbin,ndet]
	V:    The eigenvectors. [nbin][ndet,nmode]
	Kh:   The square root of the Woodbury kernel (E"+V'DV)**-0.5. [nbin][nmode,nmode]
	bins: The frequency ranges for each bin. [nbin,{from,to}]
	tmp, vtmp, divtmp: Work arrays
	"""
	if out    is None: out = dat
	if dev    is None: dev = device.get_device()
	nfreq = ftod.shape[1]
	maxnmode = divtmp.shape[1]
	for bi, (i1,i2) in enumerate(2*bins):
		bsize = i2-i1
		ndet, nmode = V[bi].shape
		# We want to perform out = iD ftod - (iD V Kh)(iD V Kh)' ftod
		# 1. divtmp = iD V      [ndet,nmode]
		# Cublas is column-major though, so to it we're doing divtmp = V iD [nmode,ndet]. OK
		dev.lib.sdgmm("R", nmode, ndet, V[bi], nmode, iD[bi], 1, divtmp[:,:nmode], maxnmode)
		# 2. vtmp   = iD V Kh   [ndet,nmode] -> vtmp = Kh divtmp [nmode,ndet]. OK
		dev.lib.sgemm("N", "N", nmode, ndet, nmode, 1, Kh[bi], nmode, divtmp, maxnmode, 0, vtmp, maxnmode)
		# 3. tmp    = (iD V Kh)' ftod  [nmode,bsize] -> tmp = ftod vtmp.T [bsize,nmode]. OK
		dev.lib.sgemm("N", "T", bsize, nmode, ndet, 1, ftod[:,i1:i2], nfreq, vtmp, maxnmode, 0, tmp[:,:bsize], tmp.shape[1])
		# 4. out    = iD ftod  [ndet,bsize] -> out = ftod iD [bsize,ndet]. OK
		dev.lib.sdgmm("R", bsize, ndet, ftod[:,i1:i2], nfreq, iD[bi], 1, out[:,i1:i2], nfreq)
		# 5. out    = iD ftod - (iD V Kh)(iD V Kh)' ftod [ndet,bsize] -> out = ftod iD - ftod vtmp.T vtmp [bsize,ndet]. OK
		dev.lib.sgemm("N", "N", bsize, ndet, nmode, -1, tmp[:,:bsize], tmp.shape[1], vtmp, maxnmode, 1, out[:,i1:i2], nfreq)

class NmatUncorr(Nmat):
	def __init__(self, spacing="exp", nbin=100, nmin=10, window=2, bins=None, ips_binned=None, ivar=None, nwin=None, dev=None):
		self.dev        = dev or device.get_device()
		self.spacing    = spacing
		self.nbin       = nbin
		self.nmin       = nmin
		self.bins       = bins
		self.ips_binned = ips_binned
		self.ivar       = ivar
		self.window     = window
		self.nwin       = nwin
		self.ready      = bins is not None and ips_binned is not None and ivar is not None
	def build(self, tod, srate, **kwargs):
		# Apply window while taking fft
		nwin  = utils.nint(self.window*srate)
		gutils.apply_window(tod, nwin)
		ft    = self.dev.lib.rfft(tod)
		# Unapply window again
		gutils.apply_window(tod, nwin, -1)
		return self.build_fourier(ft, tod.shape[1], srate, nwin=nwin)
	def build_fourier(self, ftod, nsamp, srate, nwin=0):
		ps = self.dev.np.abs(ftod)**2
		del ftod
		if   self.spacing == "exp": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
		elif self.spacing == "lin": bins = utils.expbin(ps.shape[-1], nbin=self.nbin, nmin=self.nmin)
		else: raise ValueError("Unrecognized spacing '%s'" % str(self.spacing))
		ps_binned  = self.dev.np.array([self.dev.np.mean(ps[:,b[0]:b[1]],1) for b in bins]).T
		ps_binned /= nsamp
		ips_binned = 1/ps_binned
		# Compute the representative inverse variance per sample
		ivar = self.dev.np.zeros(len(ps),ps.dtype)
		for bi, b in enumerate(bins):
			ivar += ips_binned[:,bi]*(b[1]-b[0])
		ivar /= bins[-1,1]-bins[0,0]
		return NmatUncorr(spacing=self.spacing, nbin=len(bins), nmin=self.nmin, bins=bins, ips_binned=ips_binned, ivar=ivar, window=self.window, nwin=nwin, dev=self.dev)
	def apply(self, tod, inplace=True, exp=1):
		self.check_ready()
		if not inplace: tod = tod.copy()
		gutils.apply_window(tod, self.nwin)
		ftod = self.dev.lib.rfft(tod)
		self.apply_fourier(ftod, tod.shape[1], exp=exp)
		ftod /= tod.shape[-1]
		self.dev.lib.irfft(ftod, tod)
		gutils.apply_window(tod, self.nwin)
		return tod
	def apply_fourier(self, ftod, nsamp, exp=1):
		self.check_ready()
		# Candidate for speedup in C
		for bi, b in enumerate(self.bins):
			ftod[:,b[0]:b[1]] *= (self.ips_binned[:,None,bi])**exp/nsamp
		# I divided by the normalization above instead of passing normalize=True
		# here to reduce the number of operations needed
	def white(self, tod, inplace=True):
		self.check_ready()
		if not inplace: tod = tod.copy()
		gutils.apply_window(gtod, self.nwin)
		tod *= self.ivar[:,None]
		gutils.apply_window(gtod, self.nwin)
		return tod
	def write(self, fname):
		self.check_ready()
		data = bunch.Bunch(type="NmatUncorr")
		for field in ["spacing", "nbin", "nmin", "bins", "ips_binned", "ivar", "window", "nwin"]:
			data[field] = getattr(self, field)
		bunch.write(fname, data)
	@staticmethod
	def from_bunch(data, dev=None):
		return NmatUncorr(spacing=data.spacing, nbin=data.nbin, nmin=data.nmin, bins=data.bins, ips_binned=data.ips_binned, ivar=data.ivar, window=window, nwin=nwin, dev=dev)

class NmatWhite(Nmat):
	def __init__(self, ivar=None, bsize=256, dev=None):
		self.dev   = dev or device.get_device()
		self.bsize = bsize
		self.ivar  = ivar
		self.ready = ivar is not None
	def build(self, tod, **kwargs):
		nsamp  = tod.shape[1]
		nblock = nsamp//self.bsize
		var    = self.dev.np.median(self.dev.np.var(tod[:,:nblock*self.bsize].reshape(-1,nblock,self.bsize),-1),-1)
		with utils.nowarn():
			ivar = utils.without_nan(1/var)
		return NmatWhite(ivar=ivar, bsize=self.bsize, dev=self.dev)
	def apply(self, tod, inplace=True):
		if not inplace: tod = tod.copy()
		tod *= self.ivar[:,None]
		return tod
	def white(self, tod, inplace=True): return self.apply(tod, inplace=inplace)
	def write(self, fname):
		bunch.write(fname, bunch.Bunch(type="NmatWhite"))
	@staticmethod
	def from_bunch(data, dev=None): return NmatWhite(ivar=data.ivar, dev=dev)
	def check_ready(self):
		if not self.ready:
			raise ValueError("Attempt to use partially constructed %s. Typically one gets a fully constructed one from the return value of nmat.build(tod)" % type(self).__name__)

class NmatDebug(Nmat):
	def __init__(self, ivar=None, alpha=-3, fknee=2.5, profile=None, bsize=256, dev=None):
		self.dev   = dev or device.get_device()
		self.bsize = bsize
		self.ivar  = ivar
		self.alpha = alpha
		self.fknee = fknee
		self.profile = profile
		self.ready = ivar is not None
	def build(self, tod, srate, **kwargs):
		nsamp  = tod.shape[1]
		nblock = nsamp//self.bsize
		var    = self.dev.np.median(self.dev.np.var(tod[:,:nblock*self.bsize].reshape(-1,nblock,self.bsize),-1),-1)
		with utils.nowarn():
			ivar = utils.without_nan(1/var)
		f = self.dev.np.fft.rfftfreq(nsamp, 1/srate)
		with utils.nowarn():
			profile = 1/(1+(f/self.fknee)**self.alpha)
		profile /= nsamp # fft normalization
		return NmatDebug(ivar=ivar, alpha=self.alpha, fknee=self.fknee, bsize=self.bsize, profile=profile, dev=self.dev)
	def apply(self, tod, inplace=True):
		tod = self.white(tod, inplace=inplace)
		ft  = self.dev.pools["ft"].empty((tod.shape[0],tod.shape[1]//2+1),utils.complex_dtype(tod.dtype))
		self.dev.lib.rfft(tod, ft)
		ft *= self.profile
		self.dev.lib.irfft(ft, tod)
		return tod
	def white(self, tod, inplace=True):
		if not inplace: tod = tod.copy()
		tod *= self.ivar[:,None]
		return tod

def measure_cov(d, nmax=10000):
	ap    = device.anypy(d)
	d = d[:,::max(1,d.shape[1]//nmax)]
	n,m   = d.shape
	step  = 10000
	res = ap.zeros((n,n),utils.real_dtype(d.dtype))
	for i in range(0,m,step):
		sub = ap.ascontiguousarray(d[:,i:i+step])
		res += sub.dot(ap.conj(sub.T)).real
	return res/m

def project_out(d, modes): return d-modes.T.dot(modes.dot(d))

def project_out_from_matrix(A, V):
	# Use Woodbury to project out the given vectors from the covmat A
	if V.size == 0: return A
	Q = A.dot(V)
	return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))

def freq2ind(freqs, srate, nfreq, rfun=None):
	"""Returns the index of the first fourier mode with greater than freq
	frequency, for each freq in freqs."""
	if freqs is None: return freqs
	if rfun  is None: rfun = np.ceil
	return rfun(np.asarray(freqs)/(srate/2.0)*nfreq).astype(int)

def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None):
	# Translate from frequency to index
	binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
	# Make sure no bins have two few entries
	if nmin > 0:
		binds2 = [binds[0]]
		for b in binds:
			if b-binds2[-1] >= nmin: binds2.append(b)
		binds = binds2
	# Cap at nfreq and eliminate any resulting empty bins
	binds = np.unique(np.minimum(np.concatenate([[0],binds,[nfreq]]),nfreq))
	# Go from edges to [:,{from,to}]
	bins  = np.array([binds[:-1],binds[1:]]).T
	return bins

def mycontiguous(a):
	# I used this in act for some reason, but not sure why. I vaguely remember ascontiguousarray
	# causing weird failures later in lapack
	b = np.zeros(a.shape, a.dtype)
	b[...] = a[...]
	return b

def find_modes_jon(ft, bins, eig_lim=None, single_lim=0, skip_mean=False, verbose=False, dtype=np.float32, return_eigs=False):
	ap   = device.anypy(ft)
	ndet = ft.shape[0]
	vecs = ap.zeros([ndet,0],dtype=dtype)
	eigs = []
	if not skip_mean:
		# Force the uniform common mode to be included. This
		# assumes all the detectors have accurately measured gain.
		# Forcing this avoids the possibility that we don't find
		# any modes at all.
		vecs = ap.concatenate([vecs,ap.full([ndet,1],ndet**-0.5,dtype=dtype)],1)
	for bi, b in enumerate(bins):
		d    = ft[:,b[0]:b[1]]
		cov  = measure_cov(d)
		cov  = project_out_from_matrix(cov, vecs)
		e, v = ap.linalg.eigh(cov)
		del cov
		#e, v = e.real, v.real
		#e, v = e[::-1], v[:,::-1]
		accept = ap.full(len(e), True, bool)
		if eig_lim is not None:
			# Compute median, exempting modes we don't have enough data to measure
			nsamp    = b[1]-b[0]+1
			median_e = ap.median(ap.sort(e)[::-1][:nsamp])
			accept  &= e/median_e >= eig_lim
		if verbose: print("bin %d %5d %5d: %4d modes above eig_lim" % (bi, b[0], b[1], ap.sum(accept)))
		if single_lim is not None and e.size:
			# Reject modes too concentrated into a single mode. Since v is normalized,
			# values close to 1 in a single component must mean that all other components are small
			singleness = ap.max(ap.abs(v),0)
			accept    &= singleness < single_lim
		if verbose: print("bin %d %5d %5d: %4d modes also above single_lim" % (bi, b[0], b[1], ap.sum(accept)))
		e, v = e[accept], v[:,accept]
		vecs = ap.concatenate([vecs,v],1)
		eigs.append(e)
	eigs = ap.concatenate(eigs)
	if return_eigs: return vecs, eigs
	else: return vecs

def measure_detvecs(ft, vecs):
	# Measure amps when we have non-orthogonal vecs
	ap   = device.anypy(ft)
	rhs  = vecs.T.dot(ft)
	div  = vecs.T.dot(vecs)
	amps = ap.linalg.solve(div,rhs)
	E    = ap.mean(ap.abs(amps)**2,1)
	# Project out modes for every frequency individually
	dclean = ft - vecs.dot(amps)
	# The rest is assumed to be uncorrelated
	Nu = ap.mean(ap.abs(dclean)**2,1)
	# The total auto-power
	Nd = ap.mean(ap.abs(ft)**2,1)
	return E, Nu, Nd

def sichol(A):
	iA = np.linalg.inv(A)
	try: return np.linalg.cholesky(iA), 1
	except np.linalg.LinAlgError:
		return np.linalg.cholesky(-iA), -1

def woodbury_invert(D, V, s=1):
	"""Given a compressed representation C = D + sVV', compute a
	corresponding representation for inv(C) using the Woodbury
	formula."""
	V, D = map(np.asarray, [V,D])
	# Flatten everything so we can be dimensionality-agnostic
	D = D.reshape(-1, D.shape[-1])
	V = V.reshape(-1, V.shape[-2], V.shape[-1])
	I = np.eye(V.shape[2])
	# Allocate our output arrays
	iD = gutils.safe_inv(D)
	iV = V*0
	# Invert each
	for i in range(len(D)):
		core = I*s + (V[i].T*iD[i,None,:]).dot(V[i])
		core, sout = sichol(core)
		iV[i] = iD[i,:,None]*V[i].dot(core)
	sout = -sout
	return iD, iV, sout
