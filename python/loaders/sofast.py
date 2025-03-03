import numpy as np, contextlib, json, time, os, scipy
from pixell import utils, fft, bunch, bench, sqlite
from sotodlib import preprocess, core
import fast_g3, h5py, yaml, ast, numba
import so3g
from so3g.proj import quat
from .. import gutils, gmem
import cupy, gpu_mm
from cupy.cuda import cublas

class SoFastLoader:
	def __init__(self, configfile, mul=32):
		# Set up our metadata loader
		self.config, self.context = preprocess.preprocess_util.get_preprocess_context(configfile)
		self.fast_meta = FastMeta(self.config, self.context)
		self.mul     = mul
	def query(self, query=None, wafers=None, bands=None):
		# wafers and bands should ideally be a part of the query!
		from sotodlib import mapmaking
		subids = mapmaking.get_subids(query, context=self.context)
		subids = mapmaking.filter_subids(subids, wafers=wafers, bands=bands)
		# Need base ids to look up rest of the info in obsdb
		obs_ids, wafs, bands = mapmaking.split_subids(subids)
		info     = self.context.obsdb.query()
		inds     = utils.find(info["obs_id"], obs_ids)
		# Build obsinfo for these ids
		dtype = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("r","d"),("sweep","d",(4,2))]
		obsinfo       = np.zeros(len(inds), dtype).view(np.recarray)
		obsinfo.id    = subids
		obsinfo.ndet  = 1000 # no simple way to get this :(
		obsinfo.nsamp = info["n_samples"][inds]
		obsinfo.ctime = info["start_time"][inds]
		obsinfo.dur   = info["stop_time"][inds]-info["start_time"][inds]
		# How come the parts that have to do with pointing.
		# These arrays have a nasty habit of being object dtype
		baz0  = info["az_center"][inds].astype(np.float64)
		waz   = info["az_throw" ][inds].astype(np.float64)
		bel0  = info["el_center"][inds].astype(np.float64)
		# Need the rough pointing for each observation. This isn't
		# directly available in the obsid. We will assume that each wafer-slot
		# has approximately constant pointing offsets.
		# 1. Find a reference subid for each wafer slot
		ref_ids, order, edges = get_ref_subids(subids)
		wafer_centers = np.zeros((len(subids),2))
		wafer_rads    = np.zeros((len(subids)))
		for ri, ref_id in enumerate(ref_ids):
			# 2. Get the focal plane offsets for this subid
			focal_plane = get_focal_plane(self.fast_meta, ref_id)
			mid, rad    = get_fplane_extent(focal_plane)
			wafer_centers[order[edges[ri]:edges[ri+1]]] = mid
			wafer_rads   [order[edges[ri]:edges[ri+1]]] = rad
		# Fill in the last entries in obsinfo
		obsinfo.r     = wafer_rads
		obsinfo.sweep = make_sweep(obsinfo.ctime, baz0, waz, bel0, wafer_centers)
		return obsinfo
	def load(self, subid):
		try:
			with bench.show("read meta (total)"):
				meta = self.fast_meta.read(subid)
			# Load the raw data
			with bench.show("read data (total)"):
				data = fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
			# Calibrate the data
			with bench.show("calibrate (total)"):
				obs = calibrate(data, meta)
		except Exception as e:
			# FIXME: Make this less broad
			raise utils.DataMissing(type(e) + " " + str(e))
		# Place obs.tod on gpu. Hardcoded to use scratch.tod for now
		return obs

class FastMeta:
	def __init__(self, configs, context):
		self.context = context
		# Open the database files we need for later obs lookups
		# 1. The preprocess archive index. Example:
		# /global/cfs/cdirs/sobs/sat-iso/preprocessing/satp1_20250108_init/process_archive.sqlite
		# Opening the actual file has to wait until we know what subid we have.
		self.prep_index = sqlite.open(configs["archive"]["index"])
		# 2. The det_cal index. Example:
		# /global/cfs/cdirs/sobs/metadata/satp1/manifests/det_cal/satp1_det_cal_240312m/det_cal_local.sqlite
		self.dcal_index = sqlite.open(cmeta_lookup(context["metadata"], "det_cal"))
		# 3. The detector info
		self.match_index= sqlite.open(cmeta_lookup(context["metadata"], "assignment"))
		self.det_cache  = DetCache(context.obsfiledb, self.match_index)
		# 4. Absolute calibration
		with sqlite.open(cmeta_lookup(context["metadata"], "abscal")) as index:
			acalfile = get_acalfile(index)
		self.acal_cache = AcalCache(acalfile)
		# 4. Focal plane
		self.fp_cache   = FplaneCache(cmeta_lookup(context["metadata"], "focal_plane"))
		# 5. Pointing model, which seems to be static for now
		self.pointing_model = get_pointing_model(cmeta_lookup(context["metadata"], "pointing_model"))
	def read(self, subid):
		obsid, wslot, band = subid.split(":")
		# Find which hdf files are relevant for this observation
		try:
			with bench.show("get precfile dcalfile"):
				precfile,  precgroup  = get_precfile (self.prep_index,  subid)
				dcalfile,  dcalgroup  = get_dcalfile (self.dcal_index,  subid)
		except KeyError as e: raise utils.DataMissing(str(e))
		# 1. Get our starting set of detectors
		with bench.show("get_available_dets"):
			try: readout_ids, det_ids = self.det_cache.get_dets(subid)
			except sqlite.sqlite3.OperationalError as e: raise errors.DataMissing(str(e))
			aman = core.AxisManager(core.LabelAxis("dets", readout_ids))
			aman.wrap("det_ids", det_ids, [(0,"dets")])
		# 2. Load the necessary info from det_cal
		with bench.show("det_cal"):
			with h5py.File(dcalfile, "r") as hfile:
				det_cal = hfile[dcalgroup][()]
			daman = core.AxisManager(core.LabelAxis("dets",np.char.decode(det_cal["dets:readout_id"])))
			good  = np.full(daman.dets.count, True)
		with bench.show("phase_to_pW"):
			daman.wrap("phase_to_pW", det_cal["phase_to_pW"], [(0,"dets")])
			good &= np.isfinite(daman.phase_to_pW)
		with bench.show("tau_eff"):
			daman.wrap("tau_eff", det_cal["tau_eff"], [(0,"dets")])
			good &= np.isfinite(daman.tau_eff)
		with bench.show("merge"):
			daman.restrict("dets", daman.dets.vals[good])
			aman.merge(daman)
		# 3. Load the focal plane information
		with bench.show("fp_lookup"):
			fp_info = self.fp_cache.get_by_subid(subid, self.det_cache)
		with bench.show("match dets"):
			det_ids = np.char.decode(fp_info["dets:det_id"])
			ainds, finds = utils.common_inds([aman.det_ids, det_ids])
		with bench.show("fp_setup"):
			fp_info = fp_info[finds]
			focal_plane = np.array([fp_info["xi"], fp_info["eta"], fp_info["gamma"]]).T
			good    = np.all(np.isfinite(focal_plane),1)
			aman.restrict("dets", aman.dets.vals[ainds[good]])
			aman.wrap("focal_plane", focal_plane[good], [(0,"dets")])
		# 4. Load what we need from the preprocess archive. There aren't that many
		# superflous detectors in these files, and I didn't find an efficient
		# way to read only those needed, so just read all of them and merge
		t1 = time.time()
		with PrepLoader(precfile, precgroup) as pl:
			paman = core.AxisManager(
					core.LabelAxis("dets", pl.dets),
					core.OffsetAxis("samps", count=pl.samps[0], offset=pl.samps[1]))
			# A bit awkward to time the time taken in the initialization
			bench.add("prep_loader", time.time()-t1)
			bench.print("prep_loader")
			with bench.show("relcal"):
				paman.wrap("relcal", pl.read("lpf_sig_run1/relcal","d"), [(0,"dets")])
			with bench.show("hwp_angle"):
				paman.wrap("hwp_angle", pl.read("hwp_angle/hwp_angle","s"), [(0,"samps")])
			with bench.show("hwpss_coeffs"):
				# These have order sin(1a),cos(1a),sin(2a),cos(2a),...
				paman.wrap("hwpss_coeffs", pl.read("hwpss_stats/coeffs","d"), [(0,"dets")])
			with bench.show("glitches"):
				paman.wrap("cuts_glitch", read_cuts(pl, "glitches/glitch_flags"), [(0,"dets"),(1,"samps")])
			with bench.show("cuts_2pi"):
				paman.wrap("cuts_2pi", read_cuts(pl, "jumps_2pi/jump_flag"),[(0,"dets"),(1,"samps")])
			with bench.show("cuts_slow"):
				paman.wrap("cuts_slow", read_cuts(pl, "jumps_slow/jump_flag"),[(0,"dets"),(1,"samps")])
		with bench.show("merge"):
			aman.merge(paman)
		# 4. Get stuff from the data file header
		with bench.show("get_detset (should be cached)"):
			detset = self.det_cache.get_detset(subid)
		with bench.show("get_files"):
			finfos = self.context.obsfiledb.get_files(obsid)[detset]
		# Get the filter params
		with bench.show("status"):
			status = read_wiring_status(finfos[0][0])
		with bench.show("iir_params"):
			iir_params = bunch.Bunch()
			pre = "AMCc.SmurfProcessor.Filter."
			iir_params["a"]       = np.array(ast.literal_eval(status[pre+"A"]))
			iir_params["b"]       = np.array(ast.literal_eval(status[pre+"B"]))
			iir_params["gain"]    = status[pre+"Gain"]
			iir_params["order"]   = status[pre+"Order"]
			iir_params["enabled"] = not status[pre+"Disable"]
			digitizer_freq        = 614.4e6
			ramp_max_count        = status["AMCc.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RampMaxCnt"]
			flux_ramp_rate        = digitizer_freq/2/(ramp_max_count+1)
			iir_params["fscale"]  = 1/flux_ramp_rate
		# Get our absolute calibration
		with bench.show("get_abscal"):
			abscal_cmb = self.acal_cache.get(wslot, band).abscal_cmb
		# Return our results. We don't put everything in an axismanager
		# because that has significant overhead, and we don't need an
		# axismanager for things that don't have any axes
		return bunch.Bunch(aman=aman, iir_params=iir_params, finfos=finfos,
			dac_to_phase = np.pi/2**15, timestamp_to_ctime=1e-8,
			abscal_cmb = abscal_cmb, pointing_model=self.pointing_model)

# This doesn't really belong here, unless we rename the module
def fast_data(finfos, detax, sampax, alloc=None, fields=[
		("signal",    "signal/data"),
		("timestamps","signal/times"),
		("az",        "ancil/az_enc"),
		("el",        "ancil/el_enc"),
		("roll",      "ancil/boresight_enc")]):
	import fast_g3
	if alloc is None: alloc = fast_g3.DummyAlloc()
	aman   = core.AxisManager(detax, sampax)
	fnames = [finfo[0]          for finfo in finfos]
	nsamps = [finfo[2]-finfo[1] for finfo in finfos]
	samps  = (sampax.offset, sampax.offset+sampax.count)
	i      = 0
	with fast_g3.open_multi(fnames, samps=samps, file_nsamps=nsamps) as ifile:
		fdets = ifile.fields["signal/data"].names
		rows  = utils.find(fdets, aman.dets.vals)
		for oname, iname in fields:
			ifile.queue(iname, rows=rows)
		for fi, data in enumerate(ifile.read()):
			for oname, iname in fields:
				chunk = data[iname]
				# Set up output if necessary
				if fi == 0:
					arr = alloc.zeros((chunk.shape[:-1]+(aman.samps.count,)),dtype=chunk.dtype)
					if arr.ndim == 1: aman.wrap(oname, arr, [(0,"samps")])
					else:             aman.wrap(oname, arr, [(0,"dets"),(1,"samps")])
				# Copy into output arrays
				aman[oname][...,i:i+chunk.shape[-1]] = chunk
			i += chunk.shape[-1]
	return aman

def calibrate(data, meta):
	# Merge the cuts. Easier to deal with just a single cuts object
	with bench.show("merge_cuts"):
		cuts = merge_cuts([meta.aman[name] for name in ["cuts_glitch","cuts_2pi","cuts_slow"]])
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left")
	# Go to a fourier-friendly length
	mul   = 32
	nsamp = fft.fft_len(data.signal.shape[1]//mul, factors=[2,3,5,7])*mul
	timestamps, signal, cuts, az, el, roll, hwp_angle = [a[...,:nsamp] for a in [data.timestamps,data.signal,cuts,data.az,data.el,data.roll,meta.aman.hwp_angle]]

	with bench.show("ctime"):
		ctime   = timestamps * meta.timestamp_to_ctime
	# Calibrate the pointing
	with bench.show("boresight"):
		az      = az   * utils.degree
		el      = el   * utils.degree
		roll    =-roll * utils.degree
	with bench.show("pointing correction"):
		az, el, roll = apply_pointing_model(az, el, roll, meta.pointing_model)
	# Do we need to deslope at float64 before it is safe to drop to float32?
	with bench.show("signal → gpu", tfun=gutils.cutime):
		signal_ = gmem.scratch.ft.array(signal)
	with bench.show("signal → float32", tfun=gutils.cutime):
		signal  = gmem.scratch.tod.empty(signal.shape, np.float32)
		signal[:] = signal_
	with bench.show("calibrate", tfun=gutils.cutime):
		# Calibrate to CMB µK
		phase_to_cmb = 1e6 * meta.abscal_cmb * meta.aman.phase_to_pW[:,None] * meta.aman.relcal[:,None]
		signal *= cupy.array(meta.dac_to_phase * phase_to_cmb)
	# Subtract the HWP scan-synchronous signal. We do this before deglitching because
	# it's large and doesn't follow the simple slopes and offsets we assume there
	with bench.show("subtract_hwpss", tfun=gutils.cutime):
		nmode = 16
		signal = subtract_hwpss(signal, hwp_angle, meta.aman.hwpss_coeffs[:,:nmode]*phase_to_cmb)
	# Deglitching. Measure the values v1,v2 at the edge of
	# each cut region. Subtract v2-v1 from everything following the cut, and fill the cut
	# with v1
	# For each cut region we want a region up to n samples wide on its borders, but not
	# inside other cuts
	w = 10
	with bench.show("get_cut_borders"):
		borders = get_cut_borders(cuts,w)
	# Hm, need to read out mean in each of these border regions. No easy way to do this
	# Let's see how expensive it is to loop. 28 ms. Probably worth implementing in
	# low-level code
	with bench.show("border vals", tfun=gutils.cutime):
		bvals = cupy.zeros((len(borders),2), signal.dtype)
		for di, (b1,b2) in enumerate(cuts.bins):
			for ri in range(b1,b2):
				bor   = borders[ri]
				bvals1= signal[di,bor[0,0]:bor[0,1]]
				bvals2= signal[di,bor[1,0]:bor[1,1]]
				if bvals1.size  > 0: v1 = np.mean(bvals1)
				if bvals2.size  > 0: v2 = np.mean(bvals2)
				if bvals1.size == 0: v1 = v2
				if bvals2.size == 0: v2 = v1
				bvals[ri,0] = v1
				bvals[ri,1] = v2
	# 45 ms for this one. Not as slow as I feared, but should still be
	# optimized. Unless it's faster on a gpu
	with bench.show("deglich", tfun=gutils.cutime):
		# Will subtract v2-v1 from entire region after that cut.
		jumps = bvals[:,1]-bvals[:,0]
		cumj  = cupy.cumsum(jumps)
		# Will subtract cumj[i] for range[i,1]:range[i+1,0].
		# Will set range[i,0]:range[i,1] to bvals[i,1]-cumj[i]
		for di, (b1,b2) in enumerate(cuts.bins):
			if b1 >= b2: continue
			dcumj = cumj[b1:b2]
			if b1 > 0: dcumj = dcumj - cumj[b1-1] # NB: -= would clobber!
			for i, ri in enumerate(range(b1,b2)):
				r1,r2 = cuts.ranges[ri]
				r3    = cuts.ranges[ri+1,0] if ri < b2-1 else cuts.nsamp
				signal[di,r1:r2]  = bvals[ri,1]-dcumj[i]
				signal[di,r2:r3] -= dcumj[i]

	# 100 ms for this :(
	with bench.show("deslope", tfun=gutils.cutime):
		deslope(signal, w=w, inplace=True)

	# FFT stuff should definitely be on the gpu. 640 ms
	with bench.show("fft", tfun=gutils.cutime):
		ftod = gmem.scratch.ft.empty((signal.shape[0],signal.shape[1]//2+1), utils.complex_dtype(signal.dtype))
		gpu_mm.cufft.rfft(signal, ftod, plan_cache=gmem.plan_cache)
		norm = 1/signal.shape[1]
	# Deconvolve iir and time constants
	with bench.show("iir_filter", tfun=gutils.cutime):
		dt    = (ctime[-1]-ctime[0])/(ctime.size-1)
		freqs = cupy.fft.rfftfreq(nsamp, dt).astype(signal.dtype)
		z     = cupy.exp(-2j*np.pi*meta.iir_params.fscale*freqs)
		A     = cupy.polyval(cupy.array(meta.iir_params.a[:meta.iir_params.order+1][::-1]), z)
		B     = cupy.polyval(cupy.array(meta.iir_params.b[:meta.iir_params.order+1][::-1]), z)
		iir_filter = A/B # will multiply by this
		iir_filter *= norm # Hack: cheap to handle normalization here
		ftod *= iir_filter
	with bench.show("time consts", tfun=gutils.cutime):
		# I can't find an efficient way to do this. BLAS can't
		# do it since it's a triple multiplication. Hopefully the
		# gpu won't have trouble with it
		ftod *= 1 + 2j*np.pi*cupy.array(meta.aman.tau_eff[:,None])*freqs
	# Back to real space
	with bench.show("ifft", tfun=gutils.cutime):
		gpu_mm.cufft.irfft(ftod, signal, plan_cache=gmem.plan_cache)

	with bench.show("measure noise", tfun=gutils.cutime):
		rms  = measure_rms(signal)

	# Restrict to these detectors
	with bench.show("final detector prune", tfun=gutils.cutime):
		tol    = 0.1
		ref    = np.median(rms[rms!=0])
		good   = (rms > ref*tol)&(rms < ref/tol)
		signal = cupy.ascontiguousarray(signal[good]) # 600 ms!
		good   = good.get() # cuts, dets, fplane etc. need this on the cpu
		cuts   = cuts  [good]
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left")

	# Sogma uses the cut format [{dets,starts,lens},:]. Translate to this

	with bench.show("cuts reformat"):
		ocuts = np.array([
			get_range_dets(cuts.bins, cuts.ranges),
			cuts.ranges[:,0], cuts.ranges[:,1]-cuts.ranges[:,0]], dtype=np.int32)
		ocuts = ocuts[:,ocuts[0]>=0]

	# Our goal is to output what sogma needs. Sogma works on these fields:
	res  = bunch.Bunch()
	res.dets         = meta.aman.dets.vals[good]
	res.point_offset = meta.aman.focal_plane[good,1::-1]
	res.polangle     = meta.aman.focal_plane[good,2]
	res.ctime        = ctime
	res.boresight    = np.array([el,az]) # FIXME: roll
	res.tod          = signal
	res.cuts         = ocuts
	return res

#################
# Helpers below #
#################

class DetCache:
	def __init__(self, obsfiledb, match_index):
		self.obsfiledb = obsfiledb
		self.match_index= match_index
		# There won't be many unique ones here, and they're
		# quite small, so we just cache them all
		self.hfile_cache= {}
		self.det_cache = {}
		self.dset_cache= {}
	def get_detsets(self, obsid):
		if obsid not in self.dset_cache:
			with bench.show("get_detsets"):
				self.dset_cache[obsid] = self.obsfiledb.get_detsets(obsid)
		return self.dset_cache[obsid]
	def get_detset(self, subid):
		obsid, wslot, band = subid.split(":")
		ind = int(wslot[2:])
		return self.get_detsets(obsid)[ind]
	def get_dets(self, subid):
		obsid, wslot, band = subid.split(":")
		if subid not in self.det_cache:
			detset = self.get_detset(subid)
			# Get the hdf file for this detset
			hfname, group = get_matchfile(self.match_index, detset)
			if hfname not in self.hfile_cache:
				self.hfile_cache[hfname] = h5py.File(hfname, "r")
			detinfo= self.hfile_cache[hfname][group][()]
			good   = np.char.find(detinfo["dets:det_id"],band.encode())>=0
			definfo = detinfo[good]
			dets   = np.char.decode(detinfo["dets:readout_id"])
			names  = np.char.decode(detinfo["dets:det_id"])
			self.det_cache[subid] = (dets, names)
		return self.det_cache[subid] # returns (readout_ids, det_ids)
	def close(self):
		for key in self.hfile_cache:
			self.hfile_cache[key].close()
	def __enter__(self): return self
	def __exit__(self, *args, **kwargs):
		self.close()

class FplaneCache:
	def __init__(self, fname):
		# The index file here is small, with efficient time
		# ranges, so we can just read in the whole thing
		self.fname = fname
		self.index = {}
		with sqlite.open(fname) as sfile:
			query = "select [dets:stream_id], [obs:timestamp__lo], [obs:timestamp__hi], files.name, dataset from map inner join files on file_id = files.id"
			# Wafer here is e.g. ufm_mv15, not the wafer slot (w.g. ws0)
			for wafer_name, t1, t2, hfname, gname in sfile.execute(query):
				if wafer_name not in self.index:
					self.index[wafer_name] = []
				self.index[wafer_name].append((t1,t2,hfname,gname))
		# The cache for the actual focal plane structures
		self.fp_cache = {}
	def find_entry(self, wafer_name, t):
		entries = self.index[wafer_name]
		for i, entry in enumerate(entries):
			if entry[0] <= t and t < entry[1]:
				return entry
		# Return latest entry by default
		return entry
	def get_by_wafer(self, wafer_name, t):
		"""This takes a full wafer name, which is not the same as a wafer slot"""
		entry = self.find_entry(wafer_name, t)
		key   = (entry[2],entry[3])
		if key not in self.fp_cache:
			fname = os.path.dirname(self.fname) + "/" + entry[2]
			with h5py.File(fname,"r") as hfile:
				self.fp_cache[key] = hfile[entry[3]][()]
		return self.fp_cache[key]
	def get_by_subid(self, subid, det_cache):
		"""returns array with [('dets:det_id', 'S18'), ('xi', '<f4'), ('eta', '<f4'), ('gamma', '<f4')]"""
		obs_id, wafer_slot, band = subid.split(":")
		ctime      = float(obs_id.split("_")[1])
		wafer_name = "_".join(det_cache.get_detset(subid).split("_")[:2])
		return self.get_by_wafer(wafer_name, ctime)

class AcalCache:
	def __init__(self, fname):
		self.fname  = fname
		self.raw    = bunch.read(fname).abscal
		self.lookup = {}
		for row in self.raw:
			key = row["dets:wafer_slot"].decode() + ":" + row["dets:wafer.bandpass"].decode()
			self.lookup[key] = bunch.Bunch(
				abscal_cmb = row["abscal_cmb"],
				abscal_rj  = row["abscal_rj"],
				beam_fwhm  = row["beam_fwhm"],
				beam_solid_angle = row["beam_solid_angle"],
				cal_source = row["cal_source"].decode(),
			)
	def get(self, wafer_band, band=None):
		"""Either get("ws0:f150") or get("ws0", "f150") work"""
		if band is not None: wafer_band = wafer_band + ":" + band
		return self.lookup[wafer_band]

# This takes 0.3 s the first time and 0 s later.
# Considering that the python version takes 0.3 ms,
# we would need to call this 1000 times to start
# benefiting
#
#def get_range_dets(bins, ranges):
#	bind = np.full(len(ranges),-1,np.int32)
#	_get_range_dets(bins, ranges, bind)
#	return bind
#@numba.njit
#def _get_range_dets(bins, ranges, bind):
#	for bi, bin in enumerate(bins):
#		for i in range(bin[0],bin[1]):
#			bind[i] = bi

def get_ref_subids(subids):
	"""Return one subid for each :ws:band combination"""
	subs  = np.char.partition(subids, ":")[:,2]
	uvals, order, edges = utils.find_equal_groups_fast(subs)
	uvals = subids[order[edges[:-1]]]
	return uvals, order, edges

def get_focal_plane(fast_meta, subid):
	fp_info     = fast_meta.fp_cache.get_by_subid(subid, fast_meta.det_cache)
	focal_plane = np.array([fp_info["xi"], fp_info["eta"], fp_info["gamma"]]).T # [:,{xi,eta,gamma}]
	good        = np.all(np.isfinite(focal_plane),1)
	focal_plane = focal_plane[good]
	return focal_plane

def get_fplane_extent(focal_plane):
	mid = np.mean(focal_plane,0)[:2]
	rad = np.max(utils.angdist(focal_plane[:,:2],mid[:,None]))
	return mid, rad

def make_sweep(ctime, baz0, waz, bel0, off, npoint=4):
	from pixell import coordinates
	# given ctime,baz0,waz,bel [ntod], off[ntod,{xi,eta}], make
	# make sweeps[ntod,npoint,{ra,dec}]
	# so3g can't handle per-sample pointing offsets, so it would
	# force us to loop here. We therefore simply modify the pointing
	# offsets to we can simply add them to az,el
	az_off, el = coordinates.euler_rot((0.0, -bel0, 0.0), off.T)
	az  = (baz0+az_off)[:,None] + np.linspace(-0.5,0.5,npoint)*waz[:,None]
	el  = el   [:,None] + az*0
	ts  = ctime[:,None] + az*0
	sightline = so3g.proj.coords.CelestialSightLine.az_el(
		ts.reshape(-1), az.reshape(-1), el.reshape(-1), site="so", weather="typical")
	pos_equ = np.asarray(sightline.coords()) # [ntot,{ra,dec,cos,sin}]
	sweep   = pos_equ.reshape(len(ctime),npoint,4)[:,:,:2]
	return sweep

def get_range_dets(bins, ranges):
	bind  = np.full(len(ranges),-1,np.int32)
	for bi, bin in enumerate(bins):
		bind[bin[0]:bin[1]] = bi
	return bind

def counts_to_bins(counts):
	edges  = utils.cumsum(counts, endpoint=True)
	bins   = np.zeros((len(counts),2),np.int32)
	bins[:,0] = edges[:-1]
	bins[:,1] = edges[1:]
	return bins

def simplify_cuts(bins, ranges):
	"""Get rid of empty cuts"""
	# Find out which bin each range belongs to. This is slow
	bind  = get_range_dets(bins, ranges)
	# Count how many good ranges each has
	good  = (ranges[:,1]>ranges[:,0])&(bind>=0)
	ngood = np.bincount(bind[good], minlength=len(bins))
	# Generate simplified bins and ranges
	obins   = counts_to_bins(ngood)
	oranges = ranges[good]
	return obins, oranges

def merge_cuts(cuts):
	"""Get the union of the given list of Sampcuts"""
	# 1. Flatten each cut
	ndet = len(cuts[0].bins)
	nsamp= cuts[0].nsamp
	N    = nsamp+1 # +1 to avoid merging across dets
	franges = []
	for ci, cut in enumerate(cuts):
		bind    = get_range_dets(cut.bins, cut.ranges)
		franges.append(cut.ranges + bind[:,None]*N)
	# 3. Merge overlapping ranges. We count how many times
	# we enter and exit a cut region across all, and look for places
	# where this is nonzero
	franges = np.concatenate(franges).reshape(-1)
	if franges.size == 0: return Sampcut(nsamp=nsamp)
	# FIXME: Handle everything cut-case
	vals    = np.zeros(franges.shape,np.int32)
	vals[0::2] =  1
	vals[1::2] = -1
	order   = np.argsort(franges)
	franges = franges[order]
	vals    = vals   [order]
	incut   = np.cumsum(vals)>0
	# Use this to build the output cuts
	changes = np.diff(incut.astype(np.int32))
	starts  = np.where(changes>0)[0]+1
	starts  = np.concatenate([[0],starts])
	ends    = np.where(changes<0)[0]+1
	oranges = np.array([franges[starts],franges[ends]]).T
	# Unflatten into per-detector
	bind     = oranges[:,0]//N
	oranges -= bind[:,None]*N
	nperdet  = np.bincount(bind, minlength=ndet)
	obins    = counts_to_bins(nperdet)
	# Finally construct a new cut with the result
	ocut     = Sampcut(obins, oranges, nsamp)
	return ocut

def get_cut_borders(cuts, n=1):
	bind      = get_range_dets(cuts.bins, cuts.ranges)
	border    = np.zeros((len(cuts.ranges),2,2),np.int32)
	border[:,0,1] = cuts.ranges[:,0]
	border[:,1,0] = cuts.ranges[:,1]
	# pad on both ends to make logic simpler
	padbind   = np.concatenate([[-1],bind,[-1]])
	padranges = np.concatenate([[[0,0]],cuts.ranges,[[cuts.nsamp,cuts.nsamp]]])
	left      = np.where(padbind[:-2]==padbind[1:-1],padranges[:-2,1],0)
	right     = np.where(padbind[ 2:]==padbind[1:-1],padranges[ 2:,0],cuts.nsamp)
	border[:,0,0] = np.maximum(cuts.ranges[:,0]-n, left)
	border[:,1,1] = np.minimum(cuts.ranges[:,1]+n, right)
	return border

# TODO: Consider using just [{det,start,len},:] or its transpose as
# the cuts format. This is what sogma uses directly, and may be easier
# to work with overall.
class Sampcut:
	def __init__(self, bins=None, ranges=None, nsamp=0, simplify=False):
		if bins   is None: bins   = np.zeros((0,2),np.int32)
		if ranges is None: ranges = np.zeros((0,2),np.int32)
		if simplify:
			with bench.show("simplify cuts"):
				bins, ranges = simplify_cuts(bins, ranges)
		self.bins   = np.asarray(bins,   dtype=np.int32) # (ndet,  {from,to})
		self.ranges = np.asarray(ranges, dtype=np.int32) # (nrange,{from,to})
		self.nsamp  = nsamp # not *really* necessary, but nice to have
	def __getitem__(self, sel):
		"""Extract a subset of detectors and/or samples. Always functions
		as a slice, so the reslut is a new Sampcut. Only standard slicing is
		allowed in the sample direction - no direct indexing or indexing by lists."""
		if not isinstance(sel, tuple): sel = (sel,)
		if len(sel) == 0: return self
		if len(sel) > 2: raise IndexError("Too many indices for Sampcut. At most 2 indices supported")
		# Handle detector parts
		bins = self.bins[sel[0]]
		if len(sel) == 1:
			return Sampcut(bins, self.ranges, self.nsamp)
		start = sel[1].start or 0
		stop  = sel[1].stop  or self.nsamp
		step  = sel[1].step  or 1
		if start < 0: start += self.nsamp
		if stop  < 0: stop  += self.nsamp
		if step != 1: raise ValueError("stride != 1 not supported")
		ranges = np.clip(self.ranges, start, stop)
		return Sampcut(bins, ranges, stop-start, simplify=True)
	@property
	def shape(self): return (len(self.bins),self.nsamp)

def expand_cuts_sampcut(shape, ends, intervals, inds=None):
	if len(shape) != 2:
		raise ValueError("Expected (ndet,nsamp) RangesMatrix")
	bins = np.zeros((shape[0],2),np.int32)
	# Ends is an index into flattened ranges
	ends = ends//2
	bins[ :,1] = ends
	bins[1:,0] = ends[:-1]
	cuts = Sampcut(bins, intervals.reshape(-1,2), shape[1])
	if inds is not None: cuts = cuts[inds]
	return cuts

def read_cuts(pl, path):
	shape     = pl.read(path + "/shape")
	edges     = pl.read(path + "/ends")
	intervals = pl.read(path + "/intervals")
	return expand_cuts_sampcut(shape, edges, intervals, inds = pl.inds)

def read_jumps(pl, path):
	shape = pl.read(path + "/shape")
	edges = pl.read(path + "/indptr")
	samps = pl.read(path + "/indices")
	data  = pl.read(path + "/data")
	edges, samps, data = unpad_jumps(edges, samps, data)
	return sparse.csr_matrix((data,samps,edges),shape=shape)

# This thing should basically be a lazy-loading axis-manager,
# but let's wait with that until we have something that works
class PrepLoader:
	def __init__(self, fname, path):
		self.fname, self.path = fname, path
		self.hfile = h5py.File(fname, "r")
		self.group = self.hfile[path]
		meta = json.loads(self.group.attrs["_axisman"])
		# Just get the det info for now
		for entry in meta["schema"]:
			if entry["encoding"] == "axis":
				if entry["args"][0] == "dets":
					self.dets = entry["args"][1]
				elif entry["args"][0] == "samps":
					self.samps = entry["args"][1:3]
		self.inds = np.arange(len(self.dets))
		self.ssel = slice(None)
	def __enter__(self): return self
	def __exit__(self, *args, **kwargs):
		self.close()
	def close(self): self.hfile.close()
	def restrict_dets(self, dets):
		self.inds = utils.find(self.dets, dets)
	def restrict_samps(self, sel):
		raise NotImplementedError
	def read(self, path, sel=""):
		res = self.group[path]
		pre = ()
		for c in sel:
			if   c == ":": pass
			elif c == "d": res = res[self.inds]
			elif c == "s": res = res[self.ssel]
			else: raise ValueError("Unrecognied axtype '%s'" % str(c))
			pre = pre + (slice(None),)
		res = res[()]
		return res

def get_precfile(indexdb, subid):
	obsid, wslot, band = subid.split(":")
	query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id], [dets:wafer_slot], [dets:wafer.bandpass] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' AND [dets:wafer_slot] = '%s' AND [dets:wafer.bandpass] = '%s' LIMIT 1;" % (obsid, wslot, band)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in preprocess index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_dcalfile(indexdb, subid):
	obsid, wslot, band = subid.split(":")
	query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' LIMIT 1;" % (obsid)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in det_cal index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_matchfile(indexdb, detset):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:detset] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:detset] = '%s' LIMIT 1;" % (detset)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in det match index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_acalfile(indexdb):
	return os.path.dirname(indexdb.fname) + "/" + list(indexdb.execute("SELECT name FROM files INNER JOIN map ON files.id = map.file_id WHERE map.dataset = 'abscal'"))[0][0]

def get_pointing_model(fname):
	# Looks like the pointing model doesn't support time dependence at the
	# moment, so we just have this simple function
	with sqlite.open(fname) as sfile:
		rows = list(sfile.execute("select name from files limit 2"))
		if len(rows) != 1: raise IOError("Time-dependent pointing model loading not implemented. Figure out new format and implement")
		hfname  = os.path.dirname(fname) + "/" + rows[0][0]
		with h5py.File(hfname, "r") as hfile:
			param_str = hfile["pointing_model"].attrs["_scalars"]
		return bunch.Bunch(**json.loads(param_str))

def cmeta_lookup(cmeta, name):
	for entry in cmeta:
		if "label" in entry and entry["label"] == name:
			return entry["db"]

def read_wiring_status(fname, parse=True):
	for frame in fast_g3.get_header_frames(fname)["frames"]:
		if frame["type"] == "wiring":
			wiring = frame
	status = wiring["fields"]["status"]
	if parse:
		status = yaml.safe_load(status)
	return status

def subtract_hwpss(signal, hwp_angle, coeffs):
	if signal.dtype != np.float32: raise ValueError("Only float32 supported")
	ap        = gutils.anypy(signal)
	hwp_angle = ap.asarray(hwp_angle, dtype=signal.dtype)
	coeffs    = ap.asarray(coeffs,    dtype=signal.dtype)
	ncoeff    = coeffs.shape[1]
	B         = ap.zeros((ncoeff,len(hwp_angle)),signal.dtype)
	# This can be done with recursion formulas, but
	# the gains are only few ms on the cpu. Let's keep this
	# for the time being
	with bench.show("build basis"):
		for i in range(ncoeff):
			mode = i//2+1
			fun  = [ap.sin, ap.cos][i&1]
			B[i] = fun(mode*hwp_angle)
	# We want signal -= coeffs.dot(B): [ndet,n]*[n,nsamp]. Fortran is
	# column-major though, so it wants [nsamp,n]*[n,ndet]
	if isinstance(signal, cupy.ndarray):
		handle = cupy.cuda.Device().cublas_handle
		ndet, nsamp = signal.shape
		one       = np.full(1, 1, signal.dtype)
		minus_one = np.full(1,-1, signal.dtype)
		cublas.sgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
			nsamp, ndet, ncoeff, minus_one.ctypes.data, B.data.ptr, nsamp, coeffs.data.ptr, ncoeff,
			one.ctypes.data, signal.data.ptr, nsamp)
	else:
		scipy.linalg.blas.sgemm(-1, B.T, coeffs.T, beta=1, c=signal.T, overwrite_c=True)
	return signal

def polar_2d(x, y):
	r = (x**2+y**2)**0.5
	φ = np.arctan2(y,x)
	return r, φ

def measure_rms(tod, bsize=32, nblock=10):
	ap  = gutils.anypy(tod)
	tod = tod[:,:tod.shape[1]//bsize*bsize]
	tod = tod.reshape(tod.shape[0],-1,bsize)
	bstep = max(1,tod.shape[1]//nblock)
	tod = tod[:,::bstep,:][:,:nblock,:]
	return ap.median(ap.std(tod,-1),-1)

# This takes 250 ms! And the quaternion stuff would be tedious
# (but not difficult as such) to implement on the gpu
def apply_pointing_model(az, el, roll, model):
	# Ensure they're arrays, and avoid overwriting. These are
	# small arrays anyways
	[az, el, roll] = [np.array(a) for a in [az,el,roll]]
	if   model.version == "sat_naive": pass
	elif model.version == "sat_v1":
		# Simple offsets
		az   += model.enc_offset_az
		el   += model.enc_offset_el
		roll -= model.enc_offset_boresight
		# az twist
		az   += model.az_rot * el
		# The rest is more involved
		amp, phi = polar_2d(model.base_tilt_cos, model.base_tilt_sin)
		q_base_tilt = quat.euler(2, phi) * quat.euler(1, amp) * quat.euler(2, -phi)
		q_fp_rot    = ~quat.rotation_xieta(model.fp_rot_xi0, model.fp_rot_eta0)
		q_fp_off    = quat.rotation_xieta(model.fp_offset_xi0, model.fp_offset_eta0)
		q_hor_raw   = quat.rotation_lonlat(-az,el)
		q_hor_fix   = q_base_tilt * q_hor_raw * q_fp_off * ~q_fp_rot * quat.euler(2,roll) * q_fp_rot
		az, el, roll= quat.decompose_lonlat(q_hor_fix)
		az         *= -1
	else: raise ValueError("Unrecognized model '%s'" % str(model.version))
	return az, el, roll

def detwise_axb(tod, x, a, b, inplace=False, tod_mul=0, abmul=1):
	ap = gutils.anypy(tod)
	if tod.dtype != np.float32: raise ValueError("Only float32 supported")
	if not inplace: tod = tod.copy()
	ndet, nsamp = tod.shape
	B    = ap.empty([2,nsamp],dtype=tod.dtype) # [{a,b},nsamp]
	B[0] = x
	B[1] = 1
	coeffs = ap.array([a,b]) # [{a,b},ndet]
	# We want [ndet,2]*[2,nsamp] → [ndet,nsamp], but fortran wants
	# [nsamp,2]*[2,ndet] = [nsamp,ndet]
	if isinstance(tod, cupy.ndarray):
		handle = cupy.cuda.Device().cublas_handle
		ndet, nsamp = tod.shape
		tod_mul_  = np.full(1, tod_mul, tod.dtype)
		abmul_    = np.full(1, abmul,   tod.dtype)
		cublas.sgemm(handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T,
			nsamp, ndet, 2, abmul_.ctypes.data, B.data.ptr, nsamp, coeffs.data.ptr, ndet,
			tod_mul_.ctypes.data, tod.data.ptr, nsamp)
	else:
		scipy.linalg.blas.sgemm(abmul, B.T, coeffs.T, beta=tod_mul, c=tod.T, trans_b=True, overwrite_c=True)
	return tod

def deslope(signal, w=10, inplace=False):
	ap = gutils.anypy(signal)
	if not inplace: signal = signal.copy()
	v1 = ap.mean(signal[:, :w],1)
	v2 = ap.mean(signal[:,-w:],1)
	x  = ap.linspace(0, 1, signal.shape[1], dtype=signal.dtype)
	return detwise_axb(signal, x, v2-v1, v1, tod_mul=1, abmul=-1, inplace=True)
