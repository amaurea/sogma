import numpy as np, contextlib, json, time, os, scipy, re, yaml, ast, h5py
from pixell import utils, bunch, bench, sqlite, coordsys, config
from .. import device, gutils, socut, errors
from . import socommon, soquery, minisotodlib
from .socommon import cmeta_lookup

class SoFastLoader(socommon.Loader):
	"""Uses memory pools tod and ft"""
	def __init__(self, context_or_config_or_name, dev=None, dtype=np.float32, pool_map={}):
		super().__init__(dev=dev, pool_map=pool_map, dtype=dtype)
		self.context   = socommon.get_expanded_context(context_or_config_or_name)
		self.obsdb     = sqlite.open(self.context["obsdb"])
		self.predb     = sqlite.open(socommon.cmeta_lookup(self.context, "preprocess"))
		self.fast_meta = FastMeta(self.context)
		# Precompute the set of valid tags. Sadly this requires a scan through the whole
		# database, but it's not that slow so far
		self.tags      = soquery.get_tags(self.obsdb.conn)
	def query(self, query=None, dets=None, detids=None):
		res_db, pycode, slices = soquery.eval_query(self.obsdb.conn, query, tags=self.tags, predb=self.predb)
		obsinfo = soquery.finish_query(res_db, pycode, slices)
		# If detectors are restricted, estimate how many we will end up with
		if dets   is not None: obsinfo.ndet = np.minimum(obsinfo.ndet, len(dets))
		if detids is not None: obsinfo.ndet = np.minimum(obsinfo.ndet, len(detids))
		return SoFastLoadInfo(self, obsinfo, dets=dets, detids=detids)
	def probe(self, linfo, id, dets=None, detids=None):
		# TODO: Think about exception classification and propagation here
		dets   = socommon.det_intersect(linfo.dets,   dets)
		detids = socommon.det_intersect(linfo.detids, detids)
		with bench.mark("SoFastLoader load_meta"):
			meta = self.fast_meta.read(id, dets=dets, detids=detids)
			if meta.aman.dets.count == 0:
				raise utils.DataMissing("no detectors left after meta: raw %d meta 0" % meta.ndet_full)
		ind   = linfo.omap[id]
		row   = linfo.obsinfo[ind]
		# Find which time-range we cover. Natively sample 0:obsinfo.nsamp cover
		# time obsinfo.ctime:obsinfo.ctime+obsinfo.dur, but we only use the range
		# aman.samps.offset:aman.samps.offset+aman.samps.nsamp of this
		srate = (row.nsamp-1)/row.dur
		t1    = row.ctime + meta.aman.samps.offset/srate
		nsamp = meta.aman.samps.count
		return socommon.ProbeInfo(meta.aman.dets.count, nsamp, t1, srate, meta=meta)
	def load(self, linfo, id, dets=None, detids=None, samprange=None, pinfo=None):
		if pinfo is None: pinfo = linfo.probe(id, dets=dets, detids=detids)
		# TODO: Exceptions
		meta = pinfo.meta
		# Restrict to target sample range
		if samprange is not None:
			off = meta.aman.samps.offset
			meta.aman.restrict("samps", slice(samprange[0]+off,samprange[1]+off), in_place=True)
		# Load the raw data
		with bench.mark("SoFastLoader fast_data"):
			data = fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
		# Calibrate the data
		with bench.mark("SoFastLoader calibrate"):
			obs = calibrate(data, meta, mul=self.dev.lib.bsize, dev=self.dev, dtype=self.dtype,
				pool_map=self.pool_map)
		obs.errors = []
		# Record what data we covered in the obs. Useful for logging
		obs.subids = [socommon.srange_suffix(id, samprange)]
		return obs
	def prealloc(self, linfo):
		def s(ndet, nsamp): return utils.ceil(np.max(ndet*(nsamp+2))) # fourier-safe size
		# Max size of our output tod
		obsinfo = linfo.obsinfo
		nout = s(obsinfo.ndet, obsinfo.nsamp)
		self.pool("tod").empty(nout, dtype=self.dtype)
		self.pool("ft") .empty(nout, dtype=self.dtype)
	def group_obs(self, linfo, mode="obs"):
		return socommon.group_obs(linfo.obsinfo, mode=mode)

class SoFastLoadInfo(socommon.LoadInfo): pass

config.default("cuts_optional", False, "If true, it's not an error for the expected cuts to be missing from the preprocess database")
class FastMeta:
	def __init__(self, context):
		self.context   = context
		self.obsfiledb = minisotodlib.ObsFileDb(context["obsfiledb"])
		# Open the database files we need for later obs lookups
		# 1. The preprocess archive index. Example:
		# /global/cfs/cdirs/sobs/sat-iso/preprocessing/satp1_20250108_init/process_archive.sqlite
		# Opening the actual file has to wait until we know what subid we have.
		self.prep_index = sqlite.open(cmeta_lookup(context, "preprocess"))
		# 2. The det_cal index. Example:
		# /global/cfs/cdirs/sobs/metadata/satp1/manifests/det_cal/satp1_det_cal_240312m/det_cal_local.sqlite
		self.dcal_index = sqlite.open(cmeta_lookup(context, "det_cal"))
		# 2b. Optional relcal/flatfield. Same structure as detcal
		rcal_file = cmeta_lookup(context, "relcal")
		self.rcal_cache = RelcalCache(rcal_file) if rcal_file else None
		# 3. The detector info
		smurf_info = SmurfInfo(sqlite.open(cmeta_lookup(context, "smurf")))
		match_info = AssignmentInfo(sqlite.open(cmeta_lookup(context, "assignment")))
		wafer_info = WaferInfo(sqlite.open(cmeta_lookup(context, "wafer_info")))
		self.det_cache  = DetCache(self.obsfiledb, smurf_info, match_info, wafer_info)
		# 4. Absolute calibration
		self.acal_cache = AcalCache(cmeta_lookup(context, "abscal"))
		# 4. Focal plane
		self.fp_cache   = FplaneCache(cmeta_lookup(context, "focal_plane"))
		# 5. Pointing model, which seems to be static for now
		self.pointing_model_cache = PointingModelCache(cmeta_lookup(context, "pointing_model"))
		# 6. Wiring status cache
		self.wiring_cache = WiringCache(fields=[
			"AMCc.SmurfProcessor.Filter.A",
			"AMCc.SmurfProcessor.Filter.B",
			"AMCc.SmurfProcessor.Filter.Gain",
			"AMCc.SmurfProcessor.Filter.Order",
			"AMCc.SmurfProcessor.Filter.Disable",
			"AMCc.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RampMaxCnt",
		])
	def read(self, subid, dets=None, detids=None):
		obsid, wslot, band, det_type = split_subid(subid)
		# Find which hdf files are relevant for this observation
		with bench.mark("fm_prepfile"):
			prepfile,  prepgroup  = get_prepfile (self.prep_index,  subid)
			dcalfile,  dcalgroup  = get_dcalfile (self.dcal_index,  subid)
		# 1. Get our starting set of detectors
		with bench.mark("fm_dets"):
			try: detinfo = self.det_cache.get_dets(subid)
			except sqlite.sqlite3.OperationalError as e: raise errors.DataMissing(str(e))
			# Optionally restrict by requested detectors
			good = np.full(len(detinfo.channels), True)
			if dets   is not None: good &= np.isin(detinfo.channels, dets,   assume_unique=True)
			if detids is not None: good &= np.isin(detinfo.dets,     detids, assume_unique=True)
			aman = minisotodlib.AxisManager(minisotodlib.LabelAxis("dets", detinfo.channels[good]))
			aman.wrap("det_ids", detinfo.dets[good], [(0,"dets")])
			aman.wrap("det_pix", detinfo.pix [good], [(0,"dets")])
			aman.wrap("bands",   detname2band(detinfo.dets[good]), [(0,"dets")])
		ndet_full = aman.dets.count
		# 2. Load the necessary info from det_cal
		with bench.mark("fm_detcal"):
			with h5py.File(dcalfile, "r") as hfile:
				det_cal = hfile[dcalgroup][()]
			daman = minisotodlib.AxisManager(minisotodlib.LabelAxis("dets",np.char.decode(det_cal["dets:readout_id"])))
			good  = np.full(daman.dets.count, True)
		# Apply bias step cuts
		good &= det_cal["bg"] >= 0
		good &= det_cal["r_tes"] > 0
		good &= (det_cal["r_frac"] >= 0.05) & (det_cal["r_frac"] <= 0.8)
		good &= np.isfinite(det_cal["s_i"])
		with bench.mark("fm_pW"):
			daman.wrap("phase_to_pW", det_cal["phase_to_pW"], [(0,"dets")])
			good &= np.isfinite(daman.phase_to_pW)
		with bench.mark("fm_tau"):
			daman.wrap("tau_eff", det_cal["tau_eff"], [(0,"dets")])
			good &= np.isfinite(daman.tau_eff)
		with bench.mark("fm_merge"):
			daman.restrict("dets", daman.dets.vals[good])
			aman.merge(daman)
		# Optional flatfield. Indexed by det_id, similar to focal plane
		if self.rcal_cache:
			det_ids, rcal = self.rcal_cache.get(subid)
			ainds, rinds  = utils.common_inds([aman.det_ids, det_ids])
			aman.restrict("dets", aman.dets.vals[ainds])
			aman.wrap("relcal", rcal[rinds], [(0,"dets")])
		# 3. Load the focal plane information
		with bench.mark("fm_fplane"):
			fp_info = self.fp_cache.get_by_subid(subid, self.det_cache)
			# Match detectors
			det_ids = np.char.decode(fp_info["dets:det_id"])
			ainds, finds = utils.common_inds([aman.det_ids, det_ids])
			# Set up the focal plane. We don't need valid values for dark detectors
			fp_info = fp_info[finds]
			focal_plane = np.array([fp_info["xi"], fp_info["eta"], fp_info["gamma"]]).T
			good    = np.all(np.isfinite(focal_plane),1) | (det_type == "DARK")
			aman.restrict("dets", aman.dets.vals[ainds[good]])
			aman.wrap("focal_plane", focal_plane[good], [(0,"dets")])
		# 4. Load what we need from the preprocess archive. There aren't that many
		# superflous detectors in these files, and I didn't find an efficient
		# way to read only those needed, so just read all of them and merge
		t1 = time.time()
		with PrepLoader(prepfile, prepgroup) as pl:
			if pl.samps[1] != 0: print("Nonzero pl offset for %s. Chance to investigate cuts alignment" % subid)
			paman = minisotodlib.AxisManager(
					minisotodlib.LabelAxis("dets", pl.dets),
					minisotodlib.OffsetAxis("samps", count=pl.samps[0], offset=pl.samps[1]))
			# A bit awkward to time the time taken in the initialization
			bench.add("fm_prep_loader", time.time()-t1)

			# Need a better way to determine if hwp should be present or not
			has_hwp = "hwp_angle" in pl.group
			if has_hwp:
				with bench.mark("fm_hwp_angle"):
					paman.wrap("hwp_angle", pl.read("hwp_angle/hwp_angle","s"), [(0,"samps")])
				with bench.mark("fm_hwpss"):
					# These have order sin(1a),cos(1a),sin(2a),cos(2a),...
					paman.wrap("hwpss_coeffs", pl.read("pre_hwpss_stats/coeffs","d"), [(0,"dets")])
			with bench.mark("fm_cuts"):
				optional = config.get("cuts_optional")
				glitches = "glitches_pre_hwpss" if has_hwp else "glitches"
				paman.wrap("cuts_glitch", read_cuts(pl, glitches+"/glitch_flags",optional=optional), [(0,"dets"),(1,"samps")])
				paman.wrap("cuts_smurf",  read_cuts(pl, "smurfgaps/smurfgaps",  optional=optional), [(0,"dets"),(1,"samps")])
				paman.wrap("jumps_2pi",   read_cuts(pl, "jumps_2pi/jump_flag",  optional=optional), [(0,"dets"),(1,"samps")])
				paman.wrap("jumps_slow",  read_cuts(pl, "jumps_slow/jump_flag", optional=optional), [(0,"dets"),(1,"samps")])
				#paman.wrap("cuts_turn",   read_cuts(pl, "turnaround_flags/turnarounds", optional=optional), [(0,"dets"),(1,"samps")])
			try:
				good = np.diff(np.concatenate([[0],pl.read("valid_data/ends")])) > 0
			except KeyError:
				good = np.diff(np.concatenate([[0],pl.read("valid_data/valid_data/ends")])) > 0
			paman.restrict("dets", paman.dets.vals[good])

		# Eventually we will need to be able to read in sample-ranges.
		# Easiest to handle that with paman.restrict("samps", ...) here.
		# The rest should follow automatically

		with bench.mark("fm_merge2"):
			aman.merge(paman)

		# 4. Get stuff from the data file header
		with bench.mark("fm_detsets"):
			detset = self.det_cache.get_dets(subid).detset
		with bench.mark("fm_getfiles"):
			finfos = self.obsfiledb.get_files(obsid)[detset]
		# Get the filter params
		with bench.mark("fm_status"):
			#status = read_wiring_status(finfos[0][0])
			status = self.wiring_cache.get(finfos[0][0])
		with bench.mark("fm_iir"):
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
		with bench.mark("fm_abscal"):
			stream_id = "_".join(detset.split("_")[:2])
			abscal_cmb = self.acal_cache.get_by_subid(subid, stream_id=stream_id).abscal_cmb
		with bench.mark("pointing_model"):
			pointing_model = self.pointing_model_cache.get_by_subid(subid)
		# Get our sensitivity limits
		sens_lim = socommon.sens_limits[band]
		# Return our results. We don't put everything in an axismanager
		# because that has significant overhead, and we don't need an
		# axismanager for things that don't have any axes
		return bunch.Bunch(aman=aman, iir_params=iir_params, finfos=finfos,
			dac_to_phase = np.pi/2**15, timestamp_to_ctime=1e-8,
			abscal_cmb = abscal_cmb, pointing_model=pointing_model,
			sens_lim=sens_lim, ndet_full=ndet_full)

# This doesn't really belong here, unless we rename the module
def fast_data(finfos, detax, sampax, alloc=None, fields=[
		("signal",    "signal/data"),
		("timestamps","signal/times"),
		("az",        "ancil/az_enc"),
		("el",        "ancil/el_enc"),
		# For SAT we read in the boresight rotation.
		# For LAT we read in the corotator angle.
		# These will be transformed to the roll elsewhere
		("brot",      "ancil/boresight_enc", "?"),
		("corot",     "ancil/corotator_enc", "?")]):
	import fast_g3
	if alloc is None: alloc = fast_g3.DummyAlloc()
	# Add "!", for mandatory field, to any field that doesn't have
	# the third entry present
	def field_pad(f): return (f[0],f[1],"!") if len(f)==2 else f
	fields = [field_pad(f) for f in fields]
	aman   = minisotodlib.AxisManager(detax, sampax)
	fnames = [finfo[0]          for finfo in finfos]
	nsamps = [finfo[2]-finfo[1] for finfo in finfos]
	samps  = (sampax.offset, sampax.offset+sampax.count)
	i      = 0
	with fast_g3.open_multi(fnames, samps=samps, file_nsamps=nsamps) as ifile:
		fdets = ifile.fields["signal/data"].names
		rows  = utils.find(fdets, aman.dets.vals)
		active_fields = [f for f in fields if f[2]=="!" or f[1] in ifile.fields]
		for oname, iname, _ in active_fields:
			ifile.queue(iname, rows=rows)
		for fi, data in enumerate(ifile.read()):
			for oname, iname, _ in active_fields:
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

debug_det = None # "Mv21_f090_Ar00c02A"

# This config moved to loading.py because sofast is only imported conditionally
def calibrate(data, meta, mul=32, dev=None, prev_obs=None, dtype=np.float32, pool_map={}):
	from pixell import fft
	if dev   is None: dev   = device.get_device()
	# Look up our pools
	pool_tod, pool_ft = [dev.pools[name] for name in utils.vmap(pool_map, ["tod", "ft"])]
	# Merge the cuts and jumps separately. Easier to deal with just a single cuts object
	with bench.mark("merge_cuts"):
		cut_names  = [key for key in meta.aman.keys() if key.startswith("cuts_")]
		jump_names = [key for key in meta.aman.keys() if key.startswith("jumps_")]
		raw_cuts  = socut.Sampcut.merge([meta.aman[name] for name in cut_names])
		jumps     = socut.Sampcut.merge([meta.aman[name] for name in jump_names])
		# These are what will be passed to the mapmaker in the end
		cuts   = socut.Sampcut.merge([raw_cuts, jumps])
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left")
	# Find when we're actually scanning
	i1, i2 = socommon.find_scanning(data.az)
	# Adjust to fourier-friendly length
	nsamp = fft.fft_len((i2-i1)//mul, factors=dev.lib.fft_factors)*mul
	i2    = i1+nsamp
	timestamps, signal, cuts, raw_cuts, jumps, az, el = [a[...,i1:i2] for a in [data.timestamps,data.signal,cuts,raw_cuts,jumps,data.az,data.el]]
	hwp_angle = meta.aman.hwp_angle[i1:i2] if "hwp_angle" in meta.aman else None
	ninit = data.dets.count

	# prev_obs lets us pass in the result of calibrate run on
	# a different set of detectors for the same observation.
	# This is an optimization for load_multi, letting us avoid
	# repeating the same pointing correction multiple times
	if prev_obs is not None:
		ctime        = prev_obs.ctime
		el, az, roll = prev_obs.boresight
		bore_ref     = prev_obs.bore_ref
	else:
		with bench.mark("ctime"):
			ctime   = timestamps * meta.timestamp_to_ctime
		# Calibrate the pointing
		with bench.mark("boresight"):
			az      = az   * utils.degree
			el      = el   * utils.degree
			if "brot" in data:
				# SAT: boresight angle → roll
				roll = -data.brot [i1:i2]*utils.degree
			else:
				# LAT: corotator angle → roll
				roll = -data.corot[i1:i2]*utils.degree + el - 60*utils.degree
		bore_ref = np.array([el[0], az[0], roll[0]])
		with bench.mark("pointing correction"):
			fp = meta.aman.focal_plane
			az, el, roll, fp[:] = apply_pointing_model(az, el, roll, fp, meta.pointing_model)

	# Do we need to deslope at float64 before it is safe to drop to float32?
	with bench.mark("signal → gpu", tfun=dev.time):
		signal_ = pool_ft.array(signal)
	with bench.mark("signal → dtype", tfun=dev.time):
		signal  = pool_tod.empty(signal.shape, dtype)
		signal[:] = signal_

	if debug_det:
		i = utils.find(meta.aman.det_ids, debug_det)
		print(i)
		moo = bunch.Bunch(tod=dev.get(signal[i]), dets=meta.aman.det_ids[i],
			ctime=ctime, az=az, el=el, hwp=hwp_angle, abscal=meta.abscal_cmb,
			phase_to_pW=meta.aman.phase_to_pW[i], relcal=meta.aman.relcal[i],
			dac_to_phase=meta.dac_to_phase)
		bunch.write("test_raw.hdf", moo)

	with bench.mark("calibrate", tfun=dev.time):
		# Calibrate to CMB µK
		phase_to_cmb = 1e6 * meta.abscal_cmb * meta.aman.phase_to_pW[:,None]
		if "relcal" in meta.aman: phase_to_cmb *= meta.aman.relcal[:,None]
		signal *= dev.np.array(meta.dac_to_phase * phase_to_cmb)

	if debug_det: bunch.write("test_cal.hdf", bunch.Bunch(tod=dev.get(signal[i])))

	# Subtract the HWP scan-synchronous signal. We do this before deglitching because
	# it's large and doesn't follow the simple slopes and offsets we assume there
	if hwp_angle is not None:
		with bench.mark("subtract_hwpss", tfun=dev.time):
			nmode = 16
			signal = subtract_hwpss(signal, hwp_angle, meta.aman.hwpss_coeffs[:,:nmode]*phase_to_cmb, dev=dev)

	# Deglitch and dejump
	w = 10
	with bench.mark("deglitch", tfun=dev.time):
		# A bit dangerous to do these one by one. What happens if there's a cut
		# right at the edge of a jump? The dejumping would fail. Hopefully this
		# is rare
		jumps.dejump(signal, w=w, dev=dev)
		raw_cuts.gapfill(signal, w=w, dev=dev)

	if debug_det: bunch.write("test_deglitch.hdf", bunch.Bunch(tod=dev.get(signal[i])))

	with bench.mark("deslope", tfun=dev.time):
		with pool_ft.as_allocator():
			gutils.deslope(signal, w=w, dev=dev, inplace=True)

	if debug_det: bunch.write("test_deslope1.hdf", bunch.Bunch(tod=dev.get(signal[i])))

	with bench.mark("fft", tfun=dev.time):
		ftod = pool_ft.zeros((signal.shape[0],signal.shape[1]//2+1), utils.complex_dtype(signal.dtype))
		dev.lib.rfft(signal, ftod)
		norm = 1/signal.shape[1]

	# Deconvolve iir and time constants
	with bench.mark("iir_filter", tfun=dev.time):
		dt    = (ctime[-1]-ctime[0])/(ctime.size-1)
		freqs = dev.np.fft.rfftfreq(nsamp, dt).astype(signal.dtype)
		z     = dev.np.exp(-2j*np.pi*meta.iir_params.fscale*freqs)
		A     = dev.np.polyval(dev.np.array(meta.iir_params.a[:meta.iir_params.order+1][::-1]), z)
		B     = dev.np.polyval(dev.np.array(meta.iir_params.b[:meta.iir_params.order+1][::-1]), z)
		iir_filter = A/B # will multiply by this
		iir_filter *= norm # Hack: cheap to handle normalization here
		ftod *= iir_filter
	with bench.mark("time consts", tfun=dev.time):
		# I can't find an efficient way to do this. BLAS can't
		# do it since it's a triple multiplication. Hopefully the
		# gpu won't have trouble with it
		with pool_tod.as_allocator(): # tod buffer not in use atm
			#ftod *= 1 + 2j*np.pi*dev.np.array(meta.aman.tau_eff[:,None])*freqs
			# Writing it this way saves some memroy
			tfact = dev.np.full(ftod.shape, 2j*np.pi, ftod.dtype)
			tfact *= dev.np.array(meta.aman.tau_eff)[:,None]
			tfact *= freqs[None,:]
			tfact += 1
			ftod  *= tfact
			del tfact
	# Back to real space
	with bench.mark("ifft", tfun=dev.time):
		dev.lib.irfft(ftod, signal)

	if debug_det: bunch.write("test_iir_tconst.hdf", bunch.Bunch(tod=dev.get(signal[i])))

	# Sanity checks
	with bench.mark("measure noise", tfun=dev.time):
		rms = socommon.measure_rms_der(signal, dt=dt)
	with bench.mark("final detector prune", tfun=dev.time):
		good    = socommon.sensitivity_cut(rms, meta.sens_lim)
		nrms    = dev.np.sum(good)
		# Cut detectors with too big a fraction of samples cut,
		# or cuts occuring too often.
		cutfrac = cuts.sum()/cuts.nsamp
		cutdens = (cuts.bins[:,1]-cuts.bins[:,0])/cuts.nsamp
		good   &= dev.np.array((cutfrac < 0.1)&(cutdens < 1e-3))
		ndens   = dev.np.sum(good)
		# Cut all detectors if too large a fraction is cut
		good   &= dev.np.sum(good)/meta.ndet_full > 0.25
		nfinal  = dev.np.sum(good)
		# Prune signal and make it contiguous
		# Annoying to have to copy twice to get it back into the right buffer
		signal  = pool_ft.array(signal[good])
		signal  = pool_tod.array(signal)
		good   = dev.get(good) # cuts, dets, fplane etc. need this on the cpu
		cuts   = cuts  [good]
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left after sanity cuts: raw %d meta %d rms %d cutdens %d overcut %d" % (meta.ndet_full, meta.aman.dets.count, nrms, ndens, nfinal))

	# Sogma uses the cut format [{dets,starts,lens},:]. Translate to this
	with bench.mark("cuts reformat"):
		ocuts = cuts.to_simple()

	# Our goal is to output what sogma needs. Sogma works on these fields:
	res  = bunch.Bunch()
	res.sampoff      = i1
	res.dets         = meta.aman.dets.vals[good]
	res.detids       = meta.aman.det_ids[good]
	res.detpix       = meta.aman.det_pix[good]
	res.bands        = meta.aman.bands[good]
	res.point_offset = meta.aman.focal_plane[good,1::-1]
	res.polangle     = meta.aman.focal_plane[good,2]
	res.ctime        = ctime
	res.boresight    = np.array([el,az,roll])
	res.hwp          = hwp_angle
	res.tod          = signal
	res.cuts         = ocuts
	res.fill         = ocuts
	res.site         = "so"
	res.response     = None
	# original value of the first sample of boresight, before the pointing model
	# preserves the original winding, e.g.. el > 90, az > 360, etc, so can be used
	# to restore this information where it is necessary
	res.bore_ref     = bore_ref # [el0,az0,roll0]
	# Test per-detector response
	#res.response     = dev.np.zeros((2,len(res.tod)),res.tod.dtype)
	#res.response[0]  = 2
	#res.response[1]  = -1
	res.cutinfo = bunch.Bunch(
		ndet_init=ninit, ndet_rms=nrms, ndet_dens=ndens, ndet_final=nfinal)
	return res

#################
# Helpers below #
#################

# smurf: detset → channels, detset → wafer slot
# assignment: channels → detectors
# obsfiledb: obsid → detset
class SmurfInfo:
	"""Provides mapping from detset to wafer slot and channels (readout_ids)"""
	def __init__(self, smurf_index):
		self.smurf_index = smurf_index
		self.wslot    = {}
		self.channels = {}
	def get(self, detset):
		self._prepare(detset)
		return bunch.Bunch(
			wslot    = self.wslot[detset],
			channels = self.channels[detset],
		)
	def _prepare(self, detset, force=False):
		if detset in self.wslot and not force: return
		hfname, group = get_smurffile(self.smurf_index, detset)
		# Consider reading all the groups at once here
		with h5py.File(hfname, "r") as hfile:
			data = hfile[group][()]
			self.channels[detset] = np.char.decode(data["dets:readout_id"])
			self.wslot   [detset] = data["dets:wafer_slot"][0].decode()

class AssignmentInfo:
	"""Provides mapping from channel to detector for each detset"""
	def __init__(self, match_index):
		self.match_index = match_index
		self.channels    = {}
		self.dets        = {}
	def get(self, detset):
		self._prepare(detset)
		return bunch.Bunch(
			channels = self.channels[detset],
			dets     = self.dets[detset],
		)
	def _prepare(self, detset, force=False):
		if detset in self.dets and not force: return
		hfname, group = get_matchfile(self.match_index, detset)
		with h5py.File(hfname, "r") as hfile:
			data = hfile[group][()]
			self.channels[detset] = np.char.decode(data["dets:readout_id"])
			self.dets    [detset] = np.char.decode(data["dets:det_id"])

class WaferInfo:
	"""Provides information about detector types"""
	def __init__(self, wafer_index):
		self.wafer_index = wafer_index
		self.info = {}
	def get(self, wafer_name):
		self._prepare(wafer_name)
		return self.info[wafer_name]
	def _prepare(self, wafer_name):
		if wafer_name in self.info: return
		hfname, group = get_wafer_file(self.wafer_index, wafer_name)
		def mget(data, name, default):
			try: return np.char.decode(data[name])
			except (TypeError,ValueError): return np.full(len(data),default)
		with h5py.File(hfname, "r") as hfile:
			data = hfile[group][()]
			# Sadly some of the are only sometimes present.
			# LF doesn't have rhombus. Made the ones we don't
			# absolutely need optional
			self.info[wafer_name] = bunch.Bunch(
				dets  = np.char.decode(data["dets:det_id"]),
				bands = np.char.decode(data["dets:wafer.bandpass"]),
				types = mget(data,"dets:wafer.type", "?"),
				array = mget(data,"dets:wafer.array", "?"),
				rhombus = mget(data, "dets:wafer.rhombus", "?"),
				row   = mget(data, "dets:wafer.det_row", 0),
				col   = mget(data, "dets:wafer.det_col", 0),
				raw   = data)

# Should extend DetCache to also include wafer_info. wafer_info
# is what officially lets us get the detector bandpass and type.
# One can try to infer these from the detector name (e.g.
# Mv12_f090_Cr00c06A would be an f090 detector and Mv21_DARK_Mp13b55D
# would be a dark detector. But apparently this isn't reliable.
#
# wafer_info has a list of the detector names (not channels!),
# band and type for each wafer name. Dark detectors are actually
# associated with a band, which is needed for abscal. This information
# is lost if one just uses the second field of the detector name.
# On ther other hand, abscal for dark detectors is pretty meaningless...
#
# It's tempting to just ignore this for now, and instead hack the
# handling of DARK in abscal. Preprocess archive also needs the
# band, though, and that's where we get our cuts from. With a hack,
# we would need to translate DARK to another band, and hope that
# captures all the dark detectors. One would also need to know
# which band to replace DARK by for each wafer/tube, which isn't
# obvious. Probably best to do this properly.
#
# If so, darkness is separate from band.
# 1. Extend subids to obs:slot:band:type, with type defaulting to optc.
#    Would then use type for special casing in FastMeta. Probably the
#    best approach.

class DetCache:
	def __init__(self, obsfiledb, smurf_info, ass_info, wafer_info):
		self.obsfiledb  = obsfiledb
		self.smurf_info = smurf_info
		self.ass_info   = ass_info
		self.wafer_info = wafer_info
		self.det_cache    = {}
		self.detset_cache = {}
		self.done      = set()
	def get_dets(self, subid):
		toks = split_subid(subid)
		self._prepare(toks.obsid)
		return self.det_cache[toks.subid]
	def get_detsets(self, obsid):
		self._prepare(obsid)
		return self.detset_cache[obsid]
	def _prepare(self, obsid, force=False):
		"""Read in the det lists for obsid, if necessary."""
		if obsid in self.done and not force: return
		detsets = self.obsfiledb.get_detsets(obsid)
		if len(detsets) == 0:
			raise utils.DataMissing("No detsets for %s in obsfiledb" % obsid)
		self.detset_cache[obsid] = {}
		for dset in detsets:
			wafer_name = detset2wafer_name(dset)
			try:
				sinfo = self.smurf_info.get(dset)
				ainfo = self.ass_info  .get(dset)
				winfo = self.wafer_info.get(wafer_name)
			except KeyError:
				print("warning: dset %s missing" % dset)
				# Some wafers may be missing
				continue
			# Match ainfo to winfo
			ainds, winds = utils.common_inds([ainfo.dets, winfo.dets])
			# Group by band:type
			band_type = np.char.add(np.char.add(winfo.bands[winds],":"),winfo.types[winds])
			# Build a pixel id. This is the same for all detectors that are co-located
			# in the focal plane, and is usedful for pair differencing etc
			pixid = build_pixid(winfo.array[winds], winfo.rhombus[winds], winfo.row[winds], winfo.col[winds])
			bts, order, edges = utils.find_equal_groups_fast(band_type)
			for bti, bt in enumerate(bts):
				band, type = bt.split(":")
				subid = "%s:%s:%s:%s" % (obsid, sinfo.wslot, band, type)
				ginds = order[edges[bti]:edges[bti+1]]
				inds  = ainds[ginds]
				self.det_cache[subid] = bunch.Bunch(
						channels = ainfo.channels[inds],
						dets     = ainfo.dets[inds],
						detset   = dset,
						band     = band,
						type     = type,
						pix      = pixid[ginds],
				)
			self.detset_cache[obsid][sinfo.wslot] = dset
		self.done.add(obsid)

def build_pixid(array, rhombus, row, col):
	# Annoying that numpy.char doesn't provide any real vectorized
	# operations for this, only pretend ones
	return np.array(["%s_%sr%02dc%02d" % (a,R,r,c) for a,R,r,c in zip(array, rhombus, row, col)])

class Subid:
	def __init__(self, obsid, wslot, band, type="OPTC"):
		self.obsid, self.wslot, self.band, self.type = obsid, wslot, band, type
		self.subid = ":".join(self)
		self.ctime = obsid2ctime(self.obsid)
	def __len__(self): return 4
	def __iter__(self):
		yield self.obsid
		yield self.wslot
		yield self.band
		yield self.type

def split_subid(subid): return Subid(*subid.split(":"))
def obsid2ctime(obsid): return float(obsid.split("_")[1])

def detset2wafer_name(detset): return "_".join(detset.split("_")[:2])

def trange_fmt(sfile, query_fmt):
	if "obs:timestamp__lo" in sfile.columns("map"):
		t1 = "[obs:timestamp__lo]"
		t2 = "[obs:timestamp__hi]"
	else:
		t1 = "-1e9999"
		t2 = "+1e9999"
	return query_fmt.format(t1=t1, t2=t2)

class FplaneCache:
	def __init__(self, fname):
		# The index file here is small, with efficient time
		# ranges, so we can just read in the whole thing
		self.fname = fname
		self.index = {}
		with sqlite.open(fname) as sfile:
			# This handles both the case when time ranges are present and when they aren't
			query = trange_fmt(sfile, "select [dets:stream_id], {t1}, {t2}, files.name, dataset from map inner join files on file_id = files.id")
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
			fname = os.path.join(os.path.dirname(self.fname),entry[2])
			with h5py.File(fname,"r") as hfile:
				self.fp_cache[key] = hfile[entry[3]][()]
		return self.fp_cache[key]
	def get_by_subid(self, subid, det_cache):
		"""returns array with [('dets:det_id', 'S18'), ('xi', '<f4'), ('eta', '<f4'), ('gamma', '<f4')]"""
		toks = split_subid(subid)
		ctime      = float(toks.obsid.split("_")[1])
		wafer_name = detset2wafer_name(det_cache.get_detsets(toks.obsid)[toks.wslot])
		return self.get_by_wafer(wafer_name, ctime)

# How to handle per-obs pointing model?
# Storage:
#  1. One dataset per entry, like current setup. Wasteful
#  2. One big table. Could grow pretty big. But I'd probably cache it all anyway..
#  3. Table per time-range. Probably best approach. Lets us keep the same sqlite
#     format as currently. Only the reading code would need to change.
# Table format would be
#  subid params...
# which can be stored as a structured array
#
# We only have data for deep56, which is observed for a fraction of each day.
# Should there be one entry in the sqlite per day, with each entry pointing to the
# same table? No, unnecessary complication. There just won't be any entries in the
# table for the missing data.

class PointingModelCache:
	def __init__(self, fname):
		self.fname = fname
		self.cache = {}
		with sqlite.open(fname) as sfile:
			query = trange_fmt(sfile, "select {t1}, {t2}, files.name, dataset from map inner join files on file_id = files.id")
			# Get the mapping from time-range to hdf-file and dataset
			self.index = list(sfile.execute(query))
	def find_entry(self, t):
		for i, entry in enumerate(self.index):
			if entry[0] <= t and t < entry[1]:
				return entry
		# Return latest entry by default
		return entry
	def get_by_time(self, t):
		entry = self.find_entry(t)
		hfname, gname = entry[2:4]
		key   = (hfname, gname)
		if key not in self.cache:
			fname = os.path.join(os.path.dirname(self.fname),hfname)
			with h5py.File(fname, "r") as hfile:
				# We have two formats:
				# 1. Saianeesh's static pointing model, where the parameters
				#    are in json.loads(hfile[gname].attrs["_scalars"]),
				#    which apply to all entries that map to this time-range
				# 2. My per-obs pointing model, where hfile[gname]
				#    is a numpy structured array that needs to be indexed by
				#    subid
				d = hfile[gname]
				if isinstance(d, h5py.Group):
					# Saianeesh static format
					param_str = hfile[gname].attrs["_scalars"]
					params    = bunch.Bunch(**json.loads(param_str))
					self.cache[key] = ("static", params)
				elif isinstance(d, h5py.Dataset):
					# Sigurd per-subid format. Precompute a hash map
					table = d[()]
					index = {subid.decode():i for i,subid in enumerate(table["subid"])}
					self.cache[key] = ("subid", index, table)
				else: raise ValueError("%s/%s is neither a group or dataset" % (fname, gname))
		return self.cache[key]
	def get_by_subid(self, subid):
		toks  = split_subid(subid)
		ctime = float(toks.obsid.split("_")[1])
		# Take into account the two formats
		match = self.get_by_time(ctime)
		if   match[0] == "static": return match[1]
		elif match[0] == "subid":
			# Look up row in table, and reformat it as bunch
			#print("%s in pointing: %d" % (str(subid), str(subid) in match[1]))
			try: row = match[2][match[1][str(subid)]]
			except KeyError: raise errors.DataMissing("Missing pointing correction")
			return pointing_row_to_params(row)
		else:
			raise ValueError("Unrecognized PointingModelCache entry '%s'" % str(match[0]))

def pointing_row_to_params(row):
	params = bunch.Bunch()
	for name in row.dtype.names[1:]:
		val = row[name]
		if isinstance(val, bytes):
			val = val.decode()
		params[name] = val
	return params

class EpochDb:
	def __init__(self, fname):
		self.fname = fname
		with sqlite.open(fname) as sfile:
			query = trange_fmt(sfile, "select {t1}, {t2}, files.name, dataset from map inner join files on file_id = files.id")
			# Get the mapping from time-range to hdf-file and dataset.
			# Also make relative paths work
			self.index = [(t1, t2, os.path.join(os.path.dirname(fname), fn), group) for t1,t2,fn,group in sfile.execute(query)]
	def lookup(self, t):
		for i, entry in enumerate(self.index):
			if entry[0] <= t and t < entry[1]:
				return entry[2:]
		# Return latest entry by default
		return entry[2:]

class RelcalCache:
	def __init__(self, fname):
		self.epochs = EpochDb(fname)
		self.cache  = {}
	def get(self, subid):
		# Infer the time from the subid
		info  = split_subid(subid)
		fname, group = self.epochs.lookup(info.ctime)
		key = (fname, group)
		if key not in self.cache:
			with h5py.File(fname, "r") as hfile:
				data = hfile[group][()]
				self.cache[key] = (
					np.char.decode(data["dets:det_id"]),
					utils.getrec(data, ["relcal", "rel_factor"])
				)
		return self.cache[key]

class AcalCache:
	def __init__(self, fname):
		self.epochs = EpochDb(fname)
		self.cache  = {}
	def get(self, t, stream_id, wafer, band):
		fname, group = self.epochs.lookup(t)
		# Try to look up both alternatives in cache
		try: return self.cache[(fname, group, stream_id, band)]
		except KeyError: pass
		try: return self.cache[(fname, group, wafer, band)]
		except KeyError: pass
		# If we got here, the cache doesn't have it, so try to read it
		raw = bunch.read(fname)[group]
		for wafid in ["dets:stream_id", "dets:wafer_slot"]:
			if wafid not in raw.dtype.names: continue
			for row in raw:
				self.cache[(fname, group, row[wafid].decode(), row["dets:wafer.bandpass"].decode())] = bunch.Bunch(
					abscal_cmb = row["abscal_cmb"],
					abscal_rj  = row["abscal_rj"],
				)
		# Try to look up again
		try: return self.cache[(fname, group, stream_id, band)]
		except KeyError: pass
		try: return self.cache[(fname, group, wafer, band)]
		except KeyError: pass
		# Couldn't find it!
		raise errors.DataMissing("Couldn't find abscal for %.0f %s %s %s" % (t, stream_id, wafer, band))
	def get_by_subid(self, subid, stream_id):
		toks  = split_subid(subid)
		ctime = float(toks.obsid.split("_")[1])
		return self.get(ctime, stream_id, toks.wslot, toks.band)

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

def read_cuts(pl, path, optional=False):
	try:
		shape     = pl.read(path + "/shape")
		edges     = pl.read(path + "/ends")
		intervals = pl.read(path + "/intervals")
		return expand_cuts_sampcut(shape, edges, intervals, inds = pl.inds)
	except KeyError as e:
		if optional: return socut.Sampcut.empty(pl.ndet, pl.nsamp)
		else: raise errors.DataMissing("Missing cuts %s" % str(path))

def expand_cuts_sampcut(shape, ends, intervals, inds=None):
	if len(shape) != 2:
		raise ValueError("Expected (ndet,nsamp) RangesMatrix")
	bins = np.zeros((shape[0],2),np.int32)
	# Ends is an index into flattened ranges
	ends = ends//2
	bins[ :,1] = ends
	bins[1:,0] = ends[:-1]
	cuts = socut.Sampcut(bins, intervals.reshape(-1,2), shape[1])
	if inds is not None: cuts = cuts[inds]
	return cuts

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
	@property
	def ndet(self): return len(self.dets)
	@property
	def nsamp(self): return self.samps[0]
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

def get_prepfile(indexdb, subid):
	toks = split_subid(subid)
	# Inconsistent format here too
	if "dets:wafer.bandpass" in indexdb.columns("map"):
		query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id], [dets:wafer_slot], [dets:wafer.bandpass] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' AND [dets:wafer_slot] = '%s' AND [dets:wafer.bandpass] = '%s' LIMIT 1;" % (toks.obsid, toks.wslot, toks.band)
	else:
		query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id], [dets:wafer_slot] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' AND [dets:wafer_slot] = '%s' LIMIT 1;" % (toks.obsid, toks.wslot)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise errors.DataMissing("%s not found in preprocess index" % subid)
	return os.path.join(os.path.dirname(indexdb.fname),fname), gname

def get_dcalfile(indexdb, subid):
	toks = split_subid(subid)
	query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' LIMIT 1;" % (toks.obsid)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise errors.DataMissing("%s not found in det_cal index" % subid)
	return os.path.join(os.path.dirname(indexdb.fname),fname), gname

def get_matchfile(indexdb, detset):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:detset] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:detset] = '%s' LIMIT 1;" % (detset)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise errors.DataMissing("%s not found in det match index" % detset)
	return os.path.join(os.path.dirname(indexdb.fname),fname), gname

def get_smurffile(indexdb, detset):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:detset] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:detset] = '%s' LIMIT 1;" % (detset)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise errors.DataMissing("%s not found in smurf index" % subid)
	return os.path.join(os.path.dirname(indexdb.fname),fname), gname

def get_wafer_file(waferdb, wafer_name):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:stream_id] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:stream_id] = '%s' LIMIT 1;" % (wafer_name)
	try: fname, gname = next(waferdb.execute(query))[:2]
	except StopIteration: raise errors.DataMissing("%s not found in wafer index" % wafer_name)
	return os.path.join(os.path.dirname(waferdb.fname),fname), gname

def get_acalfile(indexdb):
	return os.path.join(os.path.dirname(indexdb.fname),list(indexdb.execute("SELECT name FROM files INNER JOIN map ON files.id = map.file_id WHERE map.dataset = 'abscal'"))[0][0])

def read_wiring_status(fname, required=[]):
	import fast_g3
	for frame in fast_g3.get_header_frames(fname)["frames"]:
		if frame["type"] == "wiring":
			status = frame["fields"]["status"]
			status = yaml.safe_load(status)
			if all([name in status for name in required]):
				return status
	return status

class WiringCache:
	def __init__(self, fields=None):
		self.cache  = {}
		self.fields = fields
	def get(self, fname):
		if fname not in self.cache:
			status = read_wiring_status(fname, required=self.fields)
			if self.fields is not None:
				status = {field:status[field] for field in self.fields}
			self.cache[fname] = status
		return self.cache[fname]

def subtract_hwpss(signal, hwp_angle, coeffs, dev=None):
	if signal.dtype != np.float32: raise ValueError("Only float32 supported")
	dev       = dev or device.get_device()
	hwp_angle = dev.np.asarray(hwp_angle, dtype=signal.dtype)
	coeffs    = dev.np.asarray(coeffs,    dtype=signal.dtype)
	ncoeff    = coeffs.shape[1]
	B         = dev.np.zeros((ncoeff,len(hwp_angle)),signal.dtype)
	# This can be done with recursion formulas, but
	# the gains are only few ms on the cpu. Let's keep this
	# for the time being
	with bench.mark("build basis"):
		for i in range(ncoeff):
			mode = i//2+1
			fun  = [dev.np.sin, dev.np.cos][i&1]
			B[i] = fun(mode*hwp_angle)
	# We want signal -= coeffs.dot(B): [ndet,n]*[n,nsamp]. Fortran is
	# column-major though, so it wants [nsamp,n]*[n,ndet]
	ndet, nsamp = signal.shape
	dev.lib.gemm("N", "N", nsamp, ndet, ncoeff, -1, B, nsamp, coeffs, ncoeff, 1, signal, nsamp)
	return signal

def polar_2d(x, y):
	r = (x**2+y**2)**0.5
	φ = np.arctan2(y,x)
	return r, φ

# This takes 250 ms! And the quaternion stuff would be tedious
# (but not difficult as such) to implement on the gpu
def apply_pointing_model(az, el, roll, detoffs, model):
	"""Apply the given pointing model to az, el, roll, returning corrected values.
	The coordinates will also be normalized to -pi/2 <= el <= pi/2, see fix_overpole"""
	import pixell.coordsys as quat
	# Ensure they're arrays, and avoid overwriting. These are
	# small arrays anyways
	[az, el, roll, detoffs] = [np.array(a) for a in [az,el,roll, detoffs]]
	if   model.version == "sat_naive":
		if el[0] > np.pi/2: fix_overpole(az, el, roll, inline=True)
	elif model.version == "sat_v1":
		# Remember, roll = -boresight_angle. That's why there's a minus below
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
	elif model.version == "lat_naive": pass
	elif model.version == "lat_v0":
		# Reconstruct the corotator angle
		corot = -roll + el - 60*utils.degree
		# Apply offsets
		az    += model.az_offset * utils.degree
		el    += model.el_offset * utils.degree
		corot += model.cr_offset * utils.degree
		q_enc     = quat.rotation_lonlat(-az, el)
		q_mir     = quat.rotation_xieta(model.mir_xi_offset * utils.degree, model.mir_eta_offset * utils.degree)
		q_tel     = quat.rotation_xieta(model.el_xi_offset  * utils.degree, model.el_eta_offset  * utils.degree)
		q_rx      = quat.rotation_xieta(model.rx_xi_offset  * utils.degree, model.rx_eta_offset  * utils.degree)
		q_el_roll = quat.euler(2, el - 60*utils.degree)
		q_cr_roll = quat.euler(2, -corot)
		q_tot     = q_enc * q_mir * q_el_roll * q_tel * q_cr_roll * q_rx
		az, el, roll = quat.decompose_lonlat(q_tot)
		az       *= -1
	# This has a lot in common with v1 implementation-wise, but I keep it separate
	# to keep them independent. Could factor out some code if we get too many of these though
	elif model.version == "lat_v1":
		# Reconstruct the corotator angle
		corot = el - roll - 60*utils.degree
		# Apply offsets
		az    += model.enc_offset_az
		el    += model.enc_offset_el
		corot += model.enc_offset_cr
		# Main part
		q_lonlat     = quat.rotation_lonlat(-az, el)
		q_mir_center = ~quat.rotation_xieta(model.mir_center_xi0, model.mir_center_eta0)
		q_el_roll    = quat.euler(2, el - 60*utils.degree)
		q_el_axis_center = ~quat.rotation_xieta(model.el_axis_center_xi0, model.el_axis_center_eta0)
		q_cr_roll    = quat.euler(2, -corot)
		q_cr_center  = ~quat.rotation_xieta(model.cr_center_xi0, model.cr_center_eta0)
		q_presag     = q_lonlat * q_mir_center * q_el_roll * q_el_axis_center * q_cr_roll * q_cr_center
		# Back to angles for the el sag. We also save the corotator angle for later (see bottom)
		maz, el, roll = quat.decompose_lonlat(q_presag)
		# Now apply the el sag
		Δel = el     - model.el_sag_pivot
		el += Δel    * model.el_sag_lin
		el += Δel**2 * model.el_sag_quad
		# Back to quaternions for the final base tilt part
		q_postsag = quat.rotation_lonlat(maz, el, roll)
		# Base tilt is a bit more complicated then the others
		phi = np.arctan2(model.base_tilt_sin, model.base_tilt_cos)
		amp = (model.base_tilt_sin**2 + model.base_tilt_cos**2)**0.5
		q_base = quat.euler(2,phi) * quat.euler(1, amp) * quat.euler(2, -phi)
		# Compose into the full model
		q_tot  = q_base * q_postsag
		# Extract the final coordinates.
		az, el, roll = quat.decompose_lonlat(q_tot)
		az          *= -1
	elif model.version == "lat_v2":
		# Apply any wafer offsets if applicable
		if "waf_off_xi" in model:
			detoffs[:,0] += model.waf_off_xi
			detoffs[:,1] += model.waf_off_eta
		# Reconstruct the corotator angle
		corot = el - roll - 60*utils.degree
		# Apply offsets
		az    += model.enc_offset_az
		el    += model.enc_offset_el
		# El sag. Should the quadratic term preserve the sign?
		Δel = el     - model.el_sag_pivot
		el += Δel    * model.el_sag_lin
		el += Δel**2 * model.el_sag_quad
		# Main part
		q_lonlat     = quat.rotation_lonlat(-az, el)
		q_mir_center = ~quat.rotation_xieta(model.mir_center_xi0, model.mir_center_eta0)
		q_el_roll    = quat.euler(2, el - 60*utils.degree)
		q_el_axis_center = ~quat.rotation_xieta(model.el_axis_center_xi0, model.el_axis_center_eta0)
		q_cr_roll    = quat.euler(2, -corot)
		q_cr_center  = ~quat.rotation_xieta(model.cr_center_xi0, model.cr_center_eta0)
		# Base tilt is a bit more complicated
		phi = np.arctan2(model.base_tilt_sin, model.base_tilt_cos)
		amp = (model.base_tilt_sin**2 + model.base_tilt_cos**2)**0.5
		q_base = quat.euler(2,phi) * quat.euler(1, amp) * quat.euler(2, -phi)
		# Compose into the full model
		q_tot  = q_base * q_lonlat * q_mir_center * q_el_roll * q_el_axis_center * q_cr_roll * q_cr_center
		# Finally back to coordinates
		az, el, roll = quat.decompose_lonlat(q_tot)
		az          *= -1
	elif model.version == "arc":
		# Calcluate the rad-roll-arc offsets based on the position in the focal plane
		r2    = detoffs[:,0]**2 + detoffs[:,1]**2
		scale = model.arc_amp * (1-r2/model.arc_r0**2)
		ang   = np.mean(roll)-model.arc_roll0
		dxi   = scale * (np.cos(ang)-1)
		deta  = scale * np.sin(ang)
		# Apply this and the wafer offset
		detoffs[:,0] += dxi  + model.waf_off_xi
		detoffs[:,1] += deta + model.waf_off_eta
		# The rest is like the v2 model
		# Reconstruct the corotator angle
		corot = el - roll - 60*utils.degree
		# Apply offsets
		az    += model.enc_offset_az
		el    += model.enc_offset_el
		corot += model.enc_offset_cr
		# El sag. Should the quadratic term preserve the sign?
		Δel = el     - model.el_sag_pivot
		el += Δel    * model.el_sag_lin
		el += Δel**2 * model.el_sag_quad
		# Main part
		q_lonlat     = quat.rotation_lonlat(-az, el)
		q_mir_center = ~quat.rotation_xieta(model.mir_center_xi0, model.mir_center_eta0)
		q_el_roll    = quat.euler(2, el - 60*utils.degree)
		q_el_axis_center = ~quat.rotation_xieta(model.el_axis_center_xi0, model.el_axis_center_eta0)
		q_cr_roll    = quat.euler(2, -corot)
		q_cr_center  = ~quat.rotation_xieta(model.cr_center_xi0, model.cr_center_eta0)
		# Base tilt is a bit more complicated
		phi = np.arctan2(model.base_tilt_sin, model.base_tilt_cos)
		amp = (model.base_tilt_sin**2 + model.base_tilt_cos**2)**0.5
		q_base = quat.euler(2,phi) * quat.euler(1, amp) * quat.euler(2, -phi)
		# Compose into the full model
		q_tot  = q_base * q_lonlat * q_mir_center * q_el_roll * q_el_axis_center * q_cr_roll * q_cr_center
		# Finally back to coordinates
		az, el, roll = quat.decompose_lonlat(q_tot)
		az          *= -1
	else: raise ValueError("Unrecognized model '%s'" % str(model.version))
	return az, el, roll, detoffs

def fix_overpole(az, el, roll, inline=False):
	if not inline: az, el, roll = az.copy(), el.copy(), roll.copy()
	# el is simple, since it doesn't have a wrapping ambiguity.
	el[:]  = np.pi-el
	# az and roll will be [-pi,pi] after decompose_lonlat. They will
	# be [0,2pi] after this. The pointing code shouldn't care either way
	az   += np.pi
	roll += np.pi
	return az, el, roll

def detname2band(detnames):
	return np.char.partition(np.char.partition(detnames, "_")[:,2],"_")[:,0]
