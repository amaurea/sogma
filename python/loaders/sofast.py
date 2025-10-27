import numpy as np, contextlib, json, time, os, scipy, re, yaml, ast, h5py
from pixell import utils, bunch, bench, sqlite, coordsys
from .. import device, gutils, socut
from . import socommon
from .socommon import cmeta_lookup

# We have a "sweeps" member of obsinfo, which encapsulates the
# area hit by each observation on the sky, in equatorial coordinates.
# This was motivated by wanting to select e.g. observations that hit some
# spot on the sky, which is useful. However, the ObsDb does not contain
# this information, and it's a bit expensive to get at it, especially the
# wafer center and wafer radius. I also haven't implemented the actual
# point-in-polygon lookup yet (though it's in ACT). Until ObsDb makes this
# simpler, sweeps will just assume 0.35 degree radus wafers centered at 0,0.
# This is not good enough for obs-hits-point calculation, but it's good enough
# for task distribution, which is the only thing it's currenlty used for

# Reading multiple wafers at the same time
# ----------------------------------------
# A) Call fast_meta.read for each, then merge them,
#    followed by a single fast_data call, and a single calibrate.
#    A problem with this is that calibrate assumes a common iir_params
#    for all detectors. Also fast_data assumes that it will get all
#    all detectors in one go. So this approach would requrie quite a few
#    changes.
# B) Call fast_meta.read for each, determining a common sample range.
#    Restrict each aman to that sample range.
#    Then loop over them, calling fast_data and calibrate on each.
#    Finally, merge them into one result.
#    This approach requres no changes to any of the read functions.
#    It should also use a bit less memory (fft arrays are smaller).
#    One problem is that each calibrate thinks it has ownership of
#    the "tod" buffer, so they would clobber each other. Can fix
#    this by allocating an output array first, and then copying over
#    subobs by subobs, instead of doing it all at the end. This
#    output buffer would ideally also be "tod" though, since it's
#    assumed that this is the buffer tod lives in later.
#    Implement a mempool.swap() operation to make this cheap?
#    Or add an option to calibrate to make it not use a predefined buffer
#    for the tod? This may be good in any case, as we don't need
#    the read-in buffer later.

# Fourier-truncation can lose a significant amount of samples, it seems
# Probably best to switch to fourier-padding instead. Will be some work though,
# since one needs to keep track of the logical length separately from the full
# length. Update: I tested this, but the speed loss was significant and not
# worth it. But in the process, I discovered that I had a typo in the fourier
# prime list, and after fixing that the amount of truncation went down a lot.

class SoFastLoader:
	def __init__(self, context_or_config_or_name, dev=None, mul=32):
		from sotodlib import core
		self.context   = socommon.get_expanded_context(context_or_config_or_name)
		self.obsdb     = core.metadata.ObsDb(self.context["obsdb"])
		self.predb     = sqlite.open(socommon.cmeta_lookup(self.context, "preprocess"))
		# Precompute the set of valid tags. Sadly this requires a scan through the whole
		# database, but it's not that slow so far
		self.tags   = socommon.get_tags(self.obsdb.conn)
		self.fast_meta = FastMeta(self.context)
		self.mul     = mul
		self.dev     = dev or device.get_device()
		self.catch_list = (Exception,)
		#self.catch_list = ()
	def query(self, query=None, sweeps=True, output="sogma"):
		res_db, pycode, slices = socommon.eval_query(self.obsdb.conn, query, tags=self.tags, predb=self.predb)
		return socommon.finish_query(res_db, pycode, slices, sweeps=sweeps, output=output)
	def load(self, subid, catch="expected"):
		catch_list = catch2list(catch)
		try:
			with bench.mark("load_meta"):
				meta = self.fast_meta.read(subid)
				if meta.aman.dets.count == 0:
					raise utils.DataMissing("no detectors left after meta: raw %d meta 0" % meta.ndet_full)
			# Load the raw data
			with bench.mark("load_data"):
				data = fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
			# Calibrate the data
			with bench.mark("load_calib"):
				obs = calibrate(data, meta, mul=self.mul, dev=self.dev)
		except catch_list as e:
			# FIXME: Make this less broad
			raise utils.DataMissing(type(e).__name__ + " " + str(e))
		# Add timing info
		obs.timing = [("meta",bench.t.load_meta),("data",bench.t.load_data),("calib",bench.t.load_calib)]
		# Record which obs it is. A bit useless for load, but very useful for load_multi
		obs.subids = [subid]
		# Record any non-fatal errors
		obs.errors = []
		return obs
	def load_multi(self, subids, order="band", samprange=None, catch="expected"):
		"""Load multiple concurrent subids into a single obs"""
		# FIXME: This is inefficient:
		# * bands and dark detectors for the same wafer are stored in the same files,
		#   which are read redundantly
		# * The first g3 file must always be read, even when we just want a higher samprange
		# Should group the subids by the ones that live in the same files, and issue a single
		# fast_data and calibrate for each group. This will require a reorder afterwards, to
		# get the bands in contiguous order. Reorder will require data copying. Easiest and
		# most flexible to just copy from one buffer to another. So instead of doing
		# write to ptbuf (single obs) → append to todbuf, like we do now, we would do
		# write to todbuf (obs-group) → append to sections in ptbuf → compact into todbuf.
		# -
		# With multi-band mapping we will need the bands to be contiguous in memory.
		# The other argument supports this
		if   order == "raw":  pass
		elif order == "band":
			def srtfun(subid):
				toks = split_subid(subid)
				return toks.band + ":" + toks.type
			subids = sorted(subids, key=srtfun)
		else: raise ValueError("Unrecognized subid ordering '%s'" % str(order))
		catch_list = catch2list(catch)
		# Read all the metadata
		metas,     mids = [], []
		exceptions,eids = [], []
		for si, subid in enumerate(subids):
			try:
				with bench.mark("load_meta"):
					meta = self.fast_meta.read(subid)
					if meta.aman.dets.count == 0: raise utils.DataMissing("no detectors left after meta: raw %d meta 0" % meta.ndet_full)
					metas.append(meta)
				mids .append(subid)
			except catch_list as e:
				exceptions.append(e)
				eids      .append(subid)
		if len(metas) == 0:
			raise utils.DataMissing(format_multi_exception(exceptions, eids))
		# Restrict them to a common sample range
		sinfo = get_obs_sampinfo(self.obsdb.conn, mids)
		ndet, nsamp = make_metas_compatible(metas, sinfo)
		# Restrict to target sample range
		if samprange is not None:
			for meta in metas:
				off = meta.aman.samps.offset
				meta.aman.restrict("samps", slice(samprange[0]+off,samprange[1]+off), in_place=True)
		# Set up total obs
		otot = bunch.Bunch(ctime=None, boresight=None, hwp=None, tod=None, subids=[], errors=[], cuts=[])
		append_fields = [("dets",0),("detids",0),("bands",0),("point_offset",0),
			("polangle",0),("response",1)]
		for field, axis in append_fields: otot[field] = []
		dcum = 0
		try:
			# We want the final output tod in the "tod" buffer, but this buffer will be
			# used in calibrate(). The big "pointing" buffer is free at this point, though,
			# so we can trick calibrate() into using that one instead by swapping the
			# "pointing" and "tod" pools. By swapping back in the end, the result will end
			# up where we want it. We also need to swap back in the end to keep our total
			# allocations low, since "pointing" is supposed to be a 3x as large buffer
			self.dev.pools.want("pointing", "tod")
			self.dev.pools.swap("pointing", "tod")
			for si, (subid, meta) in enumerate(zip(mids, metas)):
				try:
					# read in and calibrate each
					with bench.mark("load_data"):
						data = fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
					# This uses buffers "tod" and "ft"
					with bench.mark("load_calib"):
						obs  = calibrate(data, meta, mul=self.mul, dev=self.dev)
					obs.subids = [subid]
					obs.errors = []
				except catch_list as e:
					exceptions.append(e)
					eids      .append(subid)
					continue
				# Initial otot setup
				if otot.tod is None:
					otot.ctime     = obs.ctime
					otot.boresight = obs.boresight
					otot.hwp       = obs.hwp
					otot.tod       = self.dev.pools["pointing"].zeros((ndet,len(obs.ctime)), obs.tod.dtype)
				# Handle the simple append cases
				for field, axis in append_fields:
					otot[field].append(obs[field])
				otot.cuts.append(obs.cuts)
				otot.subids += obs.subids
				otot.errors += obs.errors
				# Copy tod over to the right part of the output buffer
				otot.tod[dcum:dcum+len(obs.tod)] = obs.tod
				dcum += len(obs.tod)
			# Were we left with anything at all?
			if dcum == 0:
				raise utils.DataMissing(format_multi_exception(exceptions, eids))
			# Concatenate the work-lists into the final arrays
			for field, axis in append_fields:
				otot[field] = np.concatenate(otot[field],axis) if otot[field][0] is not None else None
			otot.cuts = socut.Simplecut.detcat(otot.cuts)
			# Trim tod in case we lost some detectors
			otot.tod = otot.tod[:dcum]
		finally:
			# Finally, move tod to the tod-buffer, where it's expected to be
			self.dev.pools.swap("pointing","tod")
		# Non-fatal errors
		if len(exceptions) > 0:
			otot.errors.append(utils.DataMissing(format_multi_exception(exceptions, eids)))
		return otot
	def group_obs(self, obsinfo, mode="obs"):
		return socommon.group_obs(obsinfo, mode=mode)

class FastMeta:
	def __init__(self, context):
		from sotodlib import core
		self.context   = context
		self.obsfiledb = core.metadata.ObsFileDb(context["obsfiledb"])
		# Open the database files we need for later obs lookups
		# 1. The preprocess archive index. Example:
		# /global/cfs/cdirs/sobs/sat-iso/preprocessing/satp1_20250108_init/process_archive.sqlite
		# Opening the actual file has to wait until we know what subid we have.
		self.prep_index = sqlite.open(cmeta_lookup(context, "preprocess"))
		# 2. The det_cal index. Example:
		# /global/cfs/cdirs/sobs/metadata/satp1/manifests/det_cal/satp1_det_cal_240312m/det_cal_local.sqlite
		self.dcal_index = sqlite.open(cmeta_lookup(context, "det_cal"))
		# 3. The detector info
		smurf_info = SmurfInfo(sqlite.open(cmeta_lookup(context, "smurf")))
		match_info = AssignmentInfo(sqlite.open(cmeta_lookup(context, "assignment")))
		wafer_info = WaferInfo(sqlite.open(cmeta_lookup(context, "wafer_info")))
		self.det_cache  = DetCache(self.obsfiledb, smurf_info, match_info, wafer_info)
		# 4. Absolute calibration
		with sqlite.open(cmeta_lookup(context, "abscal")) as index:
			acalfile = get_acalfile(index)
		self.acal_cache = AcalCache(acalfile)
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
	def read(self, subid):
		from sotodlib import core
		obsid, wslot, band, det_type = split_subid(subid)
		# Find which hdf files are relevant for this observation
		try:
			with bench.mark("fm_prepfile"):
				prepfile,  prepgroup  = get_prepfile (self.prep_index,  subid)
				dcalfile,  dcalgroup  = get_dcalfile (self.dcal_index,  subid)
		except KeyError as e: raise utils.DataMissing(str(e))
		# 1. Get our starting set of detectors
		with bench.mark("fm_dets"):
			try: detinfo = self.det_cache.get_dets(subid)
			except sqlite.sqlite3.OperationalError as e: raise errors.DataMissing(str(e))
			aman = core.AxisManager(core.LabelAxis("dets", detinfo.channels))
			aman.wrap("det_ids", detinfo.dets, [(0,"dets")])
			aman.wrap("bands",   detname2band(detinfo.dets), [(0,"dets")])
		ndet_full = aman.dets.count
		# 2. Load the necessary info from det_cal
		with bench.mark("fm_detcal"):
			with h5py.File(dcalfile, "r") as hfile:
				det_cal = hfile[dcalgroup][()]
			daman = core.AxisManager(core.LabelAxis("dets",np.char.decode(det_cal["dets:readout_id"])))
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
			paman = core.AxisManager(
					core.LabelAxis("dets", pl.dets),
					core.OffsetAxis("samps", count=pl.samps[0], offset=pl.samps[1]))
			# A bit awkward to time the time taken in the initialization
			bench.add("fm_prep_loader", time.time()-t1)

			# Need a better way to determine if hwp should be present or not
			has_hwp = "hwp_angle" in pl.group
			if has_hwp:
				with bench.mark("fm_hwp_angle"):
					paman.wrap("hwp_angle", pl.read("hwp_angle/hwp_angle","s"), [(0,"samps")])
				with bench.mark("fm_hwpss"):
					# These have order sin(1a),cos(1a),sin(2a),cos(2a),...
					paman.wrap("hwpss_coeffs", pl.read("hwpss_stats/coeffs","d"), [(0,"dets")])
			with bench.mark("fm_cuts"):
				paman.wrap("cuts_glitch", read_cuts(pl, "glitches/glitch_flags"),[(0,"dets"),(1,"samps")])
				paman.wrap("cuts_2pi",    read_cuts(pl, "jumps_2pi/jump_flag"),  [(0,"dets"),(1,"samps")])
				#paman.wrap("cuts_jump",   read_cuts(pl, "jumps/jump_flag"),      [(0,"dets"),(1,"samps")])
				paman.wrap("cuts_slow",   read_cuts(pl, "jumps_slow/jump_flag"), [(0,"dets"),(1,"samps")])
				#paman.wrap("cuts_turn",   read_cuts(pl, "turnaround_flags/turnarounds"), [(0,"dets"),(1,"samps")])

			# TODO: Remove the noise_mapmaking part once the transition is complete
			try:
				good = np.diff(np.concatenate([[0],pl.read("valid_data/ends")])) > 0
			except KeyError:
				try:
					good = np.diff(np.concatenate([[0],pl.read("valid_data/valid_data/ends")])) > 0
				except KeyError:
					good = np.diff(np.concatenate([[0],pl.read("noise_mapmaking/valid/ends")])) > 0
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
			abscal_cmb = self.acal_cache.get(stream_id, band).abscal_cmb
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
	from sotodlib import core
	if alloc is None: alloc = fast_g3.DummyAlloc()
	# Add "!", for mandatory field, to any field that doesn't have
	# the third entry present
	def field_pad(f): return (f[0],f[1],"!") if len(f)==2 else f
	fields = [field_pad(f) for f in fields]
	aman   = core.AxisManager(detax, sampax)
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

def calibrate(data, meta, mul=32, dev=None, prev_obs=None):
	from pixell import fft
	if dev is None: dev = device.get_device()
	# Merge the cuts. Easier to deal with just a single cuts object
	with bench.mark("merge_cuts"):
		cuts = socut.Sampcut.merge([meta.aman[name] for name in
			["cuts_glitch","cuts_2pi","cuts_slow"]])#,"cuts_jump","cuts_turn"]])
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left")
	# Go to a fourier-friendly length
	nsamp = fft.fft_len(data.signal.shape[1]//mul, factors=dev.lib.fft_factors)*mul
	timestamps, signal, cuts, az, el = [a[...,:nsamp] for a in [data.timestamps,data.signal,cuts,data.az,data.el]]
	hwp_angle = meta.aman.hwp_angle[:nsamp] if "hwp_angle" in meta.aman else None
	ninit = data.dets.count

	# prev_obs lets us pass in the result of calibrate run on
	# a different set of detectors for the same observation.
	# This is an optimization for load_multi, letting us avoid
	# repeating the same pointing correction multiple times
	if prev_obs is not None:
		ctime        = prev_obs.ctime
		el, az, roll = prev_obs.boresight
	else:
		with bench.mark("ctime"):
			ctime   = timestamps * meta.timestamp_to_ctime
		# Calibrate the pointing
		with bench.mark("boresight"):
			az      = az   * utils.degree
			el      = el   * utils.degree
			if "brot" in data:
				# SAT: boresight angle → roll
				roll = -data.brot [:nsamp]*utils.degree
			else:
				# LAT: corotator angle → roll
				roll = -data.corot[:nsamp]*utils.degree + el - 60*utils.degree
		with bench.mark("pointing correction"):
			az, el, roll = apply_pointing_model(az, el, roll, meta.pointing_model)

	# Do we need to deslope at float64 before it is safe to drop to float32?
	with bench.mark("signal → gpu", tfun=dev.time):
		signal_ = dev.pools["ft"].array(signal)
	with bench.mark("signal → float32", tfun=dev.time):
		signal  = dev.pools["tod"].empty(signal.shape, np.float32)
		signal[:] = signal_

	with bench.mark("calibrate", tfun=dev.time):
		# Calibrate to CMB µK
		phase_to_cmb = 1e6 * meta.abscal_cmb * meta.aman.phase_to_pW[:,None] # * meta.aman.relcal[:,None]
		signal *= dev.np.array(meta.dac_to_phase * phase_to_cmb)

	#with bench.mark("deproject sinel", tfun=dev.time):
	#	elrange = utils.minmax(el[::100])
	#	if elrange[1]-elrange[0] > 1*utils.arcmin:
	#		print("FIXME deprojecting el")
	#		deproject_sinel(signal, el, dev=dev)

	#with bench.mark("autocal", tfun=dev.time):
	#	autocal_elmod(signal, el, roll, meta.aman.focal_plane, dev=dev)

	# Subtract the HWP scan-synchronous signal. We do this before deglitching because
	# it's large and doesn't follow the simple slopes and offsets we assume there
	if hwp_angle is not None:
		with bench.mark("subtract_hwpss", tfun=dev.time):
			nmode = 16
			signal = subtract_hwpss(signal, hwp_angle, meta.aman.hwpss_coeffs[:,:nmode]*phase_to_cmb, dev=dev)

	# Deglitch and dejump
	w = 10
	with bench.mark("deglitch", tfun=dev.time):
		cuts.gapfill(signal, w=w, dev=dev)
		#deglitch_commonsep(signal, cuts, w=w, dev=dev)

	# 100 ms for this :(
	with bench.mark("deslope", tfun=dev.time):
		with dev.pools["ft"].as_allocator():
			gutils.deslope(signal, w=w, dev=dev, inplace=True)

	# FFT stuff should definitely be on the gpu. 640 ms
	with bench.mark("fft", tfun=dev.time):
		ftod = dev.pools["ft"].zeros((signal.shape[0],signal.shape[1]//2+1), utils.complex_dtype(signal.dtype))
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
		with dev.pools["tod"].as_allocator(): # tod buffer not in use atm
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

	# Sanity checks
	with bench.mark("measure noise", tfun=dev.time):
		rms = socommon.measure_rms(signal, dt=dt)
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
		signal = dev.np.ascontiguousarray(signal[good]) # 600 ms!
		good   = dev.get(good) # cuts, dets, fplane etc. need this on the cpu
		cuts   = cuts  [good]
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left after sanity cuts: raw %d meta %d rms %d cutdens %d overcut %d" % (meta.ndet_full, meta.aman.dets.count, nrms, ndens, nfinal))

	# Sogma uses the cut format [{dets,starts,lens},:]. Translate to this

	with bench.mark("cuts reformat"):
		ocuts = cuts.to_simple()

	# Our goal is to output what sogma needs. Sogma works on these fields:
	res  = bunch.Bunch()
	res.dets         = meta.aman.dets.vals[good]
	res.detids       = meta.aman.det_ids[good]
	res.bands        = meta.aman.bands[good]
	res.point_offset = meta.aman.focal_plane[good,1::-1]
	res.polangle     = meta.aman.focal_plane[good,2]
	res.ctime        = ctime
	res.boresight    = np.array([el,az,roll])
	res.hwp          = hwp_angle
	res.tod          = signal
	res.cuts         = ocuts
	res.response     = None
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
		with h5py.File(hfname, "r") as hfile:
			data = hfile[group][()]
			self.info[wafer_name] = bunch.Bunch(
				dets  = np.char.decode(data["dets:det_id"]),
				bands = np.char.decode(data["dets:wafer.bandpass"]),
				types = np.char.decode(data["dets:wafer.type"]),
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
			bts, order, edges = utils.find_equal_groups_fast(band_type)
			for bti, bt in enumerate(bts):
				band, type = bt.split(":")
				subid = "%s:%s:%s:%s" % (obsid, sinfo.wslot, band, type)
				inds  = ainds[order[edges[bti]:edges[bti+1]]]
				self.det_cache[subid] = bunch.Bunch(
						channels = ainfo.channels[inds],
						dets     = ainfo.dets[inds],
						detset   = dset,
						band     = band,
						type     = type,
				)
			self.detset_cache[obsid][sinfo.wslot] = dset
		self.done.add(obsid)

#def split_bands(dets):
#	bdets = {}
#	for di, det in enumerate(dets):
#		m = re.match(r"^\w+_(f\d\d\d|DARK)_.*$", det)
#		if not m: continue
#		band = m.group(1)
#		if band not in bdets:
#			bdets[band] = []
#		bdets[band].append(di)
#	return [(band, np.array(bdets[band])) for band in bdets]

class Subid:
	def __init__(self, obsid, wslot, band, type="OPTC"):
		self.obsid, self.wslot, self.band, self.type = obsid, wslot, band, type
		self.subid = ":".join(self)
	def __len__(self): return 4
	def __iter__(self):
		yield self.obsid
		yield self.wslot
		yield self.band
		yield self.type

def split_subid(subid): return Subid(*subid.split(":"))

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
			fname = os.path.dirname(self.fname) + "/" + entry[2]
			with h5py.File(fname,"r") as hfile:
				self.fp_cache[key] = hfile[entry[3]][()]
		return self.fp_cache[key]
	def get_by_subid(self, subid, det_cache):
		"""returns array with [('dets:det_id', 'S18'), ('xi', '<f4'), ('eta', '<f4'), ('gamma', '<f4')]"""
		toks = split_subid(subid)
		ctime      = float(toks.obsid.split("_")[1])
		wafer_name = detset2wafer_name(det_cache.get_detsets(toks.obsid)[toks.wslot])
		return self.get_by_wafer(wafer_name, ctime)

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
			fname = os.path.dirname(self.fname) + "/" + hfname
			with h5py.File(fname, "r") as hfile:
				param_str = hfile[gname].attrs["_scalars"]
				params    = bunch.Bunch(**json.loads(param_str))
				self.cache[key] = params
		return self.cache[key]
	def get_by_subid(self, subid):
		toks  = split_subid(subid)
		ctime = float(toks.obsid.split("_")[1])
		return self.get_by_time(ctime)

class AcalCache:
	def __init__(self, fname):
		self.fname  = fname
		self.raw    = bunch.read(fname).abscal
		self.lookup = {}
		for row in self.raw:
			key = row["dets:stream_id"].decode() + ":" + row["dets:wafer.bandpass"].decode()
			self.lookup[key] = bunch.Bunch(
				abscal_cmb = row["abscal_cmb"],
				abscal_rj  = row["abscal_rj"],
				#beam_fwhm  = row["beam_fwhm"],
				#beam_solid_angle = row["beam_solid_angle"],
				#cal_source = row["cal_source"].decode(),
			)
	def get(self, stream_id, band=None):
		"""Either get("ufm_mv13:f150") or get("ufm_mv13", "f150") work"""
		if band is not None: wafer_band = stream_id + ":" + band
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

def read_cuts(pl, path):
	shape     = pl.read(path + "/shape")
	edges     = pl.read(path + "/ends")
	intervals = pl.read(path + "/intervals")
	return expand_cuts_sampcut(shape, edges, intervals, inds = pl.inds)

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
	except StopIteration: raise KeyError("%s not found in preprocess index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_dcalfile(indexdb, subid):
	toks = split_subid(subid)
	query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' LIMIT 1;" % (toks.obsid)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in det_cal index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_matchfile(indexdb, detset):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:detset] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:detset] = '%s' LIMIT 1;" % (detset)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in det match index" % detset)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_smurffile(indexdb, detset):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:detset] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:detset] = '%s' LIMIT 1;" % (detset)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in smurf index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_wafer_file(waferdb, wafer_name):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:stream_id] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:stream_id] = '%s' LIMIT 1;" % (wafer_name)
	try: fname, gname = next(waferdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in wafer index" % wafer_name)
	return os.path.dirname(waferdb.fname) + "/" + fname, gname

def get_acalfile(indexdb):
	return os.path.dirname(indexdb.fname) + "/" + list(indexdb.execute("SELECT name FROM files INNER JOIN map ON files.id = map.file_id WHERE map.dataset = 'abscal'"))[0][0]

def read_wiring_status(fname, parse=True):
	import fast_g3
	for frame in fast_g3.get_header_frames(fname)["frames"]:
		if frame["type"] == "wiring":
			wiring = frame
	status = wiring["fields"]["status"]
	if parse:
		status = yaml.safe_load(status)
	return status

class WiringCache:
	def __init__(self, fields=None):
		self.cache  = {}
		self.fields = fields
	def get(self, fname):
		if fname not in self.cache:
			status = read_wiring_status(fname)
			if self.fields is not None:
				status = {field:status[field] for field in self.fields}
			self.cache[fname] = status
		return self.cache[fname]

def find_finfo_groups(metas):
	from sotodlib import core
	# convert finfos to strings, so we can easily find grups. A bit inelegant, but
	# no that inefficient. We assume that the finfo lists are already sorted
	# consistently
	finfo_strs = np.array(["%s,%d,%d" % (str(meta.finfos),meta.aman.samps.count,meta.aman.samps.offset) for meta in metas])
	uvals, order, edges = utils.find_equal_groups_fast(finfo_strs)
	ginfos = []
	for gi in range(len(uvals)):
		group  = order[edges[gi]:edges[gi+1]]
		finfos = metas[group[0]].finfos
		samps  = metas[group[0]].aman.samps
		dets   = core.LabelAxis("dets",np.concatenate([metas[i].aman.dets.vals for i in group]))
		ginfos.append(bunch.Bunch(finfos=finfos, dets=dets, samps=samps, inds=group))
	return ginfos

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
	dev.lib.sgemm("N", "N", nsamp, ndet, ncoeff, -1, B, nsamp, coeffs, ncoeff, 1, signal, nsamp)
	return signal

def polar_2d(x, y):
	r = (x**2+y**2)**0.5
	φ = np.arctan2(y,x)
	return r, φ

# This takes 250 ms! And the quaternion stuff would be tedious
# (but not difficult as such) to implement on the gpu
def apply_pointing_model(az, el, roll, model):
	from so3g.proj import quat
	# Ensure they're arrays, and avoid overwriting. These are
	# small arrays anyways
	[az, el, roll] = [np.array(a) for a in [az,el,roll]]
	if   model.version == "sat_naive": pass
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
	elif model.version == "lat_v1":
		# Reconstruct the corotator angle
		corot = el - roll - 60*utils.degree
		# Apply offsets
		az    += model.enc_offset_az
		el    += model.enc_offset_el
		corot += model.enc_offset_cr
		q_lonlat     = quat.rotation_lonlat(-az, el)
		q_mir_center = ~quat.rotation_xieta(model.mir_center_xi0, model.mir_center_eta0)
		q_el_roll    = quat.euler(2, el - 60*utils.degree)
		q_el_axis_center = ~quat.rotation_xieta(model.el_axis_center_xi0, model.el_axis_center_eta0)
		q_cr_roll    = quat.euler(2, -corot)
		q_cr_center  = ~quat.rotation_xieta(model.cr_center_xi0, model.cr_center_eta0)
		q_tot        = q_lonlat * q_mir_center * q_el_roll * q_el_axis_center * q_cr_roll * q_cr_center
		az, el, roll = quat.decompose_lonlat(q_tot)
		az          *= -1
	else: raise ValueError("Unrecognized model '%s'" % str(model.version))
	return az, el, roll

# This isn't robust to glitches. Should use a block-median or something
# if that's a problem
def deproject_sinel(signal, el, nmode=2, dev=None):
	ndet, nsamp = signal.shape
	if dev is None: dev = device.get_device()
	# Sky signal should go as 1/sin(el)
	depth = 1/dev.np.sin(dev.np.asarray(el, dtype=signal.dtype))
	# Build a basis
	depth -= dev.np.mean(depth)
	B   = dev.np.zeros((nmode,nsamp),signal.dtype)
	def nmat(x):
		fx = dev.lib.rfft(x)
		fx[...,:100] = 0
		return dev.lib.irfft(fx, x.copy())
	for i in range(nmode): B[i] = depth**(i+1)
	rhs = dev.np.einsum("mi,di->md",B,nmat(signal))
	div = dev.np.einsum("mi,ni->mn",B,nmat(B))
	idiv   = dev.np.linalg.inv(div)
	coeffs = idiv.dot(rhs)
	# Subtract the el-dependent parts. signal = -coeffs.T matmul B + signal
	# But fortran is column-major, so it's -coeffs
	#utils.call_help(dev.lib.sgemm, "N", "N", nsamp, ndet, nmode, -1, B, B.shape[1], coeffs, coeffs.shape[1], 1, signal, signal.shape[1])
	signal -= dev.np.einsum("md,mi->di", coeffs, B)

#def autocal_elmod(signal, bel, roll, xieta, nmode=2, bsize=2000, tol=0.1, dev=None):
#	if dev is None: dev = device.get_device()
#	assert nmode >= 2, "autocal_elmod needs nmode >= 2. The first two modes are an offset (discarded) and a slope (used for calibration)"
#	ndet, nsamp = signal.shape
#	# f (bel) = a * 1/sin(f(bel)) = a * 1/sin(f(bel0)) + a * f'(bel0) * Δbel + ...
#	# Let's call 1/sin(el) = x, and 1/sin(bel) = bx, and x = g(bx)
#	# f (bx) = a*g(bx) = a*g(bx0) + a*g'(bx0)*Δbx + ...
#	# Could expand around middle of xieta instad of boresight, but nice to have
#	# a standard location for all runs.
#	bx = 1/np.sin(bel)
#	# Need d(det_x)/dbx for each detector. Will use finite difference over the whole
#	# elevation range as representative. We assume constant roll
#	belrange = utils.minmax(bel)
#	bx1, bx2 = 1/np.sin(belrange)
#	roll = np.mean(roll)
#	x1 = 1/np.sin(calc_det_el(belrange[0], roll, xieta))
#	x2 = 1/np.sin(calc_det_el(belrange[1], roll, xieta))
#	gprime = (x2-x1)/(bx2-bx1)
#	print(gprime)
#	Δbx = dev.np.array(bx-0.5*(bx1+bx2))
#	# With this, the tod should be const + a*g'*Δbx in each block
#	# Build the blocks
#	nblock = nsamp//bsize
#	btod   = signal[:,:nblock*bsize].reshape(ndet,nblock,bsize)
#	bΔbx   = Δbx[:nblock*bsize].reshape(nblock,bsize)
#	# Build basis
#	B     = dev.np.zeros((nmode,nblock,bsize), btod.dtype)
#	B[:]  = bΔbx
#	for i in range(nmode): B[i] **= i
#	# Fit in blocks
#	rhs   = dev.np.einsum("mbs,dbs->bdm", B, btod) # [nblock,ndet,nmode]
#	div   = dev.np.einsum("mbs,nbs->bmn", B, B)   # [nblock,nmode,nmode]
#	# Reduce to blocks with healthy determinant
#	det   = dev.np.linalg.det(div)
#	good  = det > 0
#	if dev.np.sum(good) == 0: raise ValueError("elmod calibration failed for all blocks. Is elevation actually varying?")
#	good  = det > dev.np.median(det[good])*tol
#	if dev.np.sum(good) == 0: raise ValueError("elmod calibration failed for all blocks. Is elevation actually varying?")
#	rhs, div = rhs[good], div[good]
#	# Solve the remaining blocks
#	idiv  = dev.np.linalg.inv(div)
#	amps  = dev.np.einsum("bmn,bdn->bdm", idiv, rhs) # [nblock,ndet,nmode]
#	slope = dev.get(amps[:,:,1]) # µK [nblock,ndet]
#	# slope = a*g'. Divide out g' to get a
#	ba    = slope/gprime # µK [nblock,ndet]
#	# Use blocks to get robust mean and error
#	a, da = robust_mean(ba, 0)
#	
#	np.savetxt("test_elgain.txt", np.array([
#		a, da, gprime, xieta[:,0], xieta[:,1]]).T, fmt="%15.7e")
#	np.savetxt("test_elgain_t.txt", ba, fmt="%15.7e")
#	1/0
#
#
#
#def calc_det_el(bel, roll, xieta):
#	bel, roll, xi, eta = np.broadcast_arrays(bel, roll, *xieta.T[:2])
#	# Boresight
#	bore = coordsys.Coords(az=bel*0, el=bel, roll=roll)
#	# Focal plane
#	q    = coordsys.rotation_xieta(xi, eta)
#	el   = (bore*q).el
#	return el
#
#def robust_mean(arr, axis=-1, quantile=0.1):
#	axis= axis%arr.ndim
#	arr = np.sort(arr, axis=axis)
#	n   = utils.nint(arr.shape[axis]*quantile)
#	arr = arr[(slice(None),)*axis+(slice(n,-n),)]
#	mean= np.mean(arr, axis)
#	err = np.std(arr, axis)/arr.shape[axis]**0.5
#	return mean, err

# The idea of this was to effectively gapfill using the
# common mode. Maybe it could work, but in my test, it
# didn't really fix the gapfilling artifacts in the map
# (maybe a small improvement), while it significantly
# increased both small-scale and large-scale noise in the
# map, surprisingly.
def deglitch_commonsep(signal, cuts, w=10, dev=None):
	# Split signal into common mode and rest
	ccuts   = merge_det_cuts(cuts)
	cmode   = robust_common_mode(signal)
	signal -= cmode
	# Deglitch common mode subtracted tod
	cuts.gapfill(signal, w=w, dev=dev)
	# Deglitch common mode
	ccuts.gapfill(cmode[None], w=w, dev=dev)
	# Add back common mode
	signal += cmode

def robust_common_mode(signal, bsize=8, dev=None):
	if dev is None: dev = device.get_device()
	signal = signal[:signal.shape[0]//bsize//bsize*bsize*bsize].reshape(-1,bsize,bsize,signal.shape[-1])
	return dev.np.median(dev.np.median(dev.np.mean(signal,-2),-2),-2)

def merge_det_cuts(cuts):
	# 1. Turn cuts into a list of single-det cuts
	dcuts = []
	for b in cuts.bins:
		if b[1] == b[0]: continue
		dcuts.append(socut.Sampcut((b-b[0])[None], cuts.ranges[b[0]:b[1]], nsamp=cuts.nsamp))
	# 2. Merge them into a single cut
	ocuts = socut.Sampcut.merge(dcuts)
	return ocuts

def format_multi_exception(exceptions, subids):
	msgs   = np.array([str(ex) for ex in exceptions])
	uvals, order, inds = utils.find_equal_groups_fast(msgs)
	omsgs  = []
	for ui, uval in enumerate(uvals):
		mysubs = [subids[o] for o in order[inds[ui]:inds[ui+1]]]
		omsgs.append(",".join(mysubs) + ": " + uval)
	return ", ".join(omsgs)

def make_metas_compatible(metas, sinfo, tol=0.1):
	# sinfo gives us the start and end time of the raw tods,
	# but we want the ones after any offsets or truncation
	t1s_raw, t2s_raw, nsamps_raw = sinfo.T
	dts    = (t2s_raw-t1s_raw)/(nsamps_raw-1)
	offs   = np.array([meta.aman.samps.offset for meta in metas])
	nsamps = np.array([meta.aman.samps.count  for meta in metas])
	# t1s and t2s are absolute timestamps of start/end of current samples
	t1s  = t1s_raw + dts*offs
	t2s  = t1s_raw + dts*(offs+nsamps)
	# Find the narrowest ctime range. This is what we want to restrict to
	t1 = np.max(t1s)
	t2 = np.min(t2s)
	dur = t2-t1
	if dur <= 0: raise ValueError("no sample overlap")
	# i1s and i2s are the sample offsets of the target start/end from the absolute start.
	# These are the offsets axismanager will want
	i1s = (t1-t1s_raw)/dts
	i2s = (t2-t1s_raw)/dts
	# Duration in samples should be the same, to within the tolerance
	# Sample phase should also be the same
	idurs = i2s-i1s
	isubs = utils.rewind(i1s-i1s[0], ref=0, period=1)
	if np.max(np.abs(isubs)) > tol or np.max(np.abs(idurs-idurs[0])) > tol:
		raise ValueError("incompatible timestamps")
	# Restrict sample ranges
	# i1 is offset relative to our absolute start
	i1s = utils.nint(i1s)
	i2s = utils.nint(i2s)
	for mi, meta in enumerate(metas):
		meta.aman.restrict("samps", slice(i1s[mi], i2s[mi]), in_place=True)
	# Could check that detector names are unique here, but it's not really necessary
	# Return the final sample and detector count
	ndet  = sum([meta.aman.dets.count for meta in metas])
	nsamp = metas[0].aman.samps.count
	return ndet, nsamp

def get_obs_sampinfo(obsdb, subids):
	sinfo = np.zeros((len(subids),3))
	for i, subid in enumerate(subids):
		toks = split_subid(subid)
		sinfo[i] = next(obsdb.execute("select start_time, stop_time, n_samples from obs where obs_id = ?", [toks.obsid]))
	return sinfo

def detname2band(detnames):
	return np.char.partition(np.char.partition(detnames, "_")[:,2],"_")[:,0]

def catch2list(catch):
	if   isinstance(catch, (list,tuple)): return catch
	elif catch == "all":      return (Exception,)
	elif catch == "expected": return (Exception,) # TODO: Actual list here
	elif catch == "none":     return ()
	else: raise ValueError("Unrecognized catch '%s'" % str(catch))
