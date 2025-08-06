import numpy as np, contextlib, json, time, os, scipy, re
from pixell import utils, fft, bunch, bench, sqlite
from sotodlib import preprocess, core
import fast_g3, h5py, yaml, ast
from so3g.proj import quat
from .. import device, gutils
from . import soquery, socommon

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

class SoFastLoader:
	def __init__(self, configfile, dev=None, mul=32):
		# Set up our metadata loader
		self.config  = yaml.safe_load(configfile)
		self.config, self.context = preprocess.preprocess_util.get_preprocess_context(configfile)
		# Precompute the set of valid tags. Sadly this requires a scan through the whole
		# database, but it's not that slow so far
		self.tags   = soquery.get_tags(self.context.obsdb.conn)
		self.fast_meta = FastMeta(self.config, self.context)
		self.mul     = mul
		self.dev     = dev or device.get_device()
	def query(self, query=None, sweeps=True, output="sogma"):
		if output not in ["sqlite", "resultset", "sogma"]:
			raise ValueError("Unrecognized output format '%s" % str(output))
		res_db = soquery.eval_query(self.context.obsdb.conn, query, tags=self.tags)
		if output == "sqlite": return res_db
		info   = core.metadata.resultset.ResultSet.from_cursor(res_db.execute("select * from obs"))
		if output == "resultset": return info
		dtype  = [("id","U100"),("ndet","i"),("nsamp","i"),("ctime","d"),("dur","d"),("baz","d"),("waz","d"),("bel","d"),("wel","d"),("r","d"),("sweep","d",(4,2))]
		obsinfo= np.zeros(len(info), dtype).view(np.recarray)
		obsinfo.id = info["subobs_id"]
		obsinfo.ndet  = utils.dict_lookup(flavor_ndets_per_band, info["tube_flavor"])
		obsinfo.nsamp = info["n_samples"]
		obsinfo.ctime = info["start_time"]
		obsinfo.dur   = info["stop_time"]-info["start_time"]
		# Here come the parts that have to do with pointing.
		# These arrays have a nasty habit of being object dtype
		obsinfo.baz   = info["az_center"].astype(np.float64) * utils.degree
		obsinfo.bel   = info["el_center"].astype(np.float64) * utils.degree
		obsinfo.waz   = info["az_throw" ].astype(np.float64) * utils.degree
		wafer_centers, obsinfo.r = socommon.wafer_info_multi(info["tube_slot"], info["wafer_slots_list"])
		if sweeps:
			# Everything before this took around 16 µs/obs
			# Make_sweep itself takes 20 µs/obs. So it's the biggest
			# term, but not dominant
			obsinfo.sweep = socommon.make_sweep(obsinfo.ctime, obsinfo.baz, obsinfo.waz, obsinfo.bel, wafer_centers)
		return obsinfo
	def load(self, subid):
		try:
			with bench.mark("load_meta"):
				meta = self.fast_meta.read(subid)
			# Load the raw data
			with bench.mark("load_data"):
				data = fast_data(meta.finfos, meta.aman.dets, meta.aman.samps)
			# Calibrate the data
			with bench.mark("load_calib"):
				obs = calibrate(data, meta, mul=self.mul, dev=self.dev)
		#except Exception as e:
		except () as e:
			# FIXME: Make this less broad
			raise utils.DataMissing(type(e).__name__ + " " + str(e))
		# Add timing info
		obs.timing = [("meta",bench.t.load_meta),("data",bench.t.load_data),("calib",bench.t.load_calib)]
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
		smurf_info = SmurfInfo(sqlite.open(cmeta_lookup(context["metadata"], "smurf")))
		match_info = AssignmentInfo(sqlite.open(cmeta_lookup(context["metadata"], "assignment")))
		self.det_cache  = DetCache(context.obsfiledb, smurf_info, match_info)
		# 4. Absolute calibration
		with sqlite.open(cmeta_lookup(context["metadata"], "abscal")) as index:
			acalfile = get_acalfile(index)
		self.acal_cache = AcalCache(acalfile)
		# 4. Focal plane
		self.fp_cache   = FplaneCache(cmeta_lookup(context["metadata"], "focal_plane"))
		# 5. Pointing model, which seems to be static for now
		self.pointing_model_cache = PointingModelCache(cmeta_lookup(context["metadata"], "pointing_model"))
	def read(self, subid):
		obsid, wslot, band = subid.split(":")
		# Find which hdf files are relevant for this observation
		try:
			with bench.mark("fm_prepfile"):
				prepfile,  prepgroup  = get_prepfile (self.prep_index,  subid)
				dcalfile,  dcalgroup  = get_dcalfile (self.dcal_index,  subid)
		except KeyError as e: raise utils.DataMissing(str(e))
		# 1. Get our starting set of detectors
		with bench.mark("fm_dets"):
			try: detinfo = self.det_cache.get(subid)
			except sqlite.sqlite3.OperationalError as e: raise errors.DataMissing(str(e))
			aman = core.AxisManager(core.LabelAxis("dets", detinfo.channels))
			aman.wrap("det_ids", detinfo.dets, [(0,"dets")])
		# 2. Load the necessary info from det_cal
		with bench.mark("fm_detcal"):
			with h5py.File(dcalfile, "r") as hfile:
				det_cal = hfile[dcalgroup][()]
			daman = core.AxisManager(core.LabelAxis("dets",np.char.decode(det_cal["dets:readout_id"])))
			good  = np.full(daman.dets.count, True)
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
			# Set up the focal plane
			fp_info = fp_info[finds]
			focal_plane = np.array([fp_info["xi"], fp_info["eta"], fp_info["gamma"]]).T
			good    = np.all(np.isfinite(focal_plane),1)
			aman.restrict("dets", aman.dets.vals[ainds[good]])
			aman.wrap("focal_plane", focal_plane[good], [(0,"dets")])
		# 4. Load what we need from the preprocess archive. There aren't that many
		# superflous detectors in these files, and I didn't find an efficient
		# way to read only those needed, so just read all of them and merge
		t1 = time.time()
		with PrepLoader(prepfile, prepgroup) as pl:
			paman = core.AxisManager(
					core.LabelAxis("dets", pl.dets),
					core.OffsetAxis("samps", count=pl.samps[0], offset=pl.samps[1]))
			# A bit awkward to time the time taken in the initialization
			bench.add("fm_prep_loader", time.time()-t1)

			#print("FIXME skipping relcal read")
			#with bench.mark("fm_relcal"):
			#	paman.wrap("relcal", pl.read("lpf_sig_run1/relcal","d"), [(0,"dets")])

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
				paman.wrap("cuts_jump",   read_cuts(pl, "jumps/jump_flag"),      [(0,"dets"),(1,"samps")])
				paman.wrap("cuts_slow",   read_cuts(pl, "jumps_slow/jump_flag"), [(0,"dets"),(1,"samps")])
				#paman.wrap("cuts_turn",   read_cuts(pl, "turnaround_flags/turnarounds"), [(0,"dets"),(1,"samps")])

		# Eventually we will need to be able to read in sample-ranges.
		# Easiest to handle that with paman.restrict("samps", ...) here.
		# The rest should follow automatically

		#i = utils.find(paman.dets.vals, "sch_ufm_mv20_1740684471_3_294")
		#p = paman.cuts_glitch
		#print("i", i)
		#print(paman.dets.vals[i])
		#print(p.ranges[p.bins[i,0]:p.bins[i,1]])
		#1/0

		with bench.mark("fm_merge2"):
			aman.merge(paman)

		# 4. Get stuff from the data file header
		with bench.mark("fm_detsets"):
			detset = self.det_cache.get(subid).detset
		with bench.mark("fm_getfiles"):
			finfos = self.context.obsfiledb.get_files(obsid)[detset]
		# Get the filter params
		with bench.mark("fm_status"):
			status = read_wiring_status(finfos[0][0])
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
		# Return our results. We don't put everything in an axismanager
		# because that has significant overhead, and we don't need an
		# axismanager for things that don't have any axes
		return bunch.Bunch(aman=aman, iir_params=iir_params, finfos=finfos,
			dac_to_phase = np.pi/2**15, timestamp_to_ctime=1e-8,
			abscal_cmb = abscal_cmb, pointing_model=pointing_model)

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

def calibrate(data, meta, mul=32, dev=None):
	if dev is None: dev = device.get_device()
	# Merge the cuts. Easier to deal with just a single cuts object
	with bench.mark("merge_cuts"):
		cuts = merge_cuts([meta.aman[name] for name in ["cuts_glitch","cuts_2pi","cuts_jump","cuts_slow"]])#,"cuts_turn"]])
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left")
	# Go to a fourier-friendly length
	nsamp = fft.fft_len(data.signal.shape[1]//mul, factors=dev.lib.fft_factors)*mul
	timestamps, signal, cuts, az, el = [a[...,:nsamp] for a in [data.timestamps,data.signal,cuts,data.az,data.el]]
	hwp_angle = meta.aman.hwp_angle[:nsamp] if "hwp_angle" in meta.aman else None

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

	#print("FIXME doing manual relcal")
	#with bench.mark("calibrate_manual", tfun=dev.time):
	#	# Calibrate to CMB µK
	#	phase_to_cmb = 1e6 * meta.abscal_cmb * meta.aman.phase_to_pW[:,None]
	#	signal *= dev.np.array(meta.dac_to_phase * phase_to_cmb)
	#	relcal  = common_mode_calibrate(signal, dev=dev)
	#	signal *= relcal[:,None]

	with bench.mark("deproject cosel", tfun=dev.time):
		elrange = utils.minmax(el[::100])
		if elrange[1]-elrange[0] > 1*utils.arcmin:
			print("FIXME deprojecting el")
			deproject_cosel(signal, el, dev=dev)

	#print("FIXME common mode deprojection")
	#cmode = dev.np.median(signal,0)[None]
	## Smooth it
	#utils.deslope(cmode, inplace=True)
	#fmode = dev.lib.rfft(cmode)
	#fmode[:,fmode.shape[-1]//10:] = 0
	#dev.lib.irfft(fmode, cmode)
	#cmode /= cmode.shape[-1]
	#def nmath(a): return np.diff(utils.block_reduce(a, 100))
	#def deproj(a, B, nmath):
	#	hNa = nmath(a)
	#	hNB = nmath(B)
	#	rhs = np.sum(hNa[:,None,:]*hNB[None,:,:],-1)
	#	div = np.sum(hNB[:,None,:]*hNB[None,:,:],-1)
	#	idiv = np.linalg.inv(div)
	#	amp = np.einsum("ab,db->da",idiv,rhs)
	#	return a - amp.dot(B)
	#signal[:] = deproj(signal, cmode, nmath)

	# Subtract the HWP scan-synchronous signal. We do this before deglitching because
	# it's large and doesn't follow the simple slopes and offsets we assume there
	if hwp_angle is not None:
		with bench.mark("subtract_hwpss", tfun=dev.time):
			nmode = 16
			signal = subtract_hwpss(signal, hwp_angle, meta.aman.hwpss_coeffs[:,:nmode]*phase_to_cmb, dev=dev)

	# Deglitch and dejump
	w = 10
	with bench.mark("deglitch", tfun=dev.time):
		deglitch(signal, cuts, w=w, dev=dev)
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
		with dev.pools["pointing"].as_allocator():
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

	with bench.mark("measure noise", tfun=dev.time):
		rms  = measure_rms(signal)

	# Restrict to these detectors
	with bench.mark("final detector prune", tfun=dev.time):
		tol    = 0.2
		ref    = np.median(rms[rms!=0])
		good   = (rms > ref*tol)&(rms < ref/tol)
		signal = dev.np.ascontiguousarray(signal[good]) # 600 ms!
		good   = dev.get(good) # cuts, dets, fplane etc. need this on the cpu
		cuts   = cuts  [good]
		if len(cuts.bins) == 0: raise utils.DataMissing("no detectors left")

	# Sogma uses the cut format [{dets,starts,lens},:]. Translate to this

	with bench.mark("cuts reformat"):
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
	res.boresight    = np.array([el,az,roll])
	res.hwp          = hwp_angle
	res.tod          = signal
	res.cuts         = ocuts
	res.response     = None
	# Test per-detector response
	#res.response     = dev.np.zeros((2,len(res.tod)),res.tod.dtype)
	#res.response[0]  = 2
	#res.response[1]  = -1
	return res

#################
# Helpers below #
#################

# Hard-coded raw wafer detector counts per band. Independent of
# telescope type, etc.
flavor_ndets_per_band = {"lf":118, "mf": 864, "uhf": 864}

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
		return self

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
		return self

class DetCache:
	def __init__(self, obsfiledb, smurf_info, ass_info):
		self.obsfiledb = obsfiledb
		self.smurf_info= smurf_info
		self.ass_info  = ass_info
		self.cache     = {}
		self.done      = set()
	def get(self, subid):
		obsid, wslot, band = subid.split(":")
		self._prepare(obsid)
		return self.cache[subid]
	def _prepare(self, obsid, force=False):
		"""Read in the det lists for obsid, if necessary."""
		if obsid in self.done and not force: return
		for dset in self.obsfiledb.get_detsets(obsid):
			try:
				sinfo = self.smurf_info.get(dset)
				ainfo = self.ass_info  .get(dset)
			except KeyError:
				print("warning: dset %s missing" % dset)
				# Some wafers may be missing
				continue
			# split dets into bands
			# TODO: Should split cache into
			# (obsid,wslot) → detset and (detset,band) → detinfo
			for band, inds in split_bands(ainfo.dets):
				subid = "%s:%s:%s" % (obsid, sinfo.wslot, band)
				self.cache[subid] = bunch.Bunch(
						channels=ainfo.channels[inds],
						dets=ainfo.dets[inds],
						detset=dset,
				)
		self.done.add(obsid)

def split_bands(dets):
	bdets = {}
	for di, det in enumerate(dets):
		m = re.match(r"^\w+_(f\d\d\d|DARK)_.*$", det)
		if not m: continue
		band = m.group(1)
		if band not in bdets:
			bdets[band] = []
		bdets[band].append(di)
	return [(band, np.array(bdets[band])) for band in bdets]

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
		obs_id, wafer_slot, band = subid.split(":")
		ctime      = float(obs_id.split("_")[1])
		wafer_name = "_".join(det_cache.get(subid).detset.split("_")[:2])
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
		obs_id, wafer_slot, band = subid.split(":")
		ctime      = float(obs_id.split("_")[1])
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

# These functions could use some low-level acceleration

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
	"""Get sort cuts by detector, and remove empty ones"""
	obins   = []
	oranges = []
	o1 = 0
	for b1,b2 in bins:
		o2 = o1
		for ri in range(b1,b2):
			if ranges[ri,1] > ranges[ri,0]:
				o2 += 1
				oranges.append(ranges[ri])
		obins.append((o1,o2))
		o1 = o2
	obins = np.array(obins, np.int32).reshape((-1,2))
	oranges = np.array(oranges, np.int32).reshape((-1,2))
	return obins, oranges

def merge_cuts(cuts):
	"""Get the union of the given list of Sampcuts"""
	# 1. Flatten each cut
	ndet = len(cuts[0].bins)
	nsamp= cuts[0].nsamp
	N    = nsamp+1 # +1 to avoid merging across dets
	franges = []
	for ci, cut in enumerate(cuts):
		bind    = get_range_dets(cut.bins, cut.ranges).astype(np.int64) # avoid overflow
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
	oranges  = oranges.astype(np.int32)
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
# TODO: Remember to check whether the way we use ranges is consistent with
# it using absolute sample numbers, not ones relatve to the current tod start
class Sampcut:
	def __init__(self, bins=None, ranges=None, nsamp=0, simplify=False):
		if bins   is None: bins   = np.zeros((0,2),np.int32)
		if ranges is None: ranges = np.zeros((0,2),np.int32)
		if simplify:
			with bench.mark("simplify cuts"):
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

def get_prepfile(indexdb, subid):
	obsid, wslot, band = subid.split(":")
	# Inconsistent format here too
	if "dets:wafer.bandpass" in indexdb.columns("map"):
		query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id], [dets:wafer_slot], [dets:wafer.bandpass] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' AND [dets:wafer_slot] = '%s' AND [dets:wafer.bandpass] = '%s' LIMIT 1;" % (obsid, wslot, band)
	else:
		query  = "SELECT files.name, dataset, file_id, files.id, [obs:obs_id], [dets:wafer_slot] FROM map INNER JOIN files ON file_id = files.id WHERE [obs:obs_id] = '%s' AND [dets:wafer_slot] = '%s' LIMIT 1;" % (obsid, wslot)
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
	except StopIteration: raise KeyError("%s not found in det match index" % detset)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_smurffile(indexdb, detset):
	query  = "SELECT files.name, dataset, file_id, files.id, [dets:detset] FROM map INNER JOIN files ON file_id = files.id WHERE [dets:detset] = '%s' LIMIT 1;" % (detset)
	try: fname, gname = next(indexdb.execute(query))[:2]
	except StopIteration: raise KeyError("%s not found in smurf index" % subid)
	return os.path.dirname(indexdb.fname) + "/" + fname, gname

def get_acalfile(indexdb):
	return os.path.dirname(indexdb.fname) + "/" + list(indexdb.execute("SELECT name FROM files INNER JOIN map ON files.id = map.file_id WHERE map.dataset = 'abscal'"))[0][0]

#def get_pointing_model(fname):
#	# Looks like the pointing model doesn't support time dependence at the
#	# moment, so we just have this simple function
#	with sqlite.open(fname) as sfile:
#		rows = list(sfile.execute("select name from files limit 2"))
#		if len(rows) != 1: raise IOError("Time-dependent pointing model loading not implemented. Figure out new format and implement")
#		hfname  = os.path.dirname(fname) + "/" + rows[0][0]
#		with h5py.File(hfname, "r") as hfile:
#			param_str = hfile["pointing_model"].attrs["_scalars"]
#		return bunch.Bunch(**json.loads(param_str))

def cmeta_lookup(cmeta, name):
	for entry in cmeta:
		for key in ["label", "name"]: # both not standardized?
			if key in entry and entry[key] == name:
				return entry["db"]

def read_wiring_status(fname, parse=True):
	for frame in fast_g3.get_header_frames(fname)["frames"]:
		if frame["type"] == "wiring":
			wiring = frame
	status = wiring["fields"]["status"]
	if parse:
		status = yaml.safe_load(status)
	return status

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

def measure_rms(tod, bsize=32, nblock=10):
	ap  = device.anypy(tod)
	tod = tod[:,:tod.shape[1]//bsize*bsize]
	tod = tod.reshape(tod.shape[0],-1,bsize)
	bstep = max(1,tod.shape[1]//nblock)
	tod = tod[:,::bstep,:][:,:nblock,:]
	return ap.median(ap.std(tod,-1),-1)

def measure_rms2(tod, bsize=32, down=128):
	ap  = device.anypy(tod)
	tod = tod[:,:tod.shape[1]//bsize//down*bsize*down]
	tod = tod.reshape(tod.shape[0],-1,bsize,down)
	return ap.median(ap.std(ap.mean(tod,-1),-1),-1)

# This takes 250 ms! And the quaternion stuff would be tedious
# (but not difficult as such) to implement on the gpu
def apply_pointing_model(az, el, roll, model):
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

def common_mode_calibrate(signal, bsize=80, down=20, dev=None):
	ndet, nsamp = signal.shape
	if dev is None: dev = device.get_device()
	with dev.pools["pointing"].as_allocator():
		# downgrade to reduce noise, then median to reduce impact of glitches
		dsig = dev.np.median(dev.np.mean(signal[:,:nsamp//bsize//down*bsize*down].reshape(ndet,-1,bsize,down),-1),-1)
		# Find the strongest eigenmode
		cov  = dev.np.cov(dev.np.diff(dsig,axis=-1))
		# Initial normalization, to avoid single detectors dominating
		g1   = dev.np.diag(cov)**0.5
		cov /= g1[:,None]
		cov /= g1[None,:]
		# Find the strongest eigenvector
		E, V = dev.np.linalg.eigh(cov)
		g2   = V[:,-1]
		# Normalize to be mostly positive and have a mean of 1
		g2  /= dev.np.mean(g2)
	# We want to be able to apply the calibration by multiplying
	gtot = 1/g2
	return gtot

# This isn't robust to glitches. Should use a block-median or something
# if that's a problem
def deproject_cosel(signal, el, nmode=2, dev=None):
	ndet, nsamp = signal.shape
	if dev is None: dev = device.get_device()
	# Sky signal should go as 1/cos(el)
	depth = 1/dev.np.cos(dev.np.asarray(el, dtype=signal.dtype))
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
	print("div")
	print(dev.get(div))
	idiv   = dev.np.linalg.inv(div)
	coeffs = idiv.dot(rhs)
	print(dev.get(coeffs))
	# Subtract the el-dependent parts. signal = -coeffs.T matmul B + signal
	# But fortran is column-major, so it's -coeffs
	print(dev.np.std(signal))
	#utils.call_help(dev.lib.sgemm, "N", "N", nsamp, ndet, nmode, -1, B, B.shape[1], coeffs, coeffs.shape[1], 1, signal, signal.shape[1])
	signal -= dev.np.einsum("md,mi->di", coeffs, B)
	print(dev.np.std(signal))

def deglitch(signal, cuts, w=10, dev=None):
	if dev is None: dev = device.get_device()
	# Define sample ranges just before and after each
	# cut, where we will measure a mean value for dejumping
	with bench.mark("get_cut_borders", tfun=dev.time):
		borders = get_cut_borders(cuts,w)
	# Get on the format Kendrick wants
	with bench.mark("index_map", tfun=dev.time):
		ndet      = len(cuts.bins)
		ncut      = len(borders)
		nper      = cuts.bins[:,1]-cuts.bins[:,0]
		index_map        = np.zeros((ncut,5), np.int32)
		index_map[:,0]   = np.repeat(np.arange(ndet,dtype=np.int32),nper)
		index_map[:,1:]  = borders.reshape(ncut,-1)
		index_map2       = np.zeros((ncut,4), np.int32)
		index_map2[:,0]  = index_map[:,0]
		index_map2[:,1]  = np.repeat(cuts.bins[:,0],nper)
		index_map2[:,2:] = cuts.ranges
		index_map  = dev.np.array(index_map)
		index_map2 = dev.np.array(index_map2)
		bvals      = dev.np.zeros((ncut,2), signal.dtype)
	# Measure the border values
	with bench.mark("get_border_means", tfun=dev.time):
		dev.lib.get_border_means(bvals, signal, index_map)
	# Use them to deglitch
	with bench.mark("deglitch core", tfun=dev.time):
		dev.lib.deglitch(signal, bvals, index_map2)

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
	deglitch(signal, cuts, w=w, dev=dev)
	# Deglitch common mode
	deglitch(cmode[None], ccuts, w=w, dev=dev)
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
		dcuts.append(Sampcut((b-b[0])[None], cuts.ranges[b[0]:b[1]], nsamp=cuts.nsamp))
	# 2. Merge them into a single cut
	ocuts = merge_cuts(dcuts)
	return ocuts
