import time
import numpy as np
import cupy
from pixell import utils

def round_up  (n, b): return (n+b-1)//b*b
def round_down(n, b): return n//b*b

def cutime():
	cupy.cuda.runtime.deviceSynchronize()
	return time.time()

def apply_window(tod, nsamp, exp=1):
	"""Apply a cosine taper to each end of the TOD."""
	if nsamp <= 0: return
	ap = anypy(tod)
	taper   = 0.5*(1-ap.cos(ap.arange(1,nsamp+1)*ap.pi/nsamp))
	taper **= exp
	tod[...,:nsamp]  *= taper
	tod[...,-nsamp:] *= taper[::-1]

def aranges(lens):
	ntot = np.sum(lens)
	itot = np.arange(ntot)
	offs = np.repeat(np.cumsum(np.concatenate([[0],lens[:-1]])), lens)
	return itot-offs

def cumsum0(vals):
	res = np.empty(len(vals), vals.dtype)
	if res.size == 0: return res
	res[0] = 0
	res[1:] = np.cumsum(vals[:-1])
	return res

def split_ranges(dets, starts, lens, maxlen):
	# Vectorized splitting of detector ranges into subranges.
	# Probably premature optimization, since it's a bit hard to read.
	# Works by duplicating elements for too long ranges, and then
	# calculating new sub-offsets inside these
	dets, starts, lens = [np.asarray(a) for a in [dets,starts,lens]]
	nsplit  = (lens+maxlen-1)//maxlen
	odets   = np.repeat(dets, nsplit)
	subi    = aranges(nsplit)
	sublen  = np.repeat(lens, nsplit)
	subns   = np.repeat(nsplit, nsplit)
	offs    = subi*sublen//subns
	ostarts = np.repeat(starts, nsplit)+offs
	offs2   = (subi+1)*sublen//subns
	oends   = offs2-offs
	return odets, ostarts, oends

# GPU-CPU agnostic functions

def anypy(arr):
	"""Return numpy or cupy depending on what type of array we have.
	Useful for writing code that works both on cpu and gpu"""
	if   isinstance(arr, cupy.ndarray): return cupy
	else: return np

def legbasis(order, n, ap=np):
	x   = ap.linspace(-1, 1, n, dtype=np.float32)
	out = ap.zeros((order+1, n),dtype=np.float32)
	out[0] = 1
	if order>0:
		out[1] = x
	for i in range(1,order):
		out[i+1,:] = ((2*i+1)*x*out[i]-i*out[i-1])/(i+1)
	return out

def leginverses(basis):
	"""Given a lebasis B[nmode,nsamp], compute all the nsamp truncated
	inverses of BB'. The result will have shape [nsamp,nmode,nmode]"""
	# This function is 20x faster on the cpu
	ap   = anypy(basis)
	nmode, nsamp = basis.shape
	mask = ap.tril(ap.ones((nsamp,nsamp), basis.dtype))
	BBs  = ap.einsum("ai,bi,ji->jab", basis, basis, mask)
	iBBs = ap.zeros_like(BBs)
	# These ones should be safe to invert
	iBBs[nmode-1:] = ap.linalg.inv(BBs[nmode-1:])
	# Handle the unsafe ones manually
	for i in range(0, nmode-1):
		iBBs[i,:i+1,:i+1] = ap.linalg.inv(BBs[i,:i+1,:i+1])
		for j in range(i+1, nmode):
			iBBs[i,j,j] = 1
	return iBBs

def safe_inv(a):
	with utils.nowarn():
		res = 1/a
		res[~np.isfinite(res)] = 0
	return res

def safe_invert_ivar(ivar, tol=1e-3):
	vals = ivar[ivar!=0]
	ref  = np.mean(vals[::100])
	iivar= ivar*0
	good = ivar>ref*tol
	iivar[good] = 1/ivar[good]
	return iivar

# Not done
#def gemm(alpha, A, B, beta, C, Aop="N", Bop="N", Cop="N", handle=None):
#	iscup = [isinstance(a, cupy.ndarray) for a in [A,B,C]]
#	if iscup[1] != iscup[0] or iscup[2] != iscup[0]: raise ValueError("All arrays must agree on cpu/gpu-ness")
#	if B.dtype  != A.dtype  or C.dtype  != A.dtype: raise ValueError("All arrays must have the same dtype")
#	dtype = A.dtype
#	if isinstance(A, cupy.ndarray):
#		# We're on the gpu
#		if handle is None: handle = cupy.cuda.Device().cublas_handle
#		if   dtype == np.float32: fun = cublas.sgemm
#		elif dtype == np.float64: fun = cublas.dgemm
#		else: raise ValueError("Only float32 and float64 supported")
#		opmap = {"N": cublas.CUBLAS_OP_N, "T": cublas.CUBLAS_OP_T, "L": cublas.CUBLAS_SIDE_RIGHT}
#		fun(handle, opmap[Aop], 







