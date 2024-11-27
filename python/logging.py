import time
from pixell import memory, colors
from . import gmem

class Logger:
	def __init__(self, level=0, id=0, fmt="{id:3d} {t:6.2f} {mem:6.2f} {gmem:6.2f} {gmem2:6.2f} {msg:s}"):
		self.level = level
		self.id    = id
		self.fmt   = fmt
		self.t0    = time.time()
	def print(self, message, level=0, id=None, color=None, end="\n"):
		if level > self.level: return
		if id is not None and id != self.id: return
		gmem2 = gmem.get_gpu_memuse() - gmem.mempool.used_bytes()
		msg = self.fmt.format(id=self.id, t=(time.time()-self.t0)/60, mem=memory.current()/1024**3, max=memory.max()/1024**3, gmem=gmem.mempool.used_bytes()/1024**3, gmem2=gmem2/1024**3, msg=message)
		if color is not None:
			msg = color + msg + colors.reset
		print(msg, end=end)
	def setdefault(self):
		global L
		L.logger = self
		return self

class LogWrapper:
	def __init__(self, logger):
		self.logger = logger
	def print(self, message, level=0, id=None, color=None, end="\n"):
		self.logger.print(message, level=level, id=id, color=color, end=end)

default_logger = Logger()
L = LogWrapper(default_logger)
