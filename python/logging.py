import time
from pixell import memory, colors
from . import gmem, device

class Logger:
	def __init__(self, dev=None, level=0, id=0, fmt="{id:3d} {t:6.2f} {mem:6.2f} {gmem_rest:6.2f} {gmem:6.2f} {msg:s}"):
		"""Memory device memory output won't work if dev isn't passed"""
		self.level = level
		self.id    = id
		self.fmt   = fmt
		self.dev   = dev
		self.t0    = time.time()
	def print(self, message, level=0, id=None, color=None, end="\n"):
		if level > self.level: return
		if id is not None and id != self.id: return
		gmem_tot     = self.dev.memuse(type="total") if self.dev else 0
		gmem_pools   = self.dev.memuse(type="pools") if self.dev else 0
		gmem_rest    = self.dev.memuse(type="np")-gmem_pools if self.dev else 0
		gmem_unknown = gmem_tot - gmem_pools - gmem_rest
		msg = self.fmt.format(id=self.id, t=(time.time()-self.t0)/60, mem=memory.current()/1024**3, max=memory.max()/1024**3, gmem=gmem_tot/1024**3, gmem_pools=gmem_pools/1024**3, gmem_rest=gmem_rest/1024**3, gmem_unknown=gmem_unknown/1024**3, msg=message)
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
