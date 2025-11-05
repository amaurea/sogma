import numpy as np, os
from pixell import config, ephem

config.default("planet_list",   "Mercury,Venus,Mars,Jupiter,Saturn,Uranus,Neptune", "What planets the 'planets' keyword in object_cut expands to")
# Vesta has a peak brightness of 1 Jy @f150. These asteroids get within 4% of that
# (40 mJy) at some point in their orbit by extrapolation. This is the 5σ forecasted
# depth-1 sensitivity at f150, and would be even weaker after dilution from multile
# exposures, so this should be a safe level without cutting too much.
config.default("asteroid_list", "Vesta,Ceres,Pallas,Juno,Eunomia,Hebe,Iris,Pluto,Eris,Amphitrite,Makemake,Hygiea,Herculina,Metis,Flora,Dembowska,Melpomene,Haumea,Psyche,Laetitia,Massalia", "What asteroids the 'asteroids' keyword in object_cut expands to")
# This supports environment varaibles
config.default("asteroid_path", "/global/cfs/cdirs/sobs/users/sigurdkn/ephemerides/objects")

def setup(asteroid_path=None):
	asteroid_path = os.path.expandvars(config.get("asteroid_path", asteroid_path))
	if not asteroid_path: return
	astephem = ephem.PrecompEphem(asteroid_path)
	ephem.add(astephem)
