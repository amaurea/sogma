import numpy as np, os
from pixell import config, ephem

config.default("planet_list",   "Mercury,Venus,Mars,Jupiter,Saturn,Uranus,Neptune", "What planets the 'planets' keyword in object_cut expands to")
# Vesta has a peak brightness of 1 Jy @f150. These asteroids get within 4% of that
# (40 mJy) at some point in their orbit by extrapolation. This is the 5σ forecasted
# depth-1 sensitivity at f150, and would be even weaker after dilution from multile
# exposures, so this should be a safe level without cutting too much.
#config.default("asteroid_list", "Vesta,Ceres,Pallas,Juno,Eunomia,Hebe,Iris,Pluto,Eris,Amphitrite,Makemake,Hygiea,Herculina,Metis,Flora,Dembowska,Melpomene,Haumea,Psyche,Laetitia,Massalia", "What asteroids the 'asteroids' keyword in object_cut expands to")
config.default("asteroid_list", "Ceres,Vesta,Pallas,Bamberga,Iris,Juno,Hygiea,Zelinda,Daphne,Hebe,Eunomia,Fortuna,Melpomene,Interamnia,Flora,Thisbe,Metis,Prokne,Aurelia,Euphrosyne,Davida,Winchester,Europa,Desiderata,Ekard,Victoria,Atalante,Psyche,Lamberta,Vibilia,Egeria,Alexandra,Io,Julia,Amphitrite,Lampetia,Eugenia,Xanthippe,Liguria,Irene,Patientia,Isis", "Asteroids with peak f150 brightness > 100 mJy according to SPT's list")
#config.default("asteroid_list", "Ceres,Vesta,Pallas,Bamberga,Iris,Juno,Hygiea,Zelinda,Daphne,Hebe,Eunomia,Fortuna,Melpomene", "Asteroids with peak f150 brightness > 200 mJy according to SPT's list")

# This supports environment varaibles
config.default("asteroid_path", "/global/cfs/cdirs/sobs/users/sigurdkn/ephemerides/objects")

def setup(asteroid_path=None):
	asteroid_path = os.path.expandvars(config.get("asteroid_path", asteroid_path))
	if not asteroid_path: return
	astephem = ephem.PrecompEphem(asteroid_path)
	ephem.add(astephem)
