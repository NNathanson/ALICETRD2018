import numpy as np

DEFAULT_BASELINE = 0
DEFAULT_DETECTOR_DIMENSIONS = [1.0, 1.0, 0.1]
DEFAULT_DATA_FOLDER = 'DATA/'
DEFAULT_INTERESTING_DATA_FOLDER = "DATA_INTERESTING/"

DATA_EXCLUDE_MASK = np.zeros((12, 144, 30), dtype=bool)
DATA_EXCLUDE_MASK[4:8,0:144,:] = True

CURRENT_FILE = '0310_+1430_-1200_15000_0261_zs 17.13.27'