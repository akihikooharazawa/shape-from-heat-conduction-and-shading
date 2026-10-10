"""
Constants for shape from heat reconstruction.

This module provides constants and configuration parameters for:
- Camera settings
- Optimization parameters
- Physical constants
"""

import numpy as np

# CAMERA = 'Boson'
CAMERA = 'Boson'
# CAMERA = 'Boson+'
FOCUS_DISTANCE: float = 368.3
HFOV: float = 24.3  # degrees
VFOV: float = 19.5  # degrees
IMAGE_WIDTH_METRIC = 2*FOCUS_DISTANCE * \
    np.tan(np.deg2rad(HFOV/2))  # 158.58582759

SIGMA = 5.6704e-8/(1000**2) # W/(mm^2*K^4)
# SIGMA = 5.6704e-8 # W/(m^2*K^4)
EPSILON = 0.90
AMBIENT_TEMP: float = 295.372  # Kelvin
OBJECT_TEMP = AMBIENT_TEMP

# Just took this value from google for a room at 25c. The typical value ranges from 10 to 100
CONVECTION_COEFF = 10/(1000**2) # W/(mm^2*K)

# Density of the material in kg/mm^3 at 98 celcius
Density = {
    "aluminium": 2700/(1000**3),
    "pvc": 1330/(1000**3),
    "glass": 2500/(1000**3),
    "copper": 8960/(1000**3),
    "polystyrene": 1050/(1000**3),
    "wood": 897/(1000**3),
    "steel": 7930/(1000**3),
    'brick': 2200/(1000**3)

}
# Specific heat of the material in J/(kg*K) at 98 celcius
SpecificHeat = {
    "aluminium": 978,
    "pvc": 880,
    "glass": 840,
    "copper": 385,
    "polystyrene": 1300,
    "wood": 2380,
    "steel": 280,
    "brick": 800,
}
Absorptivity = {
    "aluminium": 1.0,
    "pvc": 1.0,
    "glass": 0.9,
    "copper": 0.9,
    "polystyrene": 0.9,
    "wood": 0.9,
    "steel": 0.9,
    "brick": 0.9,
}

TDiff = {
    "aluminium": 97.0, #97, #1.0, #97,
    "aluminium-6061": 64,
    "glass" : 0.34,
    "pvc": 0.17, # conductivity - 0.2 W/mK,
    "polystyrene": 0.5, #0.5,
    "copper": 111,
    "wood": 0.082,
    "steel": 4.2,
    "brick": 0.52,
}


# reflected energy
Emiss = EPSILON
distance = 0.4
TRefl = 22.22 #21.85

# atmospheric attenuation
TAtm = 295.37  # 22.22°C in Kelvin
Tau = 1.0

# external optics
TExtOptics = 20
# TransmissionExtOptics = 1.0
TransmissionExtOptics = 0.94

# Boson+ RBFO
# 6.23748836e+03 9.15243088e+01 1.08100666e+00 1.95966142e-01

# Boson RBFO
# R: 1080000.0, B: 1524.0, F: 1.0, O: 15500.0
R_const = {
    "Boson": 1080000.0,
    "Boson+": 6.23748836e+03
}
B_const = {
    "Boson": 1524.0,
    "Boson+": 9.15243088e+01
}
F_const = {
    "Boson": 1.0,
    "Boson+": 1.08100666e+00
}

O_const = {
    "Boson": 15500.0,
    "Boson+": 1.95966142e-01
}

def set_focus_distance(focus_distance: float) -> None:
    global FOCUS_DISTANCE, IMAGE_WIDTH_METRIC
    FOCUS_DISTANCE = focus_distance
    print("FOCUS_DISTANCE set: ", FOCUS_DISTANCE)
    IMAGE_WIDTH_METRIC = 2*FOCUS_DISTANCE * np.tan(np.deg2rad(HFOV/2))


