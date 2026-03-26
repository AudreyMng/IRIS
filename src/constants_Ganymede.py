from dataclasses import dataclass
import numpy as np
import os

WORK_DIR = os.path.join('C:\\', 'Users', 'moingeoa', 'These', 'Amorphization', 'flux_Poppe2018', 'test_python')

WORK_DIR_EROSION = os.path.join(WORK_DIR, "erosion")
os.makedirs(WORK_DIR_EROSION, exist_ok=True)   # Create backup folder if it does not exist

WORK_DIR_DOSE = os.path.join(WORK_DIR, "dose")
os.makedirs(WORK_DIR_DOSE, exist_ok=True)   # Create backup folder if it does not exist

WORK_DIR_AMORPH = os.path.join(WORK_DIR, "amorph")
os.makedirs(WORK_DIR_AMORPH, exist_ok=True)   # Create backup folder if it does not exist


SECOND_IN_YEAR = 365.25*24*3600
DO_BACKUP = True


# values for crystalline water ice on Ganymede
C = np.pi
M = 18  # g/mol
Na = 6.022e23   # mol-1
density = 0.93  # g/cm^3

# name of the files
def name_file_SRIM(elem):
    file_name = f"{elem} in  H- O_{density}.txt"
    return file_name

def name_file_ESTAR(data_name):
    if (data_name == "LET"):
        file_name = f"ESTAR_{density}.txt"
    elif (data_name == "range"):
        file_name = f"ESTAR_range_{density}.txt"
    else:
        raise ValueError("The parameter data_name must be 'LET' or 'range'.")
    return file_name

def name_file_ion_flux(region, elem):
    file_name = f"{region}_{elem}_accurate.txt"
    return file_name

def name_file_electron_flux(region, measure):
    file_name = f"L20_flux{measure}_{region}.txt"
    return file_name

@dataclass
class ChemicalElement:
    name: str
    atomic_mass: int
    atomic_number: int

ELEM_LISTE = np.array([
        ChemicalElement(name='H', atomic_mass=1, atomic_number=1),
        ChemicalElement(name='O', atomic_mass=16, atomic_number=8),
        ChemicalElement(name='S', atomic_mass=32, atomic_number=16),
        ChemicalElement(name='H_thermal', atomic_mass=1, atomic_number=1),
        ChemicalElement(name='O_thermal', atomic_mass=16, atomic_number=8),
    ])