# IRIS - Irradiation Response of Icy and Solid surfaces

IRIS is a numerical code designed to simulate the effects of irradiation on planetary surfaces, including surface erosion due to sputtering, dose deposition, and water ice amorphization. Different regions can be distinguished on the object, by giving different particle fluxes or different surface temperatures. It is particularly adapted for icy moons (e.g., Ganymede, Europa) and carbonaceous objects in the Solar System.

All the results are saved as .txt files in the WORK_DIR directory and can be plotted using external software (e.g. Igor).

Computations are executed by running main.py.

---

## Key Features
- Sputtering modeling: Simulates particle ejection from surfaces under irradiation, and estimates the erosion rate of the surface.
- Dose calculation: Estimates electronic, nuclear and total doses deposited into the surface.
- Amorphous fraction calculation: Models structural changes on icy surfaces.
- Generalizable: Adaptable to various Solar System objects (moons, asteroids, comets).

---

## Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `matplotlib`, `os`, `tkinter`, `dataclasses`, `unittest`, `time`, `scipy`, `shutil`, `datetime`.

---

## To be added
- sputtering yields of other materials than water ice (erosion_thickness.py)

---

## Structure
### config_Ganymede.json
It should be adapted by the user. It contains:
    - 'region_ion' and 'region_electron': list of the different regions on the object irradiated by ions and electrons, respectively. It is usually determined by the particle fluxes.
    - 'measure': string to give the measure we consider, specific to the electron flux given by Liuzzo+2020.
    - 'temperature': ranges of surface temperatures, can define different ranges for the different regions.
    - 'tasks': different computations that can be realized by IRIS (described hereafter in the main.py).

### main.py
This file contains the main functions to:
    - Compute erosion rate due to ion and electron irradiation (erosion_thickess.py): first computes the sputtering yields of the irradiated materials. Second multiplies the sputtering yield with the particle flux to obtain the erosion rate. Results are saved in a new created file 'WORK_DIR/erosion'.
    - Compute doses without sputtering as a function of depth, with a log scale to plot them (dose.py): compute basically the amount of deposited energy, considering a static surface. Results are saved in a new created file 'WORK_DIR/dose'.
    - Compute the dose taking sputtering of the surface into account (dose.py): a more realistic case where a dynamic surface is considered. Different type of sputtered molecules can be considered, to simulate a more or less important erosion. Results are saved in a newly created file 'WORK_DIR/dose'.
    - Compute the amorphous fraction induced by irradiation (amorphization.py): for icy surfaces, crystalline water ice can be amorphized due to irradiation. Different scenarios can be considered, whether recrystallization is considered additionaly to amorphization. Results are saved in a new created file 'WORK_DIR/amorph'.

### constants_Ganymede.py
It should be adapted by the user. It contains:
    - 'WORK_DIR': path where files will be created to save the results, and where file needed for computations must be stored (TO BE ADAPTED).
    - 'DO_BACKUP': TRUE (by default) for saving overwritten files from computations, False if you don't want to save previous computations.
    - Surface properties: molecular mass and density.
    - Names of the files used in the code (SRIM, ESTAR and particle fluxes).
    - 'ChemicalElement': list of ion species irradiating the surface (TO BE ADAPTED).

### upload_data.py
This file allows users to:
    - Read and upload data from SRIM and ESTAR files.
    - Read and upload files containing particle flux data. 

---

## Files the user must have in WORK_DIR
- SRIM files to compute the stopping powers of the ions impacting the surface.
- ESTAR files to compute the LET of the electrons impacting the surface.
- Files with particle flux data, can be numerization of the curves from a paper or data measurements. Files must contain one first column with the particle energy, and a second one with the particle flux.

---

## Usage example
To run IRIS:
- Adapt 'config_Ganymede.json' and 'constants_Ganymede.py' to your needs.
- Place all required files (SRIM, ESTAR, particle flux) in 'WORK_DIR'.
- Execute the main script: python main.py.