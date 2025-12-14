import numpy as np
import matplotlib.pyplot as plt
import constants_Ganymede
import os
import tkinter as tk
from tkinter import filedialog


def SRIM_data(elem):
    """
    Read a SRIM file.

    Parameters:
        elem (string): chemical element impacting Ganymede's surface, see the class ChemicalElement in constants.py.
    
    Returns:
        tuple (tuple):
            - energy (float): ion energy [keV].
            - Se (float): electronic stopping power of incident ion [keV/µm].
            - Sn (float): nuclear stopping power of incident ion [keV/µm].
            - PR: projected range of incident ion [µm].
    """
    # file_name to be changed according to the name of the SRIM file.
    file_name = f"{elem} in  H- O_0.93.txt"
    path_name = os.path.join(constants_Ganymede.WORK_DIR, file_name)

    if(path_name):
        with open(path_name, "r") as fichier:
                lines = fichier.readlines()
    else:
        root = tk.Tk()
        root.withdraw()
        # Open a window to select a file
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Text file", "*.txt")]
        )

        if file_path:
            try:
                with open(file_path, "r") as file:
                    lines = file.readlines()
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No selected file.")
    
    # Process only the lines with data
    used_lines = lines[24:-13]
    energy_list = []
    Se_list = []
    Sn_list = []
    PR_list = []
    
    for line in used_lines:
        #print(line.strip())  # .strip() to delete line break
        column = line.strip().split()
        column = [valeur.replace(",", ".") for valeur in column]
        column[0] = float(column[0])
        column[2] = float(column[2])
        column[3] = float(column[3])
        column[4] = float(column[4])

        # energies are converted in keV and the projected range (PR) in µm.
        if(column[1] == "eV"):
            column[0] = column[0] * 1e-3
        if(column[1] == "MeV"):
            column[0] = column[0] * 1e3
        if(column[5] == "A"):
            column[4] = column[4] * 1e-4
        if(column[5] == "mm"):
            column[4] = column[4] * 1e3

        energy_list.append(column[0])
        Se_list.append(column[2])
        Sn_list.append(column[3])
        PR_list.append(column[4])
    
    energy = np.array(energy_list)
    Se = np.array(Se_list)
    Sn = np.array(Sn_list)
    PR = np.array(PR_list)

    #print (f"energy = {energy}")
    #print (f"Se = {Se}")
    #print (f"Sn = {Sn}")
    #print (f"PR = {PR}")
    print (f"SRIM data from file '{file_name}' have been successfully uploaded.")

    return energy, Se, Sn, PR



def ESTAR_data(data_name):
    """
    Read a ESTAR file.

    Parameters:
        data_name (string): data read in the ESTAR file, can be "LET" or "range".

    Returns:
        tuple (tuple):
            - energy (array): energy range of the incident electron [keV]
            - data (array): LET [keV/µm] or maximum penetration depth [cm] of the electrons, depending of the parameter data_name.
    """
    # Since the energy range is not the same for the LET and the projected range, these two parameters are distinguished in parameter.
    # file_name to be changed according to the name of the SRIM file.
    if (data_name == "LET"):
        file_name = f"ESTAR_0.93.txt"
    elif (data_name == "range"):
        file_name = f"ESTAR_range_0.93.txt"
    else:
        raise ValueError("The parameter data_name must be 'LET' or 'range'.")
    path_name = os.path.join(constants_Ganymede.WORK_DIR, file_name)

    if(path_name):
        with open(path_name, "r") as file:
                lines = file.readlines()
                #print(f"lines = {lines}")
    else:
        root = tk.Tk()
        root.withdraw()
        # Open a window to select a file
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Text file", "*.txt")]
        )

        if file_path:
            try:
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    print(f"lines = {lines}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("No selected file.")
    
    # Process only the lines with data
    used_lines = lines[8:]
    #print(f"used lines = {used_lines}")
    energy_list = []
    data_list = []
    
    if (data_name == "LET"):
        for line in used_lines:
            column = line.strip().split()
            column[0] = float(column[0])
            column[3] = float(column[3])

            column[0] = column[0] * 1e3    # conversion from MeV to keV
            column[3] = column[3] * 0.93 * 0.1    # conversion from MeV.cm²/g to keV/µm

            energy_list.append(column[0])
            data_list.append(column[3])
    
    elif (data_name == "range"):
        for line in used_lines:
            column = line.strip().split()
            column[0] = float(column[0])
            column[1] = float(column[1])

            column[0] = column[0] * 1e3     # conversion from MeV to keV
            column[1] = column[1] / 0.93    # conversion from g/cm² to cm

            energy_list.append(column[0])
            data_list.append(column[1])
    
    energy = np.array(energy_list)
    data = np.array(data_list)

    #print (f"energy = {energy}")
    #print (f"{data_name} = {data}")
    print (f"ESTAR data from file {file_name} have been successfully uploaded.")

    return energy, data


def ion_flux_data(region, elem):
    """
    Read and interpolate the points from the numerization of the ion flux curves from Poppe+2018.

    Parameters:
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        elem (string): chemical element impacting Ganymede's surface, see the class ChemicalElement in constants.py.

    Returns:
        tuple (tuple):
            - energy_INT (array): ion energy range interpolated on a higher number of points [eV].
            - flux_INT (array): ion flux interpolated on a higher number of points [ions/cm²/s]
    """

    path = constants_Ganymede.WORK_DIR
    data = np.loadtxt(os.path.join(path, region + '_' + elem + '_accurate.txt'), delimiter=' ') # curves from the paper have been numerized
    energy = data[:, 0]
    flux = data[:, 1]

    energy_INT = np.logspace(np.log10(energy[0]), np.log10(energy[-1]), 15000)
    flux_INT = np.interp(energy_INT, energy, flux)  # interpolation of the numerized data on the range energy_INT
    'print("Energy_INT :", energy_INT)'
    'print("Flux_INT :", flux_INT)'

    """
    plt.plot(energy_INT, flux_INT, label='INT curves', linewidth=4)
    plt.plot(energy, flux, label='initial curves')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('energy')
    plt.ylabel('flux')
    plt.legend()
    plt.show()
    """

    return energy_INT, flux_INT


def electron_flux_data(region, measure):
    """
    Read and interpolate the points from the numerization of the electron flux curves from Liuzzo+2020.

    Parameters:
        region (string): Ganymede's regions as presented in Liuzzo+2020, can be "60N90N", "20N60N", "20N20S", "20S60S", "60S90S".
        measure (string): number of the Galileo encounter which have measured the energy flux around Ganymede, can be "G8", "G1" or "G28".

    Returns:
        tuple (tuple):
            - energy_INT (array): ion energy range interpolated on a higher number of points [keV].
            - flux_INT (array): ion flux interpolated on a higher number of points [ions/cm²/s/sr/keV]
    """

    path = constants_Ganymede.WORK_DIR
    data = np.loadtxt(os.path.join(path, 'L20_flux' + measure + '_' + region + '.txt'), delimiter=' ')
    energy = data[:, 0]
    flux = data[:, 1]

    energy_INT = np.logspace(np.log10(energy[0]), np.log10(energy[-1]), 3000)
    flux_INT = np.interp(energy_INT, energy, flux)
    #print("Energy_INT :", energy_INT)
    #print("Flux_INT :", flux_INT)

    # for the case where there is a hole in the raw data
    if (region=="20N20S" and measure=="G1"):
        for i in range(0,len(energy_INT)):
            if ((energy_INT[i]>50 and energy_INT[i]<1600) or (energy_INT[i]>12000 and energy_INT[i]<26000)):
                flux_INT[i] = 0


    """
    plt.plot(energy_INT, flux_INT, label='INT curves', linewidth=4)
    plt.plot(energy, flux, label='initial curves')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('energy')
    plt.ylabel('flux')
    plt.legend()
    plt.show()
    """

    return energy_INT, flux_INT



if __name__ == "__main__":
    SRIM_data('H')
    #print("test")
    #Liuzzo2020_data('20N20S', 'G1')
    #SRIM_data("H")
    #ESTAR_data("range")
    #ion_flux_data("polar", "H")
