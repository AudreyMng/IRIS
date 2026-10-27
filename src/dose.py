import numpy as np
import matplotlib.pyplot as plt
import time
import os

import upload_data as upload
import erosion_thickness as erosion

import constants_Ganymede

from utils import write_file

from scipy.interpolate import interp1d

def doses_incidence_i(dose_ref, depth_ref, i):
    """
    Compute the dose deposited by an ion with an incidence angle i.

    Parameters:
        dose_ref (array): dose deposited by ions arriving perpendicularly to the surface [eV/atom].
        depth_ref (array): depth on which dose_ref has been calculated [eV/atom].
        i (float): incidence angle of the ion [°].

    Returns:
        dose_incidence_i (array): dose_ref affected by the incident angle i [eV/atom].
    """
    dose_incidence_i = np.zeros(len(dose_ref))

    depth_projected = depth_ref * np.cos(i * np.pi/180)    # depth along the ion way arriving with an angle i.
    dose_incidence_i = np.interp(depth_ref, depth_projected, dose_ref)

    """
    plt.plot(depth_ref, dose_incidence_i, label='dose incidence', linewidth=4)
    plt.plot(depth_ref, dose_ref, label='dose ref')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('dose')
    plt.ylabel('depth')
    plt.legend()
    plt.show()
    """
    
    return dose_incidence_i


def compute_dose_incidence(dose_ref, depth_ref):
    """
    Compute the dose deposited by ions arriving in all directions.

    Parameters:
        dose_ref (array): dose deposited by ions arriving perpendicularly to the surface [eV/atom].
        depth_ref (array): depth on which dose_ref has been calculated [cm].

    Returns:
        dose_incidence (array): dose deposited by ions arriving in all directions [eV/atom].
    """

    dose_incidence = np.zeros(len(dose_ref))

    for i in range(0,89):
        dose_incidence_i = doses_incidence_i(dose_ref, depth_ref, i)
        dose_incidence += dose_incidence_i
    
    # the amount of energy brought by dose_ref should remain the same for dose_incidence_total
    sum_ref = np.sum(dose_ref)
    sum_incidence = np.sum(dose_incidence)
    if (sum_incidence != 0):
        dose_incidence = (dose_incidence / sum_incidence) * sum_ref

    return dose_incidence




def compute_doses_ions(region, delta_t_yr, density = 0.93, log_scale = False):
    """
    Compute electronic, nuclear and total doses deposited by ions during delta_t_yr on Ganymede's surface.

    Parameters:
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        delta_t_yr (float): time during which the doses will be calculated [yr].
        density (float, by default = 0.93): density of water ice [g/cm^3].
        log_scale (bool, by default = False): to compute the doses with a linear depth. True if you want to use a log scale.

    Returns:
        tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - dose_e (array): electronic dose deposited by ions [eV/atom].
            - dose_n (array): nuclear dose deposited by ions [eV/atom].
            - dose (array): total dose (electronic + nuclear) deposited by ions [eV/atom].
    """
    start_time = time.time()

    path_dose = constants_Ganymede.WORK_DIR_DOSE    
    delta_t = delta_t_yr*constants_Ganymede.SECOND_IN_YEAR #s

    print(f"DOSE CALCULATION for {region} region")
    # the log scale is used to plot doses without sputtering and have a good resolution for the lowest depth.
    # to compute doses with sputtering, we need a linear scale to have layers of the same depth.
    if log_scale:
        depth = np.logspace(np.log10(1e-6), np.log10(1.4e-1), 9000)
        extension = '_log.txt'
    else:
        depth = np.linspace(1e-6, 1.4e-1, 50000)
        extension = '.txt'
    
    energy_INT_eV = np.logspace(np.log10(10), np.log10(1e8), 15000) # from 10 eV to 100 MeV
    energy_INT_keV = energy_INT_eV * 1e-3
    
    dose_e, dose_n = np.zeros(len(depth)), np.zeros(len(depth))

    for element in constants_Ganymede.ELEM_LISTE:
        elem = element.name
        start_time_elem = time.time()
        print(f"Element {elem}")

        energy_INT_ion, flux_INT = upload.ion_flux_data(region, elem)  # eV
        flux_divE_INT = flux_INT / energy_INT_ion
        SRIM_energy, SRIM_Se, SRIM_Sn, _ = upload.SRIM_data(elem)

        Se_INT = np.interp(energy_INT_keV, SRIM_energy, SRIM_Se)
        Sn_INT = np.interp(energy_INT_keV, SRIM_energy, SRIM_Sn)
        """
        plt.plot(energy_INT_keV, Se_INT)
        plt.plot(energy_INT_keV, Sn_INT)
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        """

        dose_e_step, dose_n_step, dose_e_elem, dose_n_elem = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))

        for i in range(0, len(energy_INT_ion)-1):
            E = energy_INT_ion[i]
            for j in range(0, len(depth)-1):
                if (E<10):  # 10 eV
                    break
                #index_E = energy_INT.index(E)   # eV
                index_E = np.argmin(np.abs(energy_INT_eV - E))
                #print('index_E = ', index_E)
                Se = Se_INT[index_E] * 1e7      # eV/cm
                Sn = Sn_INT[index_E] * 1e7      # eV/cm

                dose_e_step[j] = constants_Ganymede.M * delta_t / (3*density*constants_Ganymede.Na) * Se * flux_divE_INT[i] * (energy_INT_ion[i+1] - energy_INT_ion[i])
                dose_n_step[j] = constants_Ganymede.M * delta_t / (3*density*constants_Ganymede.Na) * Sn * flux_divE_INT[i] * (energy_INT_ion[i+1] - energy_INT_ion[i])

                dose_e_elem[j] = dose_e_elem[j] + dose_e_step[j]
                dose_n_elem[j] = dose_n_elem[j] + dose_n_step[j]
                #print('dose_e_elem[j] = ', dose_e_elem[j])
                #print('i = ', i)
                #print('j = ', j)

                E = E - (Se+Sn)*(depth[j+1] - depth[j])
                #if (E<10):
                    #break
                #print('E = ', E)
        
        dose_e_elem_incidence = compute_dose_incidence(dose_e_elem, depth)
        dose_n_elem_incidence = compute_dose_incidence(dose_n_elem, depth)
        #print(f'dose_e_elem = {dose_e_elem}')
        #print(f'dose_e_elem_incidence = {dose_e_elem_incidence}')

        # save the doses e and n for elem
        file_dose_e_elem = os.path.join(path_dose, f'dose_e_{elem}_{region}_{delta_t_yr:.2e}yr{extension}')
        file_dose_n_elem = os.path.join(path_dose, f'dose_n_{elem}_{region}_{delta_t_yr:.2e}yr{extension}')
        write_file(file_dose_e_elem, depth, dose_e_elem_incidence)
        write_file(file_dose_n_elem, depth, dose_n_elem_incidence)

        dose_e += dose_e_elem_incidence
        dose_n += dose_n_elem_incidence

        end_time_elem = time.time()
        execution_time_elem = (end_time_elem - start_time_elem)/60

        print(f"Execution time for elem {elem}: {execution_time_elem} minutes")
        print(' ')
    
    dose = dose_e + dose_n

    # save the doses e and n for region, and the total dose
    file_dose_e = os.path.join(path_dose, f'dose_e_{region}_{delta_t_yr:.2e}yr{extension}')
    file_dose_n = os.path.join(path_dose, f'dose_n_{region}_{delta_t_yr:.2e}yr{extension}')
    file_dose = os.path.join(path_dose, f'dose_TOTAL_{region}_{delta_t_yr:.2e}yr{extension}')
    write_file(file_dose_e, depth, dose_e)
    write_file(file_dose_n, depth, dose_n)
    write_file(file_dose, depth, dose)

    print(f'Electronic dose (De) calculated for region {region}. Saved in file dose_e_{region}_{delta_t_yr:.2e}yr{extension}')
    print(f'Nuclear dose (Dn) calculated for region {region}. Saved in file dose_n_{region}_{delta_t_yr:.2e}yr{extension}')
    print(f'Total dose (De + Dn) calculated for region {region}. Saved in file dose_TOTAL_{region}_{delta_t_yr:.2e}yr{extension}')

    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(' ')

    return depth, dose_e, dose_n, dose


def compute_doses_electrons(region, delta_t_yr, measure = "G8", density = 0.93, log_scale = False):
    """
    Compute doses deposited by electrons during delta_t_yr on Ganymede's surface.

    Parameters:
        region (string): Ganymede's regions as presented in Liuzzo+2020, can be "60N90N", "20N60N", "20N20S", "20S60S", "60S90S".
        delta_t_yr (float): time during which the doses will be calculated [yr].
        measure (string, by default = "G8"): number of the Galileo encounter which have measured the energy flux around Ganymede, can be "G8", "G1" or "G28".
        density (float, by default = 0.93): density of water ice [g/cm^3].
        log_scale (bool, by default = False): to compute the doses with a linear depth. True if you want to use a log scale.

    Returns:
    tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - dose_incidence (array): dose deposited by electrons [eV/atom].
    """
    start_time = time.time()

    path_dose = constants_Ganymede.WORK_DIR_DOSE    
    delta_t = delta_t_yr*constants_Ganymede.SECOND_IN_YEAR #s

    print(f"DOSE CALCULATION for {region} region")
    # the log scale is used to plot doses without sputtering and have a good resolution for the lowest depth.
    # to compute doses with sputtering, we need a linear scale to have layers of the same depth.
    if log_scale:
        depth = np.logspace(np.log10(1e-6), np.log10(80), 2000)
        extension = '_log.txt'
    else:
        depth = np.linspace(1e-6, 80, 50000)
        extension = '.txt'
    
    dose = np.zeros(len(depth))

    energy_INT_keV = np.logspace(np.log10(1), np.log10(1e5), 15000) # from 1 keV to 100 MeV

    energy_INT_keV_elec, flux_divE_INT = upload.electron_flux_data(region, measure)
    ESTAR_energy_complete, ESTAR_stopping_power = upload.ESTAR_data("LET")

    LET_list = np.interp(energy_INT_keV, ESTAR_energy_complete, ESTAR_stopping_power)    # keV/µm
    """
    plt.plot(energy_INT_keV, LET_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    """

    dose_step = np.zeros(len(depth))

    for i in range(0, len(energy_INT_keV_elec)-1):
        E = energy_INT_keV_elec[i]*1e3   # eV
        #print(f'E = {E} eV')
        for j in range(0, len(depth)-1):
            if (E<1e3): # if E<1 keV
                break
            #index_E = energy_INT.index(E)   # eV
            index_E = np.argmin(np.abs(energy_INT_keV - E*1e-3))
                #print('index_E = ', index_E)
            LET = LET_list[index_E] * 1e7      # eV/cm

            dose_step[j] = constants_Ganymede.C * constants_Ganymede.M * delta_t / (3*density*constants_Ganymede.Na) * LET * flux_divE_INT[i] * (energy_INT_keV_elec[i+1] - energy_INT_keV_elec[i])

            dose[j] = dose[j] + dose_step[j]
            #print('dose[j] = ', dose[j])
            #print('i = ', i)
            #print('j = ', j)

            E = E - LET*(depth[j+1] - depth[j])
        
    dose_incidence = compute_dose_incidence(dose, depth)
    print(f'dose = {dose}')
    print(f'dose_incidence = {dose_incidence}')

    file_dose = os.path.join(path_dose, f'dose_electrons_{region}_{measure}_{delta_t_yr:.2e}yr{extension}')
    write_file(file_dose, depth, dose_incidence)

    print(f'Total dose calculated for region {region}. Saved in file dose_electrons_{region}_{measure}_{delta_t_yr:.2e}yr{extension}')

    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(' ')

    return depth, dose_incidence


def change_time_doses(dose_ref_path, delta_t_old, delta_t_new):
    """
    Compute the dose on a time delta_t_new using the dose computed on a time delta_t_old, since the dose is proportionnal to the time.

    Parameters:
        dose_ref_path (path): path of the dose array calculated on a delta_t_old time.
        delta_t_old (float): time on which doses where calculated [yr].
        delta_t_new (float): time on which doses will be calculated [yr].

    Returns:
        dose_new (array): dose calculated on a delta_t_new time [eV/atom].
    """

    data = np.loadtxt(dose_ref_path, delimiter='	')
    dose_old = data[:, 1]
    #print (f'first elem of dose_old = {dose_old[0]}')

    dose_new = (dose_old/delta_t_old) * delta_t_new
    #print(f'delta_t_new = {delta_t_new}')
    #print (f'first elem of dose_new = {dose_new[0]}')

    return dose_new


def determine_steady_state(erosion_rate_1yr, penetration_depth):
    """
    Estimate the time from which a steady state is obtained between deposited energy and erosion of the surface.

    Parameters:
        erosion_rate_1yr (float): erosion rate of the surface due to sputtering [nm/yr].
        penetration_depth (float): maximum of depth reached by the incident particles [cm].

    Returns:
        time_steady_state (float): time required to reach the steady state [yr].
    """

    time_steady_state = penetration_depth / erosion_rate_1yr
    print(f'steady state is reached in {time_steady_state:.2e} yr')

    if(time_steady_state > 4.55e9):
        print(f'The time to reach the steady state is higher than the age of the Universe.')
        time_steady_state = 4.55e9
        print(f'Computing on {time_steady_state} yr.')

    return time_steady_state


def compute_doses_ions_sput(region, Temp, density = 0.93, sput = "O2"):
    """
    Compute the dose deposited by ions as a function of depth, taking surface erosion due to sputtering into account.

    Parameters:
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        Temp (float): surface temperature [K].
        density (float, by default = 0.93): density of water ice [g/cm^3].
        sput (string, by default = "O2"): can be "O2" or "H2O", erosion due to sputtering of O2 molecules only ("O2") or by all the radiolysis products ("H2O").

    Returns:
        tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - dose_e_sput_total (array): electronic dose deposited by ions considering sputtering [eV/atom].
            - dose_n_sput_total (array): nuclear dose deposited by ions considering sputtering [eV/atom].
            - dose_sput_total (array): total dose (electronic + nuclear) deposited by ions considering sputtering [eV/atom].
    """

    start_time = time.time()

    path_dose = constants_Ganymede.WORK_DIR_DOSE
    print(f"DOSE CALCULATION WITH SPUTTERING for {region} region at temperature {Temp} K")

    depth = np.linspace(1e-6, 1.4e-1, 50000)
    dose_e_sput, dose_n_sput, dose_e_sput_total, dose_n_sput_total, dose_sput_total = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))

    # definition of time and depth steps
    erosion_1yr_H2O, erosion_1yr_O2, penetration_depth = erosion.compute_erosion_ion(Temp, region, density) # [nm/yr] surface erosion on 1 yr
    erosion_1yr_O2_elec, _ = erosion.compute_erosion_electron(Temp, "20N60N", "G8", density)

    if (sput == "O2"):
        erosion_1yr = (erosion_1yr_O2+erosion_1yr_O2_elec) * 1e-7
    else:
        erosion_1yr = erosion_1yr_H2O * 1e-7
    delta_x = depth[1] - depth[0]
    print(f'delta_x = {delta_x} cm, thickness of the first layer.')
    delta_t_yr = determine_steady_state(erosion_1yr, penetration_depth)
    delta_t = delta_t_yr * constants_Ganymede.SECOND_IN_YEAR #s
    time_step = delta_x / erosion_1yr
    print(f'time_step = {time_step} yr.')

    if (time_step > delta_t_yr):
        raise ValueError("STOP the time step is higher than the computation time.")
    
    compute_doses_ions(region, 4.55e9, density)
    for element in constants_Ganymede.ELEM_LISTE:
        elem = element.name
        start_time_elem = time.time()
        print(f"Element {elem}")
        file_dose_e_ref = os.path.join(path_dose, f"dose_e_{elem}_{region}_{4.55e9:.2e}yr.txt")
        file_dose_n_ref = os.path.join(path_dose, f"dose_n_{elem}_{region}_{4.55e9:.2e}yr.txt")
        #if os.path.isfile(file_dose_e_ref) and os.path.isfile(file_dose_n_ref):
            #print(f"Files 'dose_e_{elem}_{region}_{4.55e9:.2e}yr.txt' and 'dose_n_{elem}_{region}_{4.55e9:.2e}yr.txt' exist.")
        #else:
            #print(f"Files 'dose_e_{elem}_{region}_{4.55e9:.2e}yr.txt' and 'dose_n_{elem}_{region}_{4.55e9:.2e}yr.txt' do not exist.")
            #compute_doses_ions(region, 4.55e9, density)  # compute the doses on 4.55 yr for all the chemical elements

        dose_e_elem = change_time_doses(file_dose_e_ref, 4.55e9, delta_t_yr)
        dose_n_elem = change_time_doses(file_dose_n_ref, 4.55e9, delta_t_yr)
        #print("dose_e_elem = ", dose_e_elem)
        write_file(os.path.join(path_dose, f"dose_e_{elem}_{region}_{delta_t_yr:.2e}yr.txt"), depth, dose_e_elem)
        write_file(os.path.join(path_dose, f"dose_e_{elem}_{region}_{delta_t_yr:.2e}yr.txt"), depth, dose_n_elem)
        dose_e_rate = dose_e_elem / delta_t # eV/atom/s
        dose_n_rate = dose_n_elem / delta_t # eV/atom/s
        #print("dose_e_rate = ", dose_e_rate)

        t = 0
        while (t < delta_t_yr):
            dose_e_sput = dose_e_sput + dose_e_rate*time_step*constants_Ganymede.SECOND_IN_YEAR
            #print("dose_e_sput[0] = ", dose_e_sput[0])
            dose_n_sput = dose_n_sput + dose_n_rate*time_step*constants_Ganymede.SECOND_IN_YEAR

            dose_e_sput = np.roll(dose_e_sput, -1)  # remove the first layer due to sputtering
            dose_e_sput[-1] = 0
            dose_n_sput = np.roll(dose_n_sput, -1)
            dose_n_sput[-1] = 0
            
            t += time_step
            #print (f't = {t} yr')

        # save the doses e and n for elem
        file_dose_e_elem = f"{path_dose}\\dose_e_SPUT_{sput}_{elem}_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt"
        file_dose_n_elem = f"{path_dose}\\dose_n_SPUT_{sput}_{elem}_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt"
        write_file(file_dose_e_elem, depth, dose_e_sput)
        write_file(file_dose_n_elem, depth, dose_n_sput)

        dose_e_sput_total += dose_e_sput 
        dose_n_sput_total += dose_n_sput 

        end_time_elem = time.time()
        execution_time_elem = (end_time_elem - start_time_elem)/60

        print(f"Execution time for elem {elem}: {execution_time_elem} minutes")
        print(' ')
    
    dose_sput_total = dose_e_sput_total + dose_n_sput_total

    # save the doses e and n for region, and the total dose
    file_dose_e = f"{path_dose}\\dose_e_SPUT_{sput}_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt"
    file_dose_n = f"{path_dose}\\dose_n_SPUT_{sput}_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt"
    file_dose = f"{path_dose}\\dose_SPUT_{sput}_TOTAL_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt"
    write_file(file_dose_e, depth, dose_e_sput_total)
    write_file(file_dose_n, depth, dose_n_sput_total)
    write_file(file_dose, depth, dose_sput_total)

    print(f'Electronic dose (De) calculated for region {region}. Saved in file dose_e_SPUT_{sput}_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt')
    print(f'Nuclear dose (Dn) calculated for region {region}. Saved in file dose_n_SPUT_{sput}_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt')
    print(f'Total dose (De + Dn) calculated for region {region}. Saved in file dose_SPUT_{sput}_TOTAL_{region}_{Temp}K_{delta_t_yr:.2e}yr.txt')

    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(' ')

    return depth, dose_e_sput_total, dose_n_sput_total, dose_sput_total




def compute_doses_electrons_sput(region_ion, region_electron, Temp, measure = "G8", density = 0.93, sput = "O2"):
    """
    Compute the dose deposited by electrons as a function of depth, taking surface erosion due to sputtering into account.

    Parameters:
        region_ion (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        region_electron (string): Ganymede's regions as presented in Liuzzo+2020, can be "60N90N", "20N60N", "20N20S", "20S60S", "60S90S".
        Temp (float): surface temperature [K].
        measure (string, by default = "G8"): number of the Galileo encounter which have measured the energy flux around Ganymede, can be "G8", "G1" or "G28".
        density (float, by default = 0.93): density of water ice [g/cm^3].
        sput (string, by default = False): can be "O2" or "H2O", erosion due to sputtering of O2 molecules only (O2), or by all the radiolysis products (H2O).
    
    Returns:
        tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - dose_sput (array): dose deposited by electrons considering sputtering [eV/atom].
    """

    start_time = time.time()

    path_dose = constants_Ganymede.WORK_DIR_DOSE
    print(f"DOSE CALCULATION WITH SPUTTERING for {region_electron} and {region_ion} regions at temperature {Temp} K")

    depth = np.linspace(1e-6, 80, 50000)
    dose_sput = np.zeros(len(depth))

    # definition of time and depth steps
    erosion_1yr_H2O, erosion_1yr_O2, _ = erosion.compute_erosion_ion(Temp, region_ion, density) # [nm/yr] surface erosion on 1 yr
    erosion_1yr_O2_elec, penetration_depth = erosion.compute_erosion_electron(Temp, region_electron)
    if (sput == "O2"):
        erosion_1yr = (erosion_1yr_O2+erosion_1yr_O2_elec) * 1e-7
    else:
        erosion_1yr = erosion_1yr_H2O * 1e-7

    delta_t_yr = determine_steady_state(erosion_1yr, penetration_depth)
    delta_t = delta_t_yr * constants_Ganymede.SECOND_IN_YEAR #s
    delta_x = depth[1] - depth[0]
    print(f'delta_x = {delta_x} cm, thickness of the first layer.')
    time_step = delta_x / erosion_1yr
    print(f'time_step = {time_step} yr.')

    if (time_step > delta_t_yr):
        raise ValueError("STOP the time step is higher than the computation time.")

    compute_doses_electrons(region_electron, 4.55e9, density = density)  # compute the doses on 4.55 yr for all the elements
    file_dose_ref = os.path.join(path_dose, f"dose_electrons_{region_electron}_{measure}_{4.55e9:.2e}yr.txt")
    #if os.path.isfile(file_dose_ref):
        #print(f"File 'dose_electrons_{region_electron}_{measure}_{4.55e9:.2e}yr.txt' exists.")
    #else:
        #print(f"File 'dose_electrons_{region_electron}_{measure}_{4.55e9:.2e}yr.txt' does not exist.")
        #compute_doses_electrons(region_electron, 4.55e9, density = density)  # compute the doses on 4.55 yr for all the elements

    dose = change_time_doses(file_dose_ref, 4.55e9, delta_t_yr)
    write_file(os.path.join(path_dose, f"dose_electrons_{region_electron}_{delta_t_yr:.2e}yr.txt"), depth, dose)
    dose_rate = dose / delta_t # eV/atom/s

    t = 0
    while (t < delta_t_yr):
        dose_sput = dose_sput + dose_rate*time_step*constants_Ganymede.SECOND_IN_YEAR

        dose_sput = np.roll(dose_sput, -1)  # remove the first layer due to sputtering
        dose_sput[-1] = 0
            
        t += time_step
        #print (f't = {t} yr')

    file_dose = os.path.join(path_dose, f"dose_electrons_SPUT_{sput}_{region_electron}_{region_ion}_{measure}_{Temp}K_{delta_t_yr:.2e}yr.txt")
    write_file(file_dose, depth, dose_sput)

    print(f'Total dose calculated for regions {region_electron} and {region_ion}. Saved in file dose_electrons_SPUT_{sput}_{region_electron}_{region_ion}_{measure}_{Temp}K_{delta_t_yr:.2e}yr.txt')

    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(' ')

    return depth, dose_sput




if __name__ == "__main__":
    """
    compute_doses_electrons('60S90S', 4.55e9)
    compute_doses_electrons('20S60S', 4.55e9)
    compute_doses_electrons('20N20S', 4.55e9)
    compute_doses_electrons('20N60N', 4.55e9)
    compute_doses_electrons('60N90N', 4.55e9)
    

    compute_doses_ions('polar', 4.55e9)
    compute_doses_ions('leading', 4.55e9)
    compute_doses_ions('trailing', 4.55e9)

    compute_doses_ions_sput('polar', 90)
    compute_doses_ions_sput('polar', 120)

    compute_doses_ions_sput('leading', 90)
    compute_doses_ions_sput('leading', 155)
    
    compute_doses_ions_sput('trailing', 90)
    compute_doses_ions_sput('trailing', 155)

    compute_doses_ions_sput('polar', 90, sput = "H2O")
    compute_doses_ions_sput('polar', 120, sput = "H2O")

    compute_doses_ions_sput('leading', 90, sput = "H2O")
    compute_doses_ions_sput('leading', 155, sput = "H2O")
    
    compute_doses_ions_sput('trailing', 90, sput = "H2O")
    compute_doses_ions_sput('trailing', 155, sput = "H2O")


    compute_doses_electrons_sput('leading', '20N60N', 90)
    compute_doses_electrons_sput('leading', '20N60N', 155)

    compute_doses_electrons_sput('trailing', '20N60N', 90)
    compute_doses_electrons_sput('trailing', '20N60N', 155)

    compute_doses_electrons_sput('polar', '60N90N', 90)
    compute_doses_electrons_sput('polar', '60N90N', 120)

    compute_doses_electrons_sput('leading', '20N60N', 90, sput = "H2O")
    compute_doses_electrons_sput('leading', '20N60N', 155, sput = "H2O")

    compute_doses_electrons_sput('trailing', '20N60N', 90, sput = "H2O")
    compute_doses_electrons_sput('trailing', '20N60N', 155, sput = "H2O")

    compute_doses_electrons_sput('polar', '60N90N', 90, sput = "H2O")
    compute_doses_electrons_sput('polar', '60N90N', 120, sput = "H2O")
    """
    #print(doses_incidence_i(np.array([10, 9, 8, 7, 6]), np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6]), 80))
    #print(compute_dose_incidence(np.array([10, 9, 8, 7, 6]), np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])))

    #compute_doses_ions("leading", 4.55e9, density = 0.93, log_scale = False)
    #compute_doses_ions("leading", 4.55e9, density = 0.93, log_scale = True)



    

    
