import numpy as np
import matplotlib.pyplot as plt
import time
import os

import upload_data as upload
import constants_Ganymede
from utils import write_file
import erosion_thickness as erosion
import dose


"""
Created on 2025/07/16 09:29:59

The code computes the water ice amorphous fraction on Ganymede, for the different regions (poles, leading, trailing).
The amorphous fracion Phi_A is computed as phiA = 1 - exp(-sigma*fluence), where sigma is the amorphization cross-section and fluence = flux*time.
sigma is computed according to results of irradiation experiments (Moingeon+2025, in prep), as a function of the electronic stopping power of the ion (Se) and the temperature (Temp).
The surface erosion due to sputtering is also taking into account.
Erosion rate were previously estimated and included in the computation, so the surface is continually renewed and deposited energy can not accumulate at the top of the surface.
Two scenarios are considered to compute the amorphous fraction:
    (1): sigma(Temp) is computed, and no crystallization term is added in the differential equation: d(Phi_A)/dt = sigma*flux*(1-Phi_A).
    (2): sigma(40K) is computed, and a crystallization term at temperature Temp is added: d(Phi_A)/dt = sigma*flux - (1/tc + sigma*flux)*Phi_A.

@author: audrey.moingeon@univ-grenoble-alpes.fr
"""


def locate_value(value, list):
    """
    Locate a value or the closest value in a list giving its index.

    Parameters:
        value (float): value of the element searched in the list.
        list (array): array in which to find the element closest from "value".

    Returns:
        first_index (float): index of the value in the list.
    """
    # Create a bool list
    exceeds_threshold = list > value

    # Find the position of the first element higher than "value"
    first_index = np.argmax(exceeds_threshold)

    'print(f"The position of the first element higher than {value} is: {first_index}")'
    'print("index = ", first_index)'

    return first_index




####################################################################################################################
#### Compute the amorphization cross-section sigma according to experimental results, at 40, 90, 100 and 110 K. ####
####################################################################################################################
def compute_sigma_according_Temp(Se, Set, a0):
    """"
    Compute the amorphization cross-section according to expressions provided by the analytical Thermal Spike model (Szenes+1995).

    Parameters:
        Se (float): electronic stopping power of the incident ion [eV/cm].
        Set (float): threashold value of Se below which amorphization is not expected, determined by fitting experimental data [eV/cm].
        a0 (float): parameter obtained by fitting experimental data [cm].

    Returns:
        sigma_am (float): amorphization cross-section [cm²].
    """

    if (Se<2.7*Set):
        sigma_am = np.pi*a0**2 * np.log(Se/Set)
    else:
        sigma_am = (np.pi*a0**2*Se)/(2.7*Set)
    
    return sigma_am


def compute_sigma_uncertainties(a0, delta_a0, Set, delta_Set, Se):
    """
    Compute the uncertainties on the amorphization cross-section.

    Parameters:
        a0 (float): parameter obtained by fitting experimental data [cm].
        delta_a0 (float): uncertainties on a0 [cm].
        Set (float): threashold value of Se below which amorphization is not expected, determined by fitting experimental data [eV/cm].
        delta_Set (float): uncertainties on Set [eV/cm].
        Se (float): electronic stopping power of the incident ion [eV/cm].

    Returns:
        delta_sigma_am (float): uncertainties on the amorphization cross-section value [cm²].
    """
    A = 2*np.pi*a0*np.log(Se/Set)*delta_a0
    B = (np.pi*a0**2*delta_Set)/Set

    delta_sigma_am = np.sqrt(A**2 + B**2)

    return delta_sigma_am


def compute_sigma(Se, T):
    """
    Compute amorphization cross-section according to fitting parameters for temperatures between 40 and 110 K (Moingeon et al, submitted).

    Parameters:
        Se (float): electronic stopping power of the incident ion [eV/cm].
        T (float): surface temperature [K].

    Returns:
        tuple (tuple):
            - sigma_am (float): amorphization cross-section [cm²].
            - sigma_am_up (float): upper limit of the amorphization cross-section [cm²].
            - sigma_am_down (float): lower limit of the amorphization cross-section [cm²].
    """
    # fitting parameters determined in Moingeon et al. submitted for different temperatures
    if(T==40):
        Set = 1.36e8
        delta_Set = 1.64e8
        a0 = 1.44e-7
        delta_a0 = 0.87e-7
    elif(T==90):
        Set = 3.83e8
        delta_Set = 2.32e8
        a0 = 2.06e-7
        delta_a0 = 0.61e-7
    elif(T==100):
        Set = 5.38e8
        delta_Set = 1.69e8
        a0 = 2.06e-7
        delta_a0 = 0.32e-7
    elif(T==110):
        Set = 6.79e8
        delta_Set = 1.64e8
        a0 = 1.72e-7
        delta_a0 = 0.20e-7
    else:
        raise ValueError("Error with the temperature: must be 40; 90; 100 or 110 K.")
    
    sigma_am = compute_sigma_according_Temp(Se, Set, a0)
    sigma_am_up = compute_sigma_according_Temp(Se, Set, a0) + compute_sigma_uncertainties(a0, delta_a0, Set, delta_Set, Se)
    sigma_am_down = compute_sigma_according_Temp(Se, Set, a0) - compute_sigma_uncertainties(a0, delta_a0, Set, delta_Set, Se)

    # sigma_am < 0 is not physical
    if (sigma_am < 0):
        sigma_am = 0
    
    if (sigma_am_up < 0):
        sigma_am_up = 0
    
    if (sigma_am_down < 0):
        sigma_am_down = 0
    
    #print("sigma_am = ", sigma_am)
    #print("sigma_am_up = ", sigma_am_up)
    #print("sigma_am_down = ", sigma_am_down)
    
    return sigma_am, sigma_am_up, sigma_am_down


def compute_sigma_flux(depth, region, elem, Temp):
    """
    Compute amorphization cross-section * flux according to the Ganymede's region, the ion specie and the temperature.

    Parameters:
        depth (array): depth on which amorphous fraction will be estimated [cm].
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        elem (string): chemical element impacting Ganymede's surface, see the class ChemicalElement in constants.py.
        Temp (float): surface temperature [K].

    Returns:
        tuple (tuple):
            - sigma_flux_elem (array): multiplication of the amorphization cross-section by the ion flux [ions/s].
            - sigma_flux_elem_up (array): upper limit [ions/s].
            - sigma_flux_elem_down (array): lower limit [ions/s].
    """

    energy_INT_eV_ions, flux_INT = upload.ion_flux_data(region, elem)
    flux_divE_INT = flux_INT / energy_INT_eV_ions
    SRIM_energy, SRIM_Se, SRIM_Sn, _ = upload.SRIM_data(elem)

    energy_INT_eV = np.logspace(np.log10(10), np.log10(1e8), 15000) # from 10 eV to 100 MeV
    energy_INT_keV = energy_INT_eV * 1e-3

    Se_INT = np.interp(energy_INT_keV, SRIM_energy, SRIM_Se)
    Sn_INT = np.interp(energy_INT_keV, SRIM_energy, SRIM_Sn)
    #print ("Se_INT = ", Se_INT)
    #plt.plot(energy_INT_keV, Se_INT, label='INT curves', linewidth=4)
    #plt.plot(SRIM_energy, SRIM_Se, label='initial curves')
    #plt.show()

    sigma_flux_elem = np.zeros(len(depth))
    sigma_flux_step = np.zeros(len(depth))
    sigma_flux_elem_up = np.zeros(len(depth))
    sigma_flux_step_up = np.zeros(len(depth))
    sigma_flux_elem_down = np.zeros(len(depth))
    sigma_flux_step_down = np.zeros(len(depth))

    for i in range(1, len(energy_INT_eV_ions)-1):
        E = energy_INT_eV_ions[i]
        #print ("E = ", E, "eV")
        for x in range(0, len(depth)-1):
            #print ("E = ", E, "eV")
            if(E<10):   # when ion energy < 10 eV, it is considered implanted
                print(f"implantation depth = {depth[x]} cm")   # should be coherent with the projected range given by SRIM.
                break
            index_E = locate_value(E, energy_INT_eV)
            Se = Se_INT[index_E] * 1e7
            Sn = Sn_INT[index_E] * 1e7

            sigma_am, sigma_am_up, sigma_am_down = compute_sigma(Se, Temp)

            sigma_flux_step[x] = sigma_am * flux_divE_INT[i] * (energy_INT_eV_ions[i+1] - energy_INT_eV_ions[i])
            sigma_flux_elem[x] = sigma_flux_elem[x] + sigma_flux_step[x]

            sigma_flux_step_up[x] = sigma_am_up * flux_divE_INT[i] * (energy_INT_eV_ions[i+1] - energy_INT_eV_ions[i])
            sigma_flux_elem_up[x] = sigma_flux_elem_up[x] + sigma_flux_step_up[x]

            sigma_flux_step_down[x] = sigma_am_down * flux_divE_INT[i] * (energy_INT_eV_ions[i+1] - energy_INT_eV_ions[i])
            sigma_flux_elem_down[x] = sigma_flux_elem_down[x] + sigma_flux_step_down[x]

            E = E - abs(Se+Sn)*(depth[x+1] - depth[x])
    """
    plt.plot(depth, sigma_flux_elem, label='sigma_flux_'+elem)
    plt.xscale('log')
    "plt.yscale('log')"
    plt.xlabel('depth')
    plt.ylabel('sigma*flux')
    plt.legend()
    plt.show()
    print ("Computation of ", elem, " on ", region, " is done.")
    """

    # Save in a txt file
    sigma_flux_name = os.path.join(constants_Ganymede.WORK_DIR_AMORPH, f"sigma_flux_{region}_{elem}_{Temp}K.txt")
    write_file(sigma_flux_name, sigma_flux_elem, sigma_flux_elem_up, sigma_flux_elem_down)
    
    print (f"sigma*flux computed on {region} for elem {elem} at temperature {Temp} K.")
    print (f"Data are saved in file {sigma_flux_name}.")
    
    return sigma_flux_elem, sigma_flux_elem_up, sigma_flux_elem_down


def compute_sigma_flux_all_elem(depth, region, Temp):
    """
    Compute amorphization cross-section * flux for all the ion species of a region, for a given temperature.

    Parameters:
        depth (array): depth on which amorphous fraction will be estimated [cm].
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        Temp (float): surface temperature [K].

    Returns:
        tuple (tuple):
            - sigma_flux_elem (array): multiplication of the amorphization cross-section by the ion flux [ions/s].
            - sigma_flux_elem_up (array): upper limit [ions/s].
            - sigma_flux_elem_down (array): lower limit [ions/s].
    """

    sigma_flux, sigma_flux_up, sigma_flux_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))

    # compute the sum of sigma_flux and its uncertainties of each ion specie for one given region
    for elem in constants_Ganymede.ELEM_LISTE:
        sigma_flux_elem, sigma_flux_elem_up, sigma_flux_elem_down = compute_sigma_flux(depth, region, elem.name, Temp)
        sigma_flux += sigma_flux_elem
        sigma_flux_up += sigma_flux_elem_up
        sigma_flux_down += sigma_flux_elem_down
    
    file_name = os.path.join(constants_Ganymede.WORK_DIR_AMORPH, f"sigma_flux_{region}_{Temp}K.txt")
    write_file(file_name, sigma_flux, sigma_flux_up, sigma_flux_down)
    
    print (f"sigma*flux computed on {region} at temperature {Temp} K.")
    print (f"Data are saved in file {file_name}.")
    
    return sigma_flux, sigma_flux_up, sigma_flux_down


def compute_phiC_K94(Temp):
    """
    Compute the crystallization time of amorphous water ice depending on temperature, according to Kouchi+1994.

    Parameters:
        Temp (float): surface temperature [K].

    Returns:
        tc (float): time required to recrystallize amorphous water ice [s].
    """

    alpha = 2 # [no unit] geometrical factor
    S = 1e-21 # [J]
    omega = 3.25e-29 # [m^3]
    D0 = 6.1e-7 # [m²/s]
    kb = 1.38e-23 # [J/K]
    Ea = 7.36e-20 # [J]
    L = 2.6e-21 # [J/molecule]
    Tm = 273 # [K]
	
    a = (4*np.pi*S**3)/(3*L**2)
    tc = (1/(2*np.pi*alpha))**(1/4) * (kb*Temp/S)**(1/8) * (omega**(2/3))/D0 * np.exp((1/(kb*Temp))*(Ea + a*(Tm/(Tm-Temp))**2))

    print(f"tc = {tc / (3600*24)} days at {Temp} K")
	
    return tc


def f_exp(sigma_flux, amorph_move):
    """
    Differential equation to compute amorphous fraction without recrystallization term.

    Parameters:
        sigma_flux (array): multiplication of the amorphization cross-section by the ion flux [ions/s].
        amorph_move (array): amorphous fraction.

    Returns:
        phiA = amorphous fraction at the next iteration.
    """
    phiA = sigma_flux * (1 - amorph_move)

    return phiA


def f_recryst(sigma_flux, amorph_move, tc):
    """
    Differential equation to compute amorphous fraction with recrystallization term.

    Parameters:
        sigma_flux (array): multiplication of the amorphization cross-section by the ion flux [ions/s].
        amorph_move (array): amorphous fraction.
        tc (float): time required to recrystallize amorphous water ice [s].

    Returns:
        phiA = amorphous fraction at the next iteration.
    """
    phiA = sigma_flux - ((1/tc) + sigma_flux) * amorph_move

    return phiA


## FIRST CASE: where Phi_A is computed using sigma(Temp), without rerystallization.
def compute_amorph(region, Temp, pts_RK2, f = f_exp, t0 = 0):
    """
    Compute amorphous fraction on Ganymede's surface using a Runge-Kutta method of order 2, considering erosion due to sputtering.

    Parameters:
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        Temp (float): surface temperature [K].
        pts_RK2 (int): number of points to define the depth step for the RK2 method.
        f (function, by default = f_exp): function used to describe the evolution of amorphization process.
        t0 (float, by default = 0): start of time for computation [yr].
    
    Returns:
        tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - amorph_move (array): amorphous fraction as a function of depth.
            - amorph_move_up (array): upper limit.
            - amorph_move_down (array): lower limit.
    """

    start_time = time.time()

    print (f'sigma_flux_{region}_{Temp}K.txt')
    depth = np.linspace(1e-6, 1.4e-1, 9000)
    sigma_flux, sigma_flux_up, sigma_flux_down = compute_sigma_flux_all_elem(depth, region, Temp)
    # data = np.loadtxt('sigma_flux_' + region + '_' + str(Temp) + 'K.txt', delimiter='\t') # in case sigma_flux files are already computed
    #sigma_flux = data [:, 0]
    #sigma_flux_up = data [:, 1]
    #sigma_flux_down = data [:, 2]

    _, erosion_1yr_O2, penetration_depth = erosion.compute_erosion_ion(Temp, region) # [nm/yr] surface erosion on 1 yr
    erosion_1yr_cm = erosion_1yr_O2 * 1e-7
    delta_x = depth[1] - depth[0]
    print(f'delta_x = {delta_x} cm, thickness of the first layer.')
    tf_yr = dose.determine_steady_state(erosion_1yr_cm, penetration_depth)
    time_step = delta_x / erosion_1yr_cm
    print(f'time_step = {time_step} yr.')
    

    # To use the RK2 method, the time step should be small enough. We define h_RK2 = h / pts_RK2 to compute amorph_move as a function of depth.
    h = delta_x / erosion_1yr_cm
    h_RK2 = h / pts_RK2
    print ("delta_x = ", delta_x, " cm")
    print ("h = ", h, " yr")
    print ("h_RK2 = ", h_RK2, " yr")
    t = np.arange(t0, tf_yr, h)
   
    amorph_move, amorph_move_up, amorph_move_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))
    k1, k2, k1_up, k2_up, k1_down, k2_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))
    #test = np.zeros(len(t))
    #phiA_theorique = np.zeros(len(t))

    j = 0
    for i in range(1, len(t)):
        #if(i>5):   # to test on the first iterations if the RK2 method converges
            #break
        
        # resolution of the differential equation d(Phi_A)/dt = f(Phi_A, t) with the RK2 method
        while(j < h/h_RK2):
            k1 = f(sigma_flux, amorph_move)
            k2 = f(sigma_flux, amorph_move + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1)
            amorph_move = amorph_move + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1 + k2)

            k1_up = f(sigma_flux_up, amorph_move_up)
            k2_up = f(sigma_flux_up, amorph_move_up + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1_up)
            amorph_move_up = amorph_move_up + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1_up + k2_up)

            k1_down = f(sigma_flux_down, amorph_move_down)
            k2_down = f(sigma_flux_down, amorph_move_down + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1_down)
            amorph_move_down = amorph_move_down + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1_down + k2_down)

            j+=1
        #print ("t = ", i*h)
        j=0
        #print ("amorph_move_up[0] = ", amorph_move_up[0])
        #test[i] = amorph_move[0]
        #phiA_theorique[i] = 1 - np.exp(-sigma_flux[3200]*i*h*constants.SECOND_IN_YEAR)

        #delete the first water ice layer due to sputtering, and add a new one at the end, which was never irradiated
        amorph_move = np.roll(amorph_move, -1)
        amorph_move[-1] = 0

        amorph_move_up = np.roll(amorph_move_up, -1)
        amorph_move_up[-1] = 0

        amorph_move_down = np.roll(amorph_move_down, -1)
        amorph_move_down[-1] = 0


    file_name = os.path.join(constants_Ganymede.WORK_DIR_AMORPH, f'amorph_move_python_{region}_{Temp}K_{tf_yr:.2e}yr_SF{len(depth)}_RK2_{pts_RK2}pts.txt')
    write_file(file_name, depth, amorph_move, amorph_move_up, amorph_move_down)
    
    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(" ")

    return depth, amorph_move, amorph_move_up, amorph_move_down



## SECOND CASE: where Phi_A is computed using sigma(40 K), with rerystallization.
def compute_amorph_recryst(region, Temp, pts_RK2, f = f_recryst, t0 = 0):
    """
    Compute amorphous fraction on Ganymede's surface using a Runge-Kutta method of order 2, considering erosion due to sputtering and recrystallization.

    Parameters:
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        Temp (float): surface temperature [K].
        pts_RK2 (int): number of points to define the depth step for the RK2 method.
        f (function, by default = f_exp): function used to describe the evolution of amorphization process.
        t0 (float, by default = 0): start of time for computation [yr].
    
    Returns:
        tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - amorph_move (array): amorphous fraction as a function of depth.
            - amorph_move_up (array): upper limit.
            - amorph_move_down (array): lower limit.
    """
    start_time = time.time()

    print (f'sigma_flux_{region}_{Temp}K_recryst.txt')
    depth = np.linspace(1e-6, 1.4e-1, 9000)
    sigma_flux, sigma_flux_up, sigma_flux_down = compute_sigma_flux_all_elem(depth, region, Temp)
    #data = np.loadtxt('sigma_flux_' + region + '_40K.txt', delimiter='\t') # in case sigma_flux files are already computed
    #sigma_flux = data [:, 0]
    #sigma_flux_up = data [:, 1]
    #sigma_flux_down = data [:, 2]
    tc = compute_phiC_K94(Temp)

    _, erosion_1yr_O2, penetration_depth = erosion.compute_erosion_ion(Temp, region) # [nm/yr] surface erosion on 1 yr
    erosion_1yr_cm = erosion_1yr_O2 * 1e-7
    delta_x = depth[1] - depth[0]
    print(f'delta_x = {delta_x} cm, thickness of the first layer.')
    tf_yr = dose.determine_steady_state(erosion_1yr_cm, penetration_depth)
    time_step = delta_x / erosion_1yr_cm
    print(f'time_step = {time_step} yr.')
    

    # To use the RK2 method, the time step should be small enough. We define h_RK2 = h / pts_RK2 to compute amorph_move as a function of depth.
    h = delta_x / erosion_1yr_cm
    h_RK2 = h / pts_RK2
    print ("delta_x = ", delta_x, " cm")
    print ("h = ", h, " yr")
    print ("h_RK2 = ", h_RK2, " yr")
    t = np.arange(t0, tf_yr, h)
    #print (t)
    
    amorph_move, amorph_move_up, amorph_move_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))
    k1, k2, k1_up, k2_up, k1_down, k2_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))
    #verif = np.zeros(len(t))
    #phiA_theorique = np.zeros(len(t))

    j = 0
    for i in range(1, len(t)):
        #if(i>5):   # to test on the first iterations if the RK2 method converges
            #break
        
        # resolution of the differential equation d(Phi_A)/dt = f(Phi_A, t) with the RK2 method
        while(j < h/h_RK2):
            k1 = f(sigma_flux, amorph_move, tc)
            k2 = f(sigma_flux, amorph_move + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1, tc)
            amorph_move = amorph_move + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1 + k2)

            k1_up = f(sigma_flux_up, amorph_move_up, tc)
            k2_up = f(sigma_flux_up, amorph_move_up + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1_up, tc)
            amorph_move_up = amorph_move_up + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1_up + k2_up)

            k1_down = f(sigma_flux_down, amorph_move_down, tc)
            k2_down = f(sigma_flux_down, amorph_move_down + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1_down, tc)
            amorph_move_down = amorph_move_down + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1_down + k2_down)

            j+=1
        #print ("t = ", i*h)
        j=0
        #print ("amorph_move_up[0] = ", amorph_move_up[0])
        #verif[i] = amorph_move[3300]
        #phiA_theorique[i] = 1 - np.exp(-sigma_flux[3200]*i*h*constants.SECOND_IN_YEAR)

        #delete the first water ice layer due to sputtering, and add a new one at the end, which was never irradiated
        amorph_move = np.roll(amorph_move, -1)
        amorph_move[-1] = 0

        amorph_move_up = np.roll(amorph_move_up, -1)
        amorph_move_up[-1] = 0

        amorph_move_down = np.roll(amorph_move_down, -1)
        amorph_move_down[-1] = 0

    #plt.plot(t, verif)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel('time')
    #plt.ylabel('phiA at layer 1000')
    #plt.legend()
    #plt.show()

    #file_name = os.path.join(constants.WORK_DIR_AMORPH, f'test_python_{region}_{Temp}K_SF{len(depth)}_RK2_{pts_RK2}pts_40Krecryst_verif.txt')
    #write_file(file_name, t, verif)

    file_name = os.path.join(constants_Ganymede.WORK_DIR_AMORPH, f'amorph_move_python_{region}_{Temp}K_SF{len(depth)}_RK2_{pts_RK2}pts_recryst.txt')
    write_file(file_name, depth, amorph_move, amorph_move_up, amorph_move_down)
    
    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(" ")

    return depth, amorph_move, amorph_move_up, amorph_move_down



## SECOND CASE: where Phi_A is computed using sigma(40 K), with rerystallization.
def compute_amorph_40Krecryst(region, Temp, pts_RK2, f = f_recryst, t0 = 0):
    """
    Compute amorphous fraction on Ganymede's surface using a Runge-Kutta method of order 2, considering erosion due to sputtering and recrystallization.

    Parameters:
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        Temp (float): surface temperature [K].
        pts_RK2 (int): number of points to define the depth step for the RK2 method.
        f (function, by default = f_exp): function used to describe the evolution of amorphization process.
        t0 (float, by default = 0): start of time for computation [yr].
    
    Returns:
        tuple (tuple):
            - depth (array): range of depth into the surface [cm].
            - amorph_move (array): amorphous fraction as a function of depth.
            - amorph_move_up (array): upper limit.
            - amorph_move_down (array): lower limit.
    """
    start_time = time.time()

    print (f'sigma_flux_{region}_{Temp}K_40Krecryst.txt')
    depth = np.linspace(1e-6, 1.4e-1, 9000)
    sigma_flux, sigma_flux_up, sigma_flux_down = compute_sigma_flux_all_elem(depth, region, 40)
    #data = np.loadtxt('sigma_flux_' + region + '_40K.txt', delimiter='\t') # in case sigma_flux files are already computed
    #sigma_flux = data [:, 0]
    #sigma_flux_up = data [:, 1]
    #sigma_flux_down = data [:, 2]
    tc = compute_phiC_K94(Temp)

    _, erosion_1yr_O2, penetration_depth = erosion.compute_erosion_ion(Temp, region) # [nm/yr] surface erosion on 1 yr
    erosion_1yr_cm = erosion_1yr_O2 * 1e-7
    delta_x = depth[1] - depth[0]
    print(f'delta_x = {delta_x} cm, thickness of the first layer.')
    tf_yr = dose.determine_steady_state(erosion_1yr_cm, penetration_depth)
    time_step = delta_x / erosion_1yr_cm
    print(f'time_step = {time_step} yr.')

    # To use the RK2 method, the time step should be small enough. We define h_RK2 = h / pts_RK2 to compute amorph_move as a function of depth.
    h = delta_x / erosion_1yr_cm
    h_RK2 = h / pts_RK2
    print ("delta_x = ", delta_x, " cm")
    print ("h = ", h, " yr")
    print ("h_RK2 = ", h_RK2, " yr")
    t = np.arange(t0, tf_yr, h)

    amorph_move, amorph_move_up, amorph_move_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))
    k1, k2, k1_up, k2_up, k1_down, k2_down = np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth)), np.zeros(len(depth))
    #verif = np.zeros(len(t))
    #phiA_theorique = np.zeros(len(t))

    j = 0
    for i in range(1, len(t)):
        #if(i>5):   # to test on the first iterations if the RK2 method converges
            #break
        
        # resolution of the differential equation d(Phi_A)/dt = f(Phi_A, t) with the RK2 method
        while(j < h/h_RK2):
            k1 = f(sigma_flux, amorph_move, tc)
            k2 = f(sigma_flux, amorph_move + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1, tc)
            amorph_move = amorph_move + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1 + k2)

            k1_up = f(sigma_flux_up, amorph_move_up, tc)
            k2_up = f(sigma_flux_up, amorph_move_up + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1_up, tc)
            amorph_move_up = amorph_move_up + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1_up + k2_up)

            k1_down = f(sigma_flux_down, amorph_move_down, tc)
            k2_down = f(sigma_flux_down, amorph_move_down + h_RK2*constants_Ganymede.SECOND_IN_YEAR * k1_down, tc)
            amorph_move_down = amorph_move_down + (h_RK2*constants_Ganymede.SECOND_IN_YEAR / 2) * (k1_down + k2_down)

            j+=1
        #print ("t = ", i*h)
        j=0
        #print ("amorph_move_up[0] = ", amorph_move_up[0])
        #verif[i] = amorph_move[3300]
        #phiA_theorique[i] = 1 - np.exp(-sigma_flux[3200]*i*h*constants.SECOND_IN_YEAR)
        
        #delete the first water ice layer due to sputtering, and add a new one at the end, which was never irradiated
        amorph_move = np.roll(amorph_move, -1)
        amorph_move[-1] = 0

        amorph_move_up = np.roll(amorph_move_up, -1)
        amorph_move_up[-1] = 0

        amorph_move_down = np.roll(amorph_move_down, -1)
        amorph_move_down[-1] = 0

    #plt.plot(t, verif)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlabel('time')
    #plt.ylabel('phiA at layer 1000')
    #plt.legend()
    #plt.show()

    #file_name = os.path.join(constants.WORK_DIR_AMORPH, f'test_python_{region}_{Temp}K_SF{len(depth)}_RK2_{pts_RK2}pts_40Krecryst_verif.txt')
    #write_file(file_name, t, verif)

    file_name = os.path.join(constants_Ganymede.WORK_DIR_AMORPH, f'amorph_move_python_{region}_{Temp}K_SF{len(depth)}_RK2_{pts_RK2}pts_40Krecryst.txt')
    write_file(file_name, depth, amorph_move, amorph_move_up, amorph_move_down)
    
    end_time = time.time()
    execution_time = (end_time - start_time)/60

    print(f"Execution time: {execution_time} minutes")
    print(" ")

    return depth, amorph_move, amorph_move_up, amorph_move_down




if __name__ == '__main__':
    compute_phiC_K94(120)
    """
    # Compute sigma*flux for each region at 90, 100 and 110 K.
    #depth = np.linspace(1e-6, 1.4e-1, 9000)
    #compute_sigma_flux_all_elem(depth, 'polar', 40)
    #compute_sigma_flux_all_elem(depth, 'leading', 40)
    #compute_sigma_flux_all_elem(depth, 'trailing', 40)
    #compute_sigma_flux_all_elem(depth, 'polar', 90)

    compute_sigma_flux_all_elem(depth, 'polar', 90)
    compute_sigma_flux_all_elem(depth, 'polar', 100)
    compute_sigma_flux_all_elem(depth, 'polar', 110)

    compute_sigma_flux_all_elem(depth, 'leading', 90)
    compute_sigma_flux_all_elem(depth, 'leading', 100)
    compute_sigma_flux_all_elem(depth, 'leading', 110)

    compute_sigma_flux_all_elem(depth, 'trailing', 90)
    compute_sigma_flux_all_elem(depth, 'trailing', 100)
    compute_sigma_flux_all_elem(depth, 'trailing', 110)
    """

    
    # Compute amorphous fraction with no recrystallization for each region at 90, 100 and 110 K.
    
    #compute_rk2('polar', 90, 2900)
    #compute_rk2('leading', 90, 5400)
    #compute_rk2('trailing', 90, 2400)

    #compute_rk2('polar', 100, 1500)
    #compute_rk2('leading', 100, 3000)
    #compute_rk2('trailing', 100, 1300)

    #compute_rk2('polar', 110, 800)
    #compute_rk2('leading', 110, 1600)
    #compute_rk2('trailing', 110, 650)
    


    # Compute amorphous fraction using sigma(40 K) with recrystallization for each region at 90, 100 and 110 K.
    
    #compute_rk2('polar', 90, 2900)
    #compute_rk2('polar', 100, 1500)
    #compute_rk2('polar', 110, 800)
    #compute_rk2_40Krecryst('polar', 90, 8000)
    #compute_rk2_40Krecryst('polar', 100, 6000)
    #compute_rk2_40Krecryst('polar', 110, 4500)

    """
    compute_amorph('polar', 90, f_recryst, 0, 9e5, 1.5e-7, 8000)
    compute_amorph('leading', 90, f_recryst, 0, 3e6, 0.46e-7, 10100)
    compute_amorph('trailing', 90, f_recryst, 0, 4e7, 4e-9, 3500)

    compute_amorph('polar', 100, f_recryst, 0, 9e5, 2e-7, 6000)
    compute_amorph('leading', 100, f_recryst, 0, 3e6, 0.62e-7, 7500)
    compute_amorph('trailing', 100, f_recryst, 0, 4e7, 5.5e-9, 3000)

    compute_amorph('polar', 110, f_recryst, 0, 9e5, 2.8e-7, 4500)
    compute_amorph('leading', 110, f_recryst, 0, 3e6, 0.88e-7, 5500)
    compute_amorph('trailing', 110, f_recryst, 0, 4e7, 7.7e-9, 2000)
    """