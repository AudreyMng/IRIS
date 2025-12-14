import numpy as np
import matplotlib.pyplot as plt

import upload_data as upload
from constants_Ganymede import * #ChemicalElement, SECOND_IN_YEAR, ELEM_LISTE
from utils import write_file


def compute_Y_F08(m1, Z1, Se, Sn, T):
    """
    Compute the sputtering yield of all radiolysis products of water induced by water ice irradiation for low-energy ions, according to Fama+2008.

    Parameters:
        m1 (float): atomic mass of the incident ion.
        Z1 (float): atomic number of the incident ion.
        Se (float): electronic stopping power of the incident ion [keV/µm].
        Sn (float): nuclear stopping power of the incident ion [keV/µm].
        T (float): surface temperature [K]

    Returns:
        Y (float): sputtering yield of all radiolysis products of water [molecules/incident particle].
    """

    # these parameters are given in Fama+2008.
    U0 = 0.45       # eV
    C0 = 1.3e-16    # cm²
    Y1Y0 = 220      # no unit
    Ea = 0.06 * 1.6e-19 # J
    k = 1.38e-23    # J/K
    theta = 0       # rad

    alpha = 0.25574 + 1.25338*np.exp(-0.86971*m1) + 0.3793*np.exp(-0.10508*m1)
    eta = 0.0002223 + 0.0013326*np.sin(2.72923*(Z1-1)**0.31812)*np.sin(2.72923*(Z1-1)**0.31812)
    f = 1.3 * (1+np.log(m1)/10)

    #Sn/Se unit: keV/µm
	#Sn*0.10873: eV/(10^15atoms/cm²)
	#Sn*10^-15: eV.cm²/atoms
	#eta*10^32: eV^-1.cm^-4
	#Y: molecules/ion

    Y = (1/U0) * ((3/(4*np.pi**2*C0))*alpha*1e-15*0.10873*Sn+eta*(1.0873*Se)**2) * (1+Y1Y0*np.exp(-Ea/(k*T))) * (np.cos(theta))**(-f)

    return Y


def compute_Y_J09(E, m1, Z1, T):
    """
    Compute the sputtering yield of all radiolysis products of water induced by water ice irradiation for high-energy ions, according to Johnson+2009.

    Parameters:
        E (float): energy of the incident ion [keV].
        m1 (float): atomic mass of the incident ion.
        Z1 (float): atomic number of the incident ion.
        T (float): surface temperature [K].
    
    Returns:
        Y (float): sputtering yield of all radiolysis products of water [molecules/incident particle].
    """

    m1_kg = m1 * 1.6e-27   # kg

    v_ion = np.sqrt(2*E*1e3*1.6e-19/m1_kg)
    v = v_ion / (2.19e6)    # m/s, ion speed divided by 2.19e8 cm/s

    inv_Y = (1/(4.2*Z1**2.8*(v/Z1**(1/3))**2.16) + 1/(11.22*Z1**2.8*(v/Z1**(1/3))**(-2.24)))
    Y = 1/inv_Y

    # temperature dependance as described in Fama+2010
    Y1Y0 = 220  # no unit
    Ea = 0.06 * 1.6e-19 # J
    k = 1.38e-23    # J/K

    Y = Y * (1 + Y1Y0*np.exp(-Ea/(k*T)))

    return Y


def compute_Y_T17(E, T, range):
    """
    Compute the sputtering yield of O2 molecules induced by water ice irradiation, according to Johnson+2009.

    Parameters:
        E (float): energy of the incident ion [keV].
        T (float): surface temperature [K].
        range (float): penetration depth of the incident ion [cm].

    Returns:
        Y (float): sputtering yield of O2 molecules [molecules/incident particle].
    """

    # this parameters are given in Teolis+2017.
    gO2 = 0.005 # eV
    x0 = 2.8    # nm
    theta = 0   # degree
    qO2 = 1000  # +-100, no unit
    Ea = 0.06 * 1.6e-19 # J
    k = 1.38e-23    # J/K

    Y = E*1e3*gO2*x0 / (range*1e7*np.cos(theta)) * (1 - np.exp(-(range*1e7*np.cos(theta))/x0)) * (1 + qO2*np.exp(-Ea/(k*T)))

    return Y



def compute_erosion_ion(T, region, density = 0.93):
    """
    Compute the erosion rate of the surface due to sputtering induced by ion irradiation on different regions of Ganymede.
    Sputtering of all radiolysis products or only O2 molecules are considered.

    Parameters:
        T (float): surface temperature [K].
        region (string): Ganymede's regions as presented in Poppe+2018, can be "polar", "leading" or "trailing".
        density (float): density of water ice, by default = 0.93 g/cm^3.

    Returns:
        tuple (tuple):
            - delta_z_H2O (float): erosion rate considering that all radiolysis products are sputtered [nm/yr].
            - delta_z_O2 (float): erosion rate considering that only O2 molecules are sputtered [nm/yr].
            - max_PR_cm (float): maximum penetration depth of the incident particles [cm].
    """

    ToWrite = [] # contains the text that will be written in the result document saved in the computer.
    print(f"EROSION VALUES WITH A ION FLUX FROM POPPE ET AL. (2018) for T = {T} K in {region} region")
    ToWrite.append(f"EROSION VALUES WITH A ION FLUX FROM POPPE ET AL. (2018) for T = {T} K in {region} region \n")

    max_PR_cm = 0
    integrale_H2O = 0
    integrale_O2 = 0
    for element in ELEM_LISTE:
        elem = element.name
        Z1 = element.atomic_number
        m1 = element.atomic_mass
        print(f"Element {elem} with atomic number Z1 = {Z1} and atomic mass m1 = {m1}")
        ToWrite.append(f"Element {elem} with atomic number Z1 = {Z1} and atomic mass m1 = {m1}")

        energy_INT, flux_INT = upload.ion_flux_data(region, elem)
        flux_divE_INT = flux_INT / energy_INT
        SRIM_energy, SRIM_Se, SRIM_Sn, SRIM_PR = upload.SRIM_data(elem)
        energy_INT_keV = energy_INT / 1e3
        Y_H2O_elem, Y_O2_elem = np.zeros(len(energy_INT_keV)), np.zeros(len(energy_INT_keV))

        Se_INT = np.interp(energy_INT_keV, SRIM_energy, SRIM_Se)
        Sn_INT = np.interp(energy_INT_keV, SRIM_energy, SRIM_Sn)
        PR_INT_cm = np.interp(energy_INT_keV, SRIM_energy, SRIM_PR) * 1e-4  # SRIM_PR being in µm, we convert it in cm.
        max_PR_cm_elem = max(PR_INT_cm)
        #print ("max_PR_cm_elem = ", max_PR_cm_elem)

        for i in range(0,len(energy_INT_keV)):
            Y_O2_elem[i] = compute_Y_T17(energy_INT_keV[i], T, PR_INT_cm[i])
            if(energy_INT_keV[i] <= 100):
                Y_H2O_elem[i] = compute_Y_F08(m1, Z1, Se_INT[i], Sn_INT[i], T)
            else:
                Y_H2O_elem[i] = compute_Y_J09(energy_INT_keV[i], m1, Z1, T)
        
        #plt.plot(energy_INT, Y_H2O_elem)
        #plt.show()
        #plt.plot(energy_INT, Y_O2_elem)
        #plt.show()

        # Compute the integrale of sputtering yield * ion flux
        product_H2O_elem = Y_H2O_elem * flux_divE_INT
        product_O2_elem = Y_O2_elem * flux_divE_INT

        integrale_H2O_elem = np.trapz(product_H2O_elem, energy_INT)
        integrale_O2_elem = np.trapz(product_O2_elem, energy_INT)

        print(f"delta_z(H2O) for {elem}: {(M/(density*Na)) * integrale_H2O_elem * SECOND_IN_YEAR * 1e7} nm/yr")
        print(f"delta_z(O2) for {elem}: {(M/(density*Na)) * 2 * integrale_O2_elem * SECOND_IN_YEAR * 1e7} nm/yr")
        ToWrite.append(f"delta_z(H2O) for {elem}: {(M/(density*Na)) * integrale_H2O_elem * SECOND_IN_YEAR * 1e7} nm/yr")
        ToWrite.append(f"delta_z(O2) for {elem}: {(M/(density*Na)) * 2 * integrale_O2_elem * SECOND_IN_YEAR * 1e7} nm/yr \n")

        integrale_H2O += integrale_H2O_elem
        integrale_O2 += integrale_O2_elem
    
        if (max_PR_cm_elem > max_PR_cm):
            max_PR_cm = max_PR_cm_elem
    
    #print ("max_PR_cm = ", max_PR_cm)

    delta_z_H2O = (M/(density*Na)) * integrale_H2O * SECOND_IN_YEAR * 1e7
    delta_z_O2 = (M/(density*Na)) * integrale_O2 * 2 * SECOND_IN_YEAR * 1e7

    print(f"delta_z(H2O) = {delta_z_H2O} nm/yr")
    print(f"delta_z(O2) = {delta_z_O2} nm/yr")
    print(f'maximum penetration depth of the ions is {max_PR_cm} cm. \n')
    ToWrite.append(f"delta_z(H2O) = {delta_z_H2O} nm/yr")
    ToWrite.append(f"delta_z(O2) = {delta_z_O2} nm/yr")
    ToWrite.append(f'maximum penetration depth of the ions is {max_PR_cm} cm. \n')

    write_file(os.path.join(WORK_DIR_EROSION, f"erosion_ions_{region}_{T}K.txt"), ToWrite)

    print(f"Done for erosion values with ion fluxes from Poppe et al. (2018) for T = {T} K in {region} region. \n")

    return delta_z_H2O, delta_z_O2, max_PR_cm


def compute_erosion_electron(T, region, measure = "G8", density = 0.93):
    """
    Compute the erosion rate of the surface due to sputtering induced by electron irradiation on different regions of Ganymede.
    Sputtering of only O2 molecules is considered.

    Parameters:
        T (float): surface temperature [K].
        region (string): Ganymede's regions as presented in Liuzzo+2020, can be "60N90N", "20N60N", "20N20S", "20S60S", "60S90S".
        measure (string): number of the Galileo encounter which have measured the energy flux around Ganymede, can be "G8" (by default), "G1" or "G28".
        density (float): density of water ice, by default = 0.93 g/cm^3.

    Returns:
        tuple (tuple):
            - delta_z_O2 (float): erosion rate considering that only O2 molecules are sputtered [nm/yr].
            - max_PR_cm (float): maximum penetration depth of the incident particles [cm].
    """

    ToWrite = [] # contains the text that will be written in the result document saved in the computer.
    print(f"EROSION VALUES WITH AN ELECTRON FLUX FROM LIUZZO ET AL. (2020) for T = {T} K in {region} region")
    ToWrite.append(f"EROSION VALUES WITH AN ELECTRON FLUX FROM LIUZZO ET AL. (2020) for T = {T} K in {region} region \n")

    energy_INT_keV, flux_divE_INT = upload.electron_flux_data(region, measure)
    ESTAR_energy, ESTAR_range = upload.ESTAR_data("range")
    Y_O2 = np.zeros(len(energy_INT_keV))

    range_cm_INT = np.interp(energy_INT_keV, ESTAR_energy, ESTAR_range)
    max_PR_cm = max(range_cm_INT)
    
    # analytic expression available for energies ~keV (Johnson+1990)
    i = 0
    while(energy_INT_keV[i] < 100):
        range_cm_INT[i] = 0.046*1e-4*energy_INT_keV[i]**1.76
        i += 1

    for i in range(0,len(energy_INT_keV)):
        Y_O2[i] = compute_Y_T17(energy_INT_keV[i], T, range_cm_INT[i])

    #plt.plot(energy_INT, Y_O2_elem)
    #plt.show()

    # Compute the integrale of sputtering yield * ion flux
    product_O2 = Y_O2 * flux_divE_INT
    integrale_O2 = np.trapz(product_O2, energy_INT_keV)

    delta_z_O2 = C * (M/(density*Na)) * 2 * integrale_O2 * SECOND_IN_YEAR * 1e7     # *C to compensate the sr^-1 in the electron flux

    print(f"delta_z(O2): {delta_z_O2} nm/yr")
    print(f'maximum penetration depth of the ions is {max_PR_cm} cm. \n')
    ToWrite.append(f"delta_z(O2) = {delta_z_O2} nm/yr")
    ToWrite.append(f'maximum penetration depth of the ions is {max_PR_cm} cm. \n')

    write_file(os.path.join(WORK_DIR_EROSION, f"erosion_electrons_{region}_{T}K.txt"), ToWrite)

    print(f"Done for erosion values with electron fluxes from Liuzzo et al. (2020) for T = {T} K in {region} region. \n")

    return delta_z_O2, max_PR_cm



if __name__ == "__main__":
    """
    compute_erosion_ion(90, "polar")
    compute_erosion_ion(120, "polar")
    compute_erosion_ion(90, "leading")
    compute_erosion_ion(155, "leading")
    compute_erosion_ion(90, "trailing")
    compute_erosion_ion(155, "trailing")

    compute_erosion_electron(90, "60N90N")
    compute_erosion_electron(120, "60N90N")
    compute_erosion_electron(90, "20N60N")
    compute_erosion_electron(155, "20N60N")
    compute_erosion_electron(90, "20N20S")
    compute_erosion_electron(155, "20N20S")
    compute_erosion_electron(90, "20S60S")
    compute_erosion_electron(155, "20S60S")
    compute_erosion_electron(90, "60S90S")
    compute_erosion_electron(120, "60S90S")
    """
    compute_erosion_electron(90, "60N90N")


