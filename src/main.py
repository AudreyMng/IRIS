import numpy as np

import erosion_thickness as erosion
import dose
import amorphization as amorph
import constants_Ganymede
import json

config = ""
with open('config.json', 'r') as file:
    config = json.load(file)

region_ion_list = config['region_ion'] 
# region_ion_list = ["polar", "leading", "trailing"],
region_electron_list = config['region_electron']
measure = config['measure']  # name of the Galileo flyby during which electron flux was measured (see Liuzzo+2020)

temperatures = config['temperature']
Temp_range_poles = temperatures['range_poles']
Temp_range_equator = temperatures['range_equator']
Temp_amorph = temperatures['amorph']

# What do you want to compute?
tasks = config['task']

if(tasks["erosion_rate"]):
    #compute erosion rate due to ion irradiation
    for region in region_ion_list:
        if (region == "polar"):
            for Temp in Temp_range_poles:
                erosion.compute_erosion_ion(Temp, region, density = constants_Ganymede.density)
        else:
            for Temp in Temp_range_equator:
                erosion.compute_erosion_ion(Temp, region, density = constants_Ganymede.density)

    #compute erosion rate due to electron irradiation
    for region in region_electron_list:
        if (region == "60N90N" or region == "60S90S"):
            for Temp in Temp_range_poles:
                erosion.compute_erosion_electron(Temp, region, measure = measure, density = constants_Ganymede.density)
        else:
            for Temp in Temp_range_equator:
                erosion.compute_erosion_electron(Temp, region, measure = measure, density = constants_Ganymede.density)


if(tasks["dose_without_sput"]):
    # Compute doses without sputtering as a function of depth, with a log scale to plot them
    for region in region_ion_list:
        dose.compute_doses_ions(region, 4.55e9, density = constants_Ganymede.density, log_scale = True)

    for region in region_electron_list:
        dose.compute_doses_electrons(region, 4.55e9, measure = measure, density = constants_Ganymede.density, log_scale = True)


if(tasks["dose_with_sput_H2O"]):
    # Compute the dose taking sputtering of all radiolysis products into account
    for region in region_ion_list:
        if (region == "polar"):
            for Temp in Temp_range_poles:
                dose.compute_doses_ions_sput(region, Temp, density = constants_Ganymede.density, sput = "H2O")
                dose.compute_doses_electrons_sput(region, "60N90N", Temp, measure = measure, density = constants_Ganymede.density, sput = "H2O")
        else:
            for Temp in Temp_range_equator:
                dose.compute_doses_ions_sput(region, Temp, density = constants_Ganymede.density, sput = "H2O")
                dose.compute_doses_electrons_sput(region, "20N60N", Temp, measure = measure, density = constants_Ganymede.density, sput = "H2O")


if(tasks["dose_with_sput_O2"]):
    # Compute the dose taking sputtering of O2 molecules into account
    for region in region_ion_list:
        if (region == "polar"):
            for Temp in Temp_range_poles:
                dose.compute_doses_ions_sput(region, Temp, density = constants_Ganymede.density, sput = "O2")
                dose.compute_doses_electrons_sput(region, "60N90N", Temp, measure = measure, density = constants_Ganymede.density, sput = "O2")
        else:
            for Temp in Temp_range_equator:
                dose.compute_doses_ions_sput(region, Temp, density = constants_Ganymede.density, sput = "O2")
                dose.compute_doses_electrons_sput(region, "20N60N", Temp, measure = measure, density = constants_Ganymede.density, sput = "O2")


if(tasks["amorph_poles"]):
    # Compute the amorphous fraction on Ganymede' poles induced by ion irradiation
    number_RK2_points = [2900, 1500, 800]   # necessary point number for the RK2 method to converge, should be adapted.
    for (Temp, pts_RK2) in zip(Temp_amorph, number_RK2_points):
        amorph.compute_amorph("polar", Temp, pts_RK2)

if(tasks["amorph_poles_recryst"]):
    # Compute the amorphous fraction on Ganymede' poles induced by ion irradiation
    number_RK2_points_recryst = [8000, 6000, 4500]   # necessary point number for the RK2 method to converge, should be adapted.
    for (Temp, pts_RK2_recryst) in zip(Temp_amorph, number_RK2_points_recryst):
        amorph.compute_amorph_recryst("polar", Temp, pts_RK2_recryst)

if(tasks["amorph_poles_40Krecryst"]):
    # Compute the amorphous fraction on Ganymede' poles induced by ion irradiation
    number_RK2_points_40K = [8000, 6000, 4500]  # necessary point number for the RK2 method to converge, should be adapted.
    for (Temp, pts_RK2_40K) in zip(Temp_amorph, number_RK2_points_40K):
        amorph.compute_amorph_40Krecryst("polar", Temp, pts_RK2_40K)