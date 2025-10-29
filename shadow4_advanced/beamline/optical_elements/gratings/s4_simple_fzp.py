#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2025, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2025. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #
import copy

import numpy

from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.optical_element import OpticalElement
from syned.beamline.shape import Circle

from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.s4_optical_element_decorators import S4OpticalElementDecorator
from shadow4.beamline.s4_beamline_element import S4BeamlineElement
from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen, S4ScreenElement


from dabax.dabax_xraylib import DabaxXraylib
from srxraylib.util.chemical_formula import ChemicalFormulaParser
import scipy.constants as codata

materials_library = DabaxXraylib()

E2K = 2.0 * numpy.pi / (codata.h * codata.c / codata.e * 1e2)
W2E = (codata.h*codata.c/codata.e)

class FZPType:
    AMPLITUDE_ZP = 0
    PHASE_ZP     = 1

GOOD    = 1
LOST_ZP = -191919
GOOD_ZP = 191919

COLLIMATED_SOURCE_LIMIT = 1e4

class S4SimpleFZP(OpticalElement, S4OpticalElementDecorator):
    """
    Defines an ideal Fresnel Zone Plate.

    Constructor.

    Parameters
    ----------
    name : str, optional
        The name of the optical element.
    focusing_direction : int
        0=None, 1=x (sagittal), 2=z (meridional), 3=2D focusing.
    focal : float
        The focal length in meters.
    nominal_wavelength : float
        The nominal wavelength in m for where the focal length is defined.
    diameter : float
        The FZP diameter in m.
    """
    def __init__(self,
                 name                 = "Simple FZP",
                 diameter             = 0.001, # FZP diameter in m
                 delta_rn             = 25.0 * 1e-9, # nm
                 source_distance      = 0.0,
                 type_of_zp           = FZPType.PHASE_ZP,
                 zone_plate_material  = "Au",
                 zone_plate_thickness = 200.0 * 1e-9, #nm
                 substrate_material   = "Si3N4",
                 substrate_thickness  = 50.0 * 1e-9, #nm
                 ):
        super().__init__(name=name,
                         boundary_shape=Circle(radius=diameter/2))

        # back to design um to avoid complications

        self.__type_of_zp           = type_of_zp
        self.__delta_rn             = delta_rn
        self.__source_distance      = source_distance
        self.__zone_plate_material  = zone_plate_material
        self.__zone_plate_thickness = zone_plate_thickness
        self.__substrate_material   = substrate_material
        self.__substrate_thickness  = substrate_thickness

    def diameter(self, native=False): 
        diameter = self.get_boundary_shape().get_radius()*2
        return  diameter*1e6 if native else diameter

    def type_of_zp(self): return self.__type_of_zp

    def delta_rn(self, native=False): return self.__delta_rn*1e9 if native else self.__delta_rn

    def source_distance(self): return self.__source_distance

    def zone_plate_material(self): return self.__zone_plate_material

    def zone_plate_thickness(self, native=False): return self.__zone_plate_thickness*1e9 if native else self.__zone_plate_thickness

    def substrate_material(self): return self.__substrate_material

    def substrate_thickness(self, native=False): return self.__substrate_thickness*1e9 if native else self.__substrate_thickness

    def focal_distance(self, wavelength_in_nm):
        wavelength = wavelength_in_nm*1e-9
        return self.delta_rn(native=False)*self.diameter(native=False)/wavelength
    
    def image_position(self, focal_distance):
        source_distance = self.source_distance()
        return focal_distance*source_distance/(source_distance - focal_distance)
    
    def magnification(self, image_position):
        return abs(image_position / self.source_distance())   

    def calculate_efficiency(self, wavelength_in_nm):
        if self.__type_of_zp == FZPType.PHASE_ZP:
            efficiency, max_efficiency, thickness_max_efficiency = _calculate_efficiency(wavelength=wavelength_in_nm,
                                                                                         zone_plate_material=self.zone_plate_material(),
                                                                                         zone_plate_thickness=self.zone_plate_thickness(native=True))
        else:
            efficiency               = 100 / (numpy.pi ** 2)
            max_efficiency           = numpy.nan
            thickness_max_efficiency = numpy.nan

        return efficiency, max_efficiency, thickness_max_efficiency

    def get_efficiency_by_energy(self, energies):
        efficiencies = numpy.zeros(len(energies))
        for index in range(len(efficiencies)):
            efficiencies[index], _, _ = _calculate_efficiency(wavelength=W2E / energies[index] * 1e9,
                                                              zone_plate_material=self.zone_plate_material(),
                                                              zone_plate_thickness=self.zone_plate_thickness(native=True))
        return efficiencies

    def get_efficiency_by_thickness(self, wavelength_in_nm, thicknesses_in_nm):
        efficiencies = numpy.zeros(len(thicknesses_in_nm))
        for index in range(len(efficiencies)):
            efficiencies[index], _, _ = _calculate_efficiency(wavelength=wavelength_in_nm,
                                                              zone_plate_material=self.zone_plate_material(),
                                                              zone_plate_thickness=thicknesses_in_nm[index])
        return efficiencies

    def to_python_code(self, **kwargs):
        """
        Creates the python code for defining the element.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        str
            Python code.
        """
        txt_pre = """

from shadow4_advanced.beamline.optical_elements.gratings.s4_simple_fzp import S4SimpleFZP
optical_element = S4SimpleFZP(name                 = '{name:s}',
                              diameter             = {diameter:d},
                              delta_rn             = {delta_rn:d},
                              source_distance      = {source_distance:d},
                              type_of_zp           = {type_of_zp:d},
                              zone_plate_material  = '{zone_plate_material:s}',
                              zone_plate_thickness = {zone_plate_thickness:d},
                              substrate_material   = '{substrate_material:s}',
                              substrate_thickness  = {substrate_thickness:d})
"""
        txt = txt_pre.format(**{ 'name': self.get_name(),
                                 'diameter': self.diameter(),
                                 'delta_rn': self.delta_rn(),
                                 'source_distance': self.source_distance(),
                                 'type_of_zp': self.type_of_zp(),
                                 'zone_plate_material' : self.zone_plate_material(),
                                 'zone_plate_thickness' :  self.zone_plate_thickness(),
                                 'substrate_material' : self.substrate_material(),
                                 'substrate_thickness' : self.substrate_thickness()})
        return txt

class S4SimpleFZPElement(S4BeamlineElement):
    """
    Constructor.

    Parameters
    ----------
    optical_element : instance of OpticalElement, optional
        The syned optical element.
    coordinates : instance of ElementCoordinates, optional
        The syned element coordinates.
    input_beam : instance of S4Beam, optional
        The S4 incident beam.

    Returns
    -------
    instance of S4IdealFZPElement.
    """
    def __init__(self,
                 optical_element : S4SimpleFZP = None,
                 coordinates : ElementCoordinates = None,
                 input_beam : S4Beam = None):
        super().__init__(optical_element=optical_element if optical_element is not None else S4SimpleFZP(),
                         coordinates=coordinates if coordinates is not None else ElementCoordinates(),
                         input_beam=input_beam)

    def trace_beam(self, **params):
        """
        Runs (ray tracing) the input beam through the element.

        Parameters
        ----------
        **params

        Returns
        -------
        tuple
            (output_beam, footprint) instances of S4Beam.
        """
        fzp: S4SimpleFZP = self.get_optical_element()

        output_beam, number_of_zones = _apply_fresnel_zone_plate(self._get_initial_beam(),
                                                                 fzp.type_of_zp(),
                                                                 fzp.diameter(native=True),
                                                                 fzp.delta_rn(native=True),
                                                                 fzp.substrate_material(),
                                                                 fzp.substrate_thickness(native=True),
                                                                 fzp.zone_plate_material(),
                                                                 fzp.zone_plate_thickness(native=True),
                                                                 fzp.source_distance())
        output_beam.retrace(self.get_coordinates().q())

        output_beam : S4Beam = output_beam

        footprint = output_beam.duplicate()
        footprint.rotate(numpy.pi / 2, axis=1)

        return output_beam, footprint, {"number_of_zones": number_of_zones}

    def get_output_parameters(self):
        return

    def to_python_code(self, **kwargs):
        """
        Creates the python code for defining the element.

        Parameters
        ----------
        **kwargs

        Returns
        -------
        str
            Python code.
        """
        txt = "\n\n# optical element number XX"
        txt += self.get_optical_element().to_python_code()
        txt += self.to_python_code_coordinates()
        txt += "\nfrom shadow4_advanced.beamline.optical_elements.gratings.s4_simple_fzp import S4SimpleFZPElement"
        txt += "\nbeamline_element = S4SimpleFZPElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)"
        txt += "\n\nbeam, mirr, calculation_result = beamline_element.trace_beam()"

        txt += "\nzone_plate_out = beamline_element.get_optical_element()"

        txt += "\n\navg_wavelength = output_beam.get_photon_wavelength(nolost=1)*1e9"
        txt += "\nfocal_distance   = round(zone_plate_out.focal_distance(avg_wavelength), 6)"
        txt += "\nimage_position   = round(zone_plate_out.image_position(focal_distance), 6)"
        txt += "\nmagnification    = round(zone_plate_out.magnification(image_positiomn), 6)"
        txt += "\nnumber_of_zones  = calculation_result.get('number_of_zones', -1)"

        txt += "\nprint(\"Average Wavelength [nm]:\", avg_wavelength)"
        txt += "\nprint(\"Image Distance [m]     :\", image_position)"
        txt += "\nprint(\"Number of Zones        :\", number_of_zones)"
        txt += "\nprint(\"Focal Distance [m]     :\", focal_distance)"
        txt += "\nprint(\"Magnification          :\", magnification)"

        return txt

    def _get_initial_beam(self):
        screen_element = S4ScreenElement(optical_element=S4Screen(boundary_shape=self.get_optical_element().get_boundary_shape()),
                                         coordinates=ElementCoordinates(p=self.get_coordinates().p(), q=0.0),
                                         input_beam=self.get_input_beam().duplicate())
        output_beam, _ = screen_element.trace_beam()
        
        return output_beam

# ------------------------------------------------------
# FROM SHADOWOUI
# ------------------------------------------------------

def _get_material_density(material: str):
    elements = ChemicalFormulaParser.parse_formula(material)

    mass   = 0.0
    volume = 0.0
    for element in elements:
        mass   += element._molecular_weight * element._n_atoms
        volume += 10. * element._n_atoms

    rho = mass / (0.602 * volume)

    return rho

def _get_material_weight_factor(rays, material, thickness):
    mu = numpy.zeros(len(rays))

    for i in range(0, len(mu)):
        energy_in_KeV = (E2K / rays[i, 10]) * 1e-3
        mu[i]         = materials_library.CS_Total_CP(material, energy_in_KeV)

    rho = _get_material_density(material)

    return numpy.sqrt(numpy.exp(-mu * rho * thickness * 1e-7))  # thickness in CM

def _get_delta_beta(rays, material):
    beta  = numpy.zeros(len(rays))
    delta = numpy.zeros(len(rays))
    density = materials_library.ElementDensity(materials_library.SymbolToAtomicNumber(material))

    for i in range(0, len(rays)):
        energy_in_KeV = (E2K / rays[i, 10]) * 1e-3

        delta[i] = (1 - materials_library.Refractive_Index_Re(material, energy_in_KeV, density))
        beta[i]  = materials_library.Refractive_Index_Im(material, energy_in_KeV, density)

    return delta, beta

def _analyze_zone(zones, focused_beam, p_zp):
    to_analyze = numpy.where(focused_beam.rays[:, 9] == LOST_ZP)

    candidate_rays = copy.deepcopy(focused_beam.rays[to_analyze])

    if len(candidate_rays) > 0:
        xp = candidate_rays[:, 3]
        zp = candidate_rays[:, 5]

        is_collimated = (numpy.max(numpy.abs(xp)) < 1e-9 and numpy.max(numpy.abs(zp)) < 1e-9)

        if is_collimated and not p_zp > COLLIMATED_SOURCE_LIMIT:
            raise ValueError("Beam is collimated, Source Distance should be set to infinite ('Different' and > 10 Km)")

        r = numpy.sqrt(candidate_rays[:, 0]**2 + candidate_rays[:, 2]**2)

        for zone in zones:
            t = numpy.where(numpy.logical_and(r >= zone[0], r <= zone[1]))

            intercepted_rays_f = candidate_rays[t]

            if len(intercepted_rays_f) > 0:
                xp_int = intercepted_rays_f[:, 3]
                zp_int = intercepted_rays_f[:, 5]

                k_mod_int = intercepted_rays_f[:, 10] # CM-1

                k_x_int = k_mod_int*xp_int # CM-1
                k_z_int = k_mod_int*zp_int # CM-1

                # (see formulas in A.G. Michette, "X-ray science and technology"
                #  Institute of Physics Publishing (1993))
                # par. 8.6, pg. 332-337
                x_int_f = intercepted_rays_f[:, 0] # WS Units
                z_int_f = intercepted_rays_f[:, 2] # WS Units

                r_int = numpy.sqrt((x_int_f)**2 + (z_int_f)**2) # WS Units

                d = (zone[1] - zone[0])*100  # to CM

                # computing G (the "grating" wavevector in workspace units^-1)
                gx = -(numpy.pi / d) * x_int_f/r_int
                gz = -(numpy.pi / d) * z_int_f/r_int

                k_x_out = k_x_int + gx
                k_z_out = k_z_int + gz

                k_y_out = numpy.sqrt(k_mod_int**2 - (k_z_out**2 + k_x_out**2)) # keep energy of the photon constant

                xp_out = k_x_out / k_mod_int
                yp_out = k_y_out / k_mod_int
                zp_out = k_z_out / k_mod_int

                candidate_rays[t, 3] = xp_out
                candidate_rays[t, 4] = yp_out
                candidate_rays[t, 5] = zp_out
                candidate_rays[t, 9] = GOOD_ZP

        focused_beam.rays[to_analyze] = candidate_rays

def _apply_fresnel_zone_plate(zone_plate_beam,
                              type_of_zp,
                              diameter,
                              delta_rn,
                              substrate_material,
                              substrate_thickness,
                              zone_plate_material,
                              zone_plate_thickness,
                              source_distance):
    max_zones_number = int(diameter * 1000 / (4 * delta_rn))

    focused_beam = zone_plate_beam.duplicate()
    go           = numpy.where(focused_beam.rays[:, 9] == GOOD)

    if type_of_zp == FZPType.PHASE_ZP:
        substrate_weight_factor = _get_material_weight_factor(focused_beam.rays[go], substrate_material, substrate_thickness)

        focused_beam.rays[go, 6]  = focused_beam.rays[go, 6] * substrate_weight_factor[:]
        focused_beam.rays[go, 7]  = focused_beam.rays[go, 7] * substrate_weight_factor[:]
        focused_beam.rays[go, 8]  = focused_beam.rays[go, 8] * substrate_weight_factor[:]
        focused_beam.rays[go, 15] = focused_beam.rays[go, 15] * substrate_weight_factor[:]
        focused_beam.rays[go, 16] = focused_beam.rays[go, 16] * substrate_weight_factor[:]
        focused_beam.rays[go, 17] = focused_beam.rays[go, 17] * substrate_weight_factor[:]

    clear_zones       = []
    dark_zones        = []
    r_zone_i_previous = 0.0
    for i in range(1, max_zones_number + 1):
        r_zone_i = numpy.sqrt(i * diameter * 1e-6 * delta_rn * 1e-9)
        if i % 2 == 0: clear_zones.append([r_zone_i_previous, r_zone_i])
        else:          dark_zones.append([r_zone_i_previous, r_zone_i])
        r_zone_i_previous = r_zone_i

    focused_beam.rays[go, 9] = LOST_ZP

    _analyze_zone(clear_zones, focused_beam, source_distance)
    if type_of_zp == FZPType.PHASE_ZP: _analyze_zone(dark_zones, focused_beam, source_distance)

    go_2 = numpy.where(focused_beam.rays[:, 9] == GOOD_ZP)

    intensity_go_2 = numpy.sum(focused_beam.rays[go_2, 6] ** 2 + focused_beam.rays[go_2, 7] ** 2 + focused_beam.rays[go_2, 8] ** 2 + \
                               focused_beam.rays[go_2, 15] ** 2 + focused_beam.rays[go_2, 16] ** 2 + focused_beam.rays[go_2, 17] ** 2)

    if type_of_zp == FZPType.PHASE_ZP:
        wavelength  = (2 * numpy.pi / focused_beam.rays[go_2, 10]) * 1e+7  # nm
        delta, beta = _get_delta_beta(focused_beam.rays[go_2], zone_plate_material)

        phi = 2 * numpy.pi * zone_plate_thickness * delta / wavelength
        rho = beta / delta

        efficiency_zp = (1 / (numpy.pi ** 2)) * (1 + numpy.exp(-2 * rho * phi) - (2 * numpy.exp(-rho * phi) * numpy.cos(phi)))
        efficiency_weight_factor = numpy.sqrt(efficiency_zp)

    elif type_of_zp == FZPType.AMPLITUDE_ZP:
        lo_2 = numpy.where(focused_beam.rays[:, 9] == LOST_ZP)

        intensity_lo_2 = numpy.sum(focused_beam.rays[lo_2, 6] ** 2 + focused_beam.rays[lo_2, 7] ** 2 + focused_beam.rays[lo_2, 8] ** 2 + \
                                   focused_beam.rays[lo_2, 15] ** 2 + focused_beam.rays[lo_2, 16] ** 2 + focused_beam.rays[lo_2, 17] ** 2)

        efficiency_zp = numpy.ones(len(focused_beam.rays[go_2])) / (numpy.pi ** 2)
        efficiency_weight_factor = numpy.sqrt(efficiency_zp * (1 + (intensity_lo_2 / intensity_go_2)))

    focused_beam.rays[go_2, 6]  = focused_beam.rays[go_2, 6] * efficiency_weight_factor[:]
    focused_beam.rays[go_2, 7]  = focused_beam.rays[go_2, 7] * efficiency_weight_factor[:]
    focused_beam.rays[go_2, 8]  = focused_beam.rays[go_2, 8] * efficiency_weight_factor[:]
    focused_beam.rays[go_2, 15] = focused_beam.rays[go_2, 15] * efficiency_weight_factor[:]
    focused_beam.rays[go_2, 16] = focused_beam.rays[go_2, 16] * efficiency_weight_factor[:]
    focused_beam.rays[go_2, 17] = focused_beam.rays[go_2, 17] * efficiency_weight_factor[:]
    focused_beam.rays[go_2, 9]  = GOOD

    return focused_beam, max_zones_number

def _calculate_efficiency(wavelength, zone_plate_material, zone_plate_thickness):
    energy_in_KeV = (W2E / wavelength) * 1e6

    density = materials_library.ElementDensity(materials_library.SymbolToAtomicNumber(zone_plate_material))
    delta   = (1 - materials_library.Refractive_Index_Re(zone_plate_material, energy_in_KeV, density))
    beta    = materials_library.Refractive_Index_Im(zone_plate_material, energy_in_KeV, density)
    phi     = 2 * numpy.pi * zone_plate_thickness * delta / wavelength
    rho     = beta / delta

    efficiency               = (1 / (numpy.pi ** 2)) * (1 + numpy.exp(-2 * rho * phi) - (2 * numpy.exp(-rho * phi) * numpy.cos(phi)))
    max_efficiency           = (1 / (numpy.pi ** 2)) * (1 + numpy.exp(-2 * rho * numpy.pi) + (2 * numpy.exp(-rho * numpy.pi)))
    thickness_max_efficiency = numpy.round(wavelength / (2 * delta), 2)

    return efficiency, max_efficiency, thickness_max_efficiency