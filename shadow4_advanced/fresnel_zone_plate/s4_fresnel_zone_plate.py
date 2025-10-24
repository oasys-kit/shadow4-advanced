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
from typing import Any, Tuple
import numpy
import copy

from hybrid_methods.fresnel_zone_plate.hybrid_fresnel_zone_plate import HybridFresnelZonePlate, FZPAttributes, FZPSimulatorOptions, FZPCalculationInputParameters, FZPCalculationResult
from srxraylib.util.inverse_method_sampler import Sampler2D

from syned.beamline.optical_element import OpticalElement
from syned.beamline.shape import Circle
from syned.beamline.element_coordinates import ElementCoordinates

from shadow4.beam.s4_beam import S4Beam
from shadow4.beamline.s4_optical_element_decorators import S4OpticalElementDecorator
from shadow4.beamline.s4_beamline_element import S4BeamlineElement

from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen, S4ScreenElement
from shadow4.beamline.optical_elements.ideal_elements.s4_ideal_lens import S4IdealLens, S4IdealLensElement

GOOD = 1

class S4FresnelZonePlate(HybridFresnelZonePlate, OpticalElement, S4OpticalElementDecorator):
    def __init__(self,
                 name="Undefined",
                 input_parameters: FZPCalculationInputParameters = FZPCalculationInputParameters(),
                 options: FZPSimulatorOptions = FZPSimulatorOptions(),
                 attributes: FZPAttributes = FZPAttributes()):
        """
        fresnel Zone Plate

        Parameters
        ----------
        name : str, optional
            The name of the optical element.
        boundary_shape : instance of BoundaryShape, optional
            The geometry of the slit aperture. if None, it is initialized to BoundaryShape().

        """
        HybridFresnelZonePlate.__init__(self, options=options, attributes=attributes)
        OpticalElement.__init__(self, name=name, boundary_shape=Circle(radius=0.5*attributes.diameter))
        self.__input_parameters = input_parameters
        self.__source_distance = None
        self.__input_beam      = None

    @property
    def calculation_input_parameters(self) -> FZPCalculationInputParameters: return self.__input_parameters
    @property
    def zp_focal_distance(self): return self._simulator.zp_focal_distance
    @property
    def zp_image_distance(self): return self._simulator.zp_image_distance

    @property
    def source_distance(self): return self.__source_distance
    @source_distance.setter
    def source_distance(self, source_distance): self.__source_distance = source_distance

    @property
    def input_beam(self): return self.__input_beam
    @input_beam.setter
    def input_beam(self, input_beam): self.__input_beam = input_beam

    def to_python_code(self, **kwargs):
        options : FZPSimulatorOptions = self._simulator.options
        attributes : FZPAttributes    = self._simulator.attributes

        txt = f"""

from shadow4_advanced.fresnel_zone_plate.s4_fresnel_zone_plate import S4FresnelZonePlate
from hybrid_methods.fresnel_zone_plate.hybrid_fresnel_zone_plate import FZPAttributes, FZPSimulatorOptions, FZPCalculationInputParameters

input_parameters = FZPCalculationInputParameters(source_distance={self.calculation_input_parameters.source_distance},
                                                 image_distance={self.calculation_input_parameters.image_distance},
                                                 n_points={self.calculation_input_parameters.n_points},
                                                 multipool={self.calculation_input_parameters.multipool},
                                                 profile_last_index={self.calculation_input_parameters.profile_last_index},
                                                 increase_resolution={self.calculation_input_parameters.increase_resolution},
                                                 increase_points={self.calculation_input_parameters.increase_points})
options = FZPSimulatorOptions(with_central_stop={options.with_central_stop},
                              cs_diameter={options.cs_diameter},
                              with_order_sorting_aperture={options.with_order_sorting_aperture},
                              osa_position={options.osa_position},
                              osa_diameter={options.osa_diameter},
                              zone_plate_type={options.zone_plate_type},
                              width_coating={options.width_coating},
                              height1_factor={options.height1_factor},
                              height2_factor={options.height2_factor},
                              with_range={options.with_range},
                              with_multi_slicing={options.with_multi_slicing},
                              n_slices={options.n_slices},
                              with_complex_amplitude={options.with_complex_amplitude},
                              store_partial_results={options.store_partial_results})
attributes = FZPAttributes(height={attributes.height},
                           diameter={attributes.diameter},
                           b_min={attributes.b_min},
                           zone_plate_material='{attributes.zone_plate_material}',
                           template_material='{attributes.template_material}')

optical_element=S4FresnelZonePlate(name='{self.get_name()}',
                                   input_parameters=input_parameters,
                                   options=options,
                                   attributes=attributes)

        """

        return txt

    def _get_zone_plate_aperture_beam(self, attributes: FZPAttributes, **kwargs) -> Tuple[S4Beam, float]:
        screen_element = S4ScreenElement(optical_element=S4Screen(boundary_shape=self.get_boundary_shape()),
                                         coordinates=ElementCoordinates(p=self.source_distance, q=0.0),
                                         input_beam=self.input_beam)

        output_beam, _ = screen_element.trace_beam()
        energy_KeV     = 1e-3*round(numpy.average(output_beam.get_photon_energy_eV(nolost=1)))

        return output_beam, energy_KeV

    def _get_ideal_lens_beam(self, zone_plate_beam: S4Beam, **kwargs) -> Any:
        focal_distance = self.zp_focal_distance

        ideal_lens_element = S4IdealLensElement(optical_element=S4IdealLens(focal_x=focal_distance,
                                                                            focal_y=focal_distance),
                                                coordinates=ElementCoordinates(p=0.0, q=0.0),
                                                input_beam=zone_plate_beam)
        output_beam, _ = ideal_lens_element.trace_beam()

        return output_beam

    def _apply_convolution_to_rays(self, output_beam: S4Beam, calculation_result: FZPCalculationResult, **kwargs):
        go = numpy.where(output_beam.rays[:, 9] == GOOD)

        dx_ray = numpy.arctan(output_beam.rays[go, 3] / output_beam.rays[go, 4])  # calculate divergence from direction cosines from SHADOW file  dx = atan(v_x/v_y)
        dz_ray = numpy.arctan(output_beam.rays[go, 5] / output_beam.rays[go, 4])  # calculate divergence from direction cosines from SHADOW file  dz = atan(v_z/v_y)

        s2d = Sampler2D(calculation_result.dif_xpzp, calculation_result.xp, calculation_result.zp)

        pos_dif_x, pos_dif_z = s2d.get_n_sampled_points(dx_ray.shape[1])

        # new divergence distribution: convolution
        dx_conv = dx_ray + numpy.arctan(pos_dif_x)  # add the ray divergence kicks
        dz_conv = dz_ray + numpy.arctan(pos_dif_z)  # add the ray divergence kicks

        # correction to the position with the divergence kick from the waveoptics calculation
        # the correction is made on the positions at the hybrid screen (T_IMAGE = 0)
        if self._simulator.image_distance is None: image_distance = self.zp_image_distance
        else:                                      image_distance = self._simulator.image_distance

        xx_image = output_beam.rays[go, 0] + image_distance * numpy.tan(dx_conv)  # ray tracing to the image plane
        zz_image = output_beam.rays[go, 2] + image_distance * numpy.tan(dz_conv)  # ray tracing to the image plane

        angle_num = numpy.sqrt(1 + (numpy.tan(dz_conv)) ** 2 + (numpy.tan(dx_conv)) ** 2)

        output_beam.rays[go, 0] = copy.deepcopy(xx_image)
        output_beam.rays[go, 2] = copy.deepcopy(zz_image)
        output_beam.rays[go, 3] = numpy.tan(dx_conv) / angle_num
        output_beam.rays[go, 4] = 1 / angle_num
        output_beam.rays[go, 5] = numpy.tan(dz_conv) / angle_num
        # ----------------------------------------------------------------------------------------

        efficiency_factor = numpy.sqrt(calculation_result.efficiency)

        output_beam.rays[go, 6] *= efficiency_factor
        output_beam.rays[go, 7] *= efficiency_factor
        output_beam.rays[go, 8] *= efficiency_factor
        output_beam.rays[go, 15] *= efficiency_factor
        output_beam.rays[go, 16] *= efficiency_factor
        output_beam.rays[go, 17] *= efficiency_factor


class S4FresnelZonePlateElement(S4BeamlineElement):
    def __init__(self,
                 optical_element: S4FresnelZonePlate = None,
                 coordinates: ElementCoordinates = None,
                 input_beam: S4Beam = None):
        super().__init__(optical_element=optical_element if optical_element is not None else S4FresnelZonePlateElement(),
                         coordinates=coordinates if coordinates is not None else ElementCoordinates(),
                         input_beam=input_beam)

    def trace_beam(self, **params):
        input_beam = self.get_input_beam().duplicate()
        zone_plate: S4FresnelZonePlate = self.get_optical_element()

        p, q = self.get_coordinates().get_p_and_q()

        zone_plate.source_distance = p
        zone_plate.input_beam      = input_beam

        return zone_plate.run_fzp_hybrid_method(zone_plate.calculation_input_parameters)

    def to_python_code(self, **kwargs):
        txt = "\n\n# optical element number XX"
        txt += self.get_optical_element().to_python_code()

        coordinates = self.get_coordinates()

        txt += "\nfrom syned.beamline.element_coordinates import ElementCoordinates"
        txt += "\n\ncoordinates = ElementCoordinates(p=%g, q=%g, angle_radial=%g, angle_azimuthal=%g, angle_radial_out=%g)" % \
               (coordinates.p(), coordinates.q(), coordinates.angle_radial(), coordinates.angle_azimuthal(), coordinates.angle_radial_out())

        txt += "\n\nfrom shadow4_advanced.fresnel_zone_plate.s4_fresnel_zone_plate import S4FresnelZonePlateElement"
        txt += "\n\nbeamline_element = S4FresnelZonePlateElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)"
        txt += "\n\nbeam, calculation_result = beamline_element.trace_beam()"

        txt += "\nzone_plate_out           = beamline_element.get_optical_element()"

        txt += "\n\navg_energy      = 1e3 * zone_plate_out.get_energy_in_KeV()"
        txt += "\nimage_distance  = round(zone_plate_out.zp_image_distance, 6)"
        txt += "\nnumber_of_zones = zone_plate_out.get_n_zones()"
        txt += "\nfocal_distance  = round(zone_plate_out.zp_focal_distance, 6)"
        txt += "\nefficiency      = round(calculation_result.efficiency * 100, 2)"

        txt += "\nprint(\"Average Energy [eV]:\", avg_energy)"
        txt += "\nprint(\"Image Distance [m] :\", image_distance)"
        txt += "\nprint(\"Number of Zones    :\", number_of_zones)"
        txt += "\nprint(\"Focal Distance [m] :\", focal_distance)"
        txt += "\nprint(\"Efficiency (%)     :\", efficiency)"


        return txt