#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2023. UChicago Argonne, LLC. This software was produced       #
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
import numpy
import copy
from typing import Tuple, Union

import numpy as np
from srxraylib.util.data_structures import ScaledMatrix

from syned.beamline.shape import Ellipsoid, EllipticalCylinder, Hyperboloid, HyperbolicCylinder, Circle
from syned.beamline.beamline_element import BeamlineElement

from shadow4.beam.s4_beam import S4Beam
from shadow4.optical_surfaces.s4_mesh import S4Mesh
from shadow4.beamline.s4_beamline_element import S4BeamlineElement, ElementCoordinates
from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen, S4ScreenElement
from shadow4.beamline.optical_elements.mirrors.s4_mirror import S4MirrorElement
from shadow4.beamline.optical_elements.gratings.s4_grating import S4GratingElement
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirror, S4AdditionalNumericalMeshMirrorElement
from shadow4.beamline.optical_elements.gratings.s4_additional_numerical_mesh_grating import S4AdditionalNumericalMeshGrating, S4AdditionalNumericalMeshGratingElement
from shadow4.beamline.optical_elements.refractors.s4_crl import S4CRLElement
from shadow4.beamline.optical_elements.refractors.s4_lens import S4LensElement

from hybrid_methods.coherence.hybrid_screen import HybridBeamWrapper, HybridLengthUnits, HybridOEWrapper, HybridGeometryAnalysis, HybridCalculationType, HybridInputParameters, HybridCalculationResult, \
    HybridDiffractionPlane, HybridPropagationType, HybridWaveOpticsProvider, AbstractHybridScreen, AbstractSimpleApertureHybridScreen, AbstractMirrorOrGratingSizeHybridScreen, \
    AbstractMirrorSizeAndErrorHybridScreen, AbstractGratingSizeAndErrorHybridScreen, AbstractCRLSizeHybridScreen, AbstractCRLSizeAndErrorHybridScreen, \
    AbstractKBMirrorSizeHybridScreen, AbstractKBMirrorSizeAndErrorHybridScreen, HybridScreenManager

IMPLEMENTATION = "SHADOW4"

# -------------------------------------------------------------
# RAY-TRACING WRAPPERS
# -------------------------------------------------------------

class S4HybridBeam(HybridBeamWrapper):
    def __init__(self, beam : Union[S4Beam, Tuple[S4Beam, S4Beam]], **kwargs):
        super(S4HybridBeam, self).__init__(beam, HybridLengthUnits.METERS, **kwargs)

    def duplicate(self, **kwargs):
        if isinstance(self.wrapped_beam, S4Beam): return S4HybridBeam(self.wrapped_beam.duplicate(**kwargs))
        else:                                     return S4HybridBeam([wb.duplicate(**kwargs) for wb in self.wrapped_beam])

class S4HybridOE(HybridOEWrapper):
    def __init__(self, optical_element: Union[S4BeamlineElement, Tuple[S4BeamlineElement, S4BeamlineElement]], name=None):
        if isinstance(optical_element, S4BeamlineElement):  super(S4HybridOE, self).__init__(optical_element, name=optical_element.get_optical_element().get_name() if name is None else name)
        else:                                               super(S4HybridOE, self).__init__(optical_element, name="KB-mirror" if name is None else name)

    def check_congruence(self, calculation_type : int):
        if   calculation_type == HybridCalculationType.SIMPLE_APERTURE                and not isinstance(self.wrapped_optical_element, S4ScreenElement):
            raise Exception("Simple Aperture calculation runs for Slits O.E. only")
        elif calculation_type == HybridCalculationType.MIRROR_OR_GRATING_SIZE         and not (isinstance(self.wrapped_optical_element, S4MirrorElement) or
                                                                                               isinstance(self.wrapped_optical_element, S4GratingElement)):
            raise Exception("Mirror/Grating calculation runs for Mirror/Grating O.E. only")
        elif calculation_type == HybridCalculationType.MIRROR_SIZE_AND_ERROR_PROFILE  and not isinstance(self.wrapped_optical_element, S4AdditionalNumericalMeshMirrorElement):
            raise Exception("Mirror calculation runs for Mirror O.E. with error profile only")
        elif calculation_type == HybridCalculationType.GRATING_SIZE_AND_ERROR_PROFILE and not isinstance(self.wrapped_optical_element, S4AdditionalNumericalMeshGratingElement):
            raise Exception("Grating calculation runs for Grating O.E. with error profile only")
        elif calculation_type in [HybridCalculationType.CRL_SIZE, HybridCalculationType.CRL_SIZE_AND_ERROR_PROFILE] and not (isinstance(self.wrapped_optical_element, S4LensElement) or
                                                                                                                             isinstance(self.wrapped_optical_element, S4CRLElement)):
            raise Exception("CRL calculation runs for Lens and CRLs O.E. only")
        elif calculation_type == HybridCalculationType.KB_SIZE                        and not (isinstance(self.wrapped_optical_element, list) and not \
                                                                                               (isinstance(self.wrapped_optical_element[0], S4MirrorElement) or
                                                                                                isinstance(self.wrapped_optical_element[1], S4MirrorElement))):
            raise Exception("KB-Mirror calculation runs for a pari of Mirror O.E.s only")
        elif calculation_type == HybridCalculationType.KB_SIZE_AND_ERROR_PROFILE     and not (isinstance(self.wrapped_optical_element, list) and not \
                                                                                              (isinstance(self.wrapped_optical_element[0], S4AdditionalNumericalMeshMirrorElement) or
                                                                                               isinstance(self.wrapped_optical_element[1], S4AdditionalNumericalMeshMirrorElement))):
            raise Exception("KB-Mirror calculation runs for a pari of Mirror O.E.s with error profile only")

    def duplicate(self):
        if isinstance(self.wrapped_optical_element, S4BeamlineElement): return S4HybridOE(self.wrapped_optical_element.duplicate())
        else:                                                           return S4HybridOE([wo.duplicate() for wo in self.wrapped_optical_element])

# -------------------------------------------------------------
# HYBRID SCREENS HELPER CLASSES
# -------------------------------------------------------------

class _ShadowOEHybridScreen():
    NPOLY_ANGLE = 3
    NPOLY_L     = 6

    def _set_image_distance_from_optical_element(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        beamline_element = input_parameters.optical_element.wrapped_optical_element

        calculation_parameters.image_plane_distance = beamline_element.get_coordinates().q()

    def _no_lost_rays_from_oe(self, input_parameters : HybridInputParameters) -> bool:
        shadow_beam      = input_parameters.beam.wrapped_beam
        beamline_element = input_parameters.optical_element.wrapped_optical_element

        beam_after  = shadow_beam
        beam_before = beamline_element.get_input_beam()

        number_of_good_rays_before = len(beam_before.rays[numpy.where(beam_before.rays[:, 9] == 1)])
        number_of_good_rays_after  = len(beam_after.rays[numpy.where(beam_after.rays[:, 9] == 1)])

        return number_of_good_rays_before == number_of_good_rays_after

    def _manage_common_initial_screen_projection_data(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        shadow_element = input_parameters.optical_element.wrapped_optical_element.duplicate()
        shadow_element.set_input_beam(self._get_shadow_beam_for_initial_tracing(input_parameters, calculation_parameters))
        shadow_element.set_optical_element(self._get_shadow_optical_element_for_initial_tracing(input_parameters, calculation_parameters, shadow_element))

        coordinates = shadow_element.get_coordinates()
        movements   = shadow_element.get_movements()

        if not movements is None:
            # tracing must be done without o.e. movements: hybrid is going to take care of that
            x_rot = movements.rotation_x
            y_rot = movements.rotation_y
            z_rot = movements.rotation_z

            movements.rotation_x = 0.0
            movements.rotation_y = 0.0
            movements.rotation_z = 0.0

        beam_at_image_plane, footprint_beam = shadow_element.trace_beam()

        if not movements is None:
            # restore o.e. setting for further calculations
            movements.rotation_x = x_rot
            movements.rotation_y = y_rot
            movements.rotation_z = z_rot

        calculation_parameters.set("shadow_beam",    beam_at_image_plane)
        calculation_parameters.set("footprint_beam", footprint_beam)

        image_beam_good, image_beam_lost, good_only_cursor = self._process_shadow_beam(beam_at_image_plane, lost=True)  # xshi change from 0 to 1
        #image_beam.set_initial_flux(input_parameters.original_beam.wrapped_beam.get_initial_flux())

        calculation_parameters.set("image_plane_beam_good", image_beam_good)
        calculation_parameters.set("image_plane_beam_lost", image_beam_lost)
        calculation_parameters.set("good_only_cursor",      good_only_cursor)

        input_parameters.listener.status_message("Projecting beam at HYBRID screen")

        hybrid_screen_beam = beam_at_image_plane.duplicate()
        hybrid_screen_beam.rays = hybrid_screen_beam.rays[numpy.where(hybrid_screen_beam.rays[:, 9] == 1)]
        hybrid_screen_beam.retrace(-coordinates.q()) # hybrid screen is at center

        calculation_parameters.set("screen_plane_beam", hybrid_screen_beam)

        energy     = hybrid_screen_beam.get_photon_energy_eV()   # eV
        wavelength = hybrid_screen_beam.get_photon_wavelength()  # m

        input_parameters.listener.status_message("Using MEAN photon energy [eV]:" + str(numpy.average(energy)))

        xx_screen = hybrid_screen_beam.rays[:, 0]
        zz_screen = hybrid_screen_beam.rays[:, 2]
        xp_screen = hybrid_screen_beam.rays[:, 3]
        yp_screen = hybrid_screen_beam.rays[:, 4]
        zp_screen = hybrid_screen_beam.rays[:, 5]

        x_min   = numpy.min(xx_screen)
        x_max   = numpy.max(xx_screen)
        z_min   = numpy.min(zz_screen)
        z_max   = numpy.max(zz_screen)
        dx_rays = numpy.arctan(xp_screen / yp_screen)  # calculate divergence from direction cosines from SHADOW file  dx = atan(v_x/v_y)
        dz_rays = numpy.arctan(zp_screen / yp_screen)  # calculate divergence from direction cosines from SHADOW file  dz = atan(v_z/v_y)

        calculation_parameters.energy     = numpy.average(energy)
        calculation_parameters.wavelength = numpy.average(wavelength)
        calculation_parameters.xx_screen = xx_screen
        calculation_parameters.zz_screen = zz_screen
        calculation_parameters.xp_screen = xp_screen
        calculation_parameters.yp_screen = yp_screen
        calculation_parameters.zp_screen = zp_screen
        calculation_parameters.x_min = x_min
        calculation_parameters.x_max = x_max
        calculation_parameters.z_min = z_min
        calculation_parameters.z_max = z_max
        calculation_parameters.dx_rays = dx_rays  # radians
        calculation_parameters.dz_rays = dz_rays  # radians

        return calculation_parameters

    def _get_shadow_element(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        shadow_element = input_parameters.optical_element.wrapped_optical_element.duplicate()
        shadow_element.set_input_beam(self._get_shadow_beam_for_initial_tracing(input_parameters, calculation_parameters))
        shadow_element.set_optical_element(self._get_shadow_optical_element_for_initial_tracing(input_parameters, calculation_parameters, shadow_element))

    def _get_shadow_beam_for_initial_tracing(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        return input_parameters.optical_element.wrapped_optical_element.get_input_beam()

    def _get_shadow_optical_element_for_initial_tracing(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters, beamline_element: BeamlineElement):
        return beamline_element.get_optical_element()

    def _get_screen_plane_histograms(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        screen_plane_beam = calculation_parameters.get("screen_plane_beam")

        histogram_s  = None
        bins_s       = None
        histogram_t  = None
        bins_t       = None
        histogram_2D = None

        if input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D:
            ticket = screen_plane_beam.histo2(col_h=1,
                                              col_v=3,
                                              nbins_h=int(input_parameters.n_bins_x),
                                              nbins_v=int(input_parameters.n_bins_z),
                                              xrange=[calculation_parameters.x_min, calculation_parameters.x_max],
                                              yrange=[calculation_parameters.z_min, calculation_parameters.z_max],
                                              nolost=1,
                                              ref=23)

            histogram_s  = ticket['histogram_h']
            bins_s       = ticket['bin_h_center']
            histogram_t  = ticket['histogram_v']
            bins_t       = ticket['bin_v_center']
            histogram_2D = ticket['histogram']
        else:
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in X
                ticket = screen_plane_beam.histo1(1,
                                                  nbins=int(input_parameters.n_bins_x),
                                                  xrange=[calculation_parameters.x_min, calculation_parameters.x_max],
                                                  nolost=1,
                                                  ref=23)

                histogram_s = ticket['histogram']
                bins_s      = ticket['bins']
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:  # 1d in X
                ticket = screen_plane_beam.histo1(3,
                                                  nbins=int(input_parameters.n_bins_z),
                                                  xrange=[calculation_parameters.z_min, calculation_parameters.z_max],
                                                  nolost=1,
                                                  ref=23)

                histogram_t = ticket['histogram']
                bins_t      = ticket['bins']

        return histogram_s, bins_s, histogram_t, bins_t, histogram_2D

    @staticmethod
    def _process_shadow_beam(shadow_beam: S4Beam, lost=False):
        cursor_go = numpy.where(shadow_beam.rays[:, 9] == 1)
    
        image_beam_rays = copy.deepcopy(shadow_beam.rays[cursor_go])
        image_beam_rays[:, 11] = numpy.arange(1, len(image_beam_rays) + 1, 1)
    
        out_beam_go = S4Beam()
        out_beam_go.rays = image_beam_rays
    
        if lost:
            cursor_lo = numpy.where(shadow_beam.rays[:, 9] != 1)
    
            lost_rays = copy.deepcopy(shadow_beam.rays[cursor_lo])
            lost_rays[:, 11] = numpy.arange(1, len(lost_rays) + 1, 1)
    
            out_beam_lo = S4Beam()
            out_beam_lo.rays = lost_rays
    
            return out_beam_go, out_beam_lo, cursor_go
        else:
            return out_beam_go, None, cursor_go

    def _apply_convolution_to_rays(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters):
        image_plane_beam_good = calculation_parameters.get("image_plane_beam_good")
        image_plane_beam_lost = calculation_parameters.get("image_plane_beam_lost")

        ff_beam = None
        nf_beam = None

        if input_parameters.diffraction_plane == HybridDiffractionPlane.BOTH_2D:
            ff_beam = image_plane_beam_good.duplicate()

            angle_num = numpy.sqrt(1 + (numpy.tan(calculation_parameters.dz_convolution)) ** 2 + (numpy.tan(calculation_parameters.dx_convolution)) ** 2)

            ff_beam.rays[:, 0] = copy.deepcopy(calculation_parameters.xx_image_ff)
            ff_beam.rays[:, 2] = copy.deepcopy(calculation_parameters.zz_image_ff)
            ff_beam.rays[:, 3] = numpy.tan(calculation_parameters.dx_convolution) / angle_num
            ff_beam.rays[:, 4] = 1 / angle_num
            ff_beam.rays[:, 5] = numpy.tan(calculation_parameters.dz_convolution) / angle_num
        else:
            if input_parameters.diffraction_plane in [HybridDiffractionPlane.SAGITTAL, HybridDiffractionPlane.BOTH_2X1D]:
                # FAR FIELD PROPAGATION
                if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
                    ff_beam = image_plane_beam_good.duplicate()

                    angle_perpen = numpy.arctan(calculation_parameters.zp_screen / calculation_parameters.yp_screen)
                    angle_num    = numpy.sqrt(1 + (numpy.tan(angle_perpen)) ** 2 + (numpy.tan(calculation_parameters.dx_convolution)) ** 2)

                    ff_beam.rays[:, 0] = copy.deepcopy(calculation_parameters.xx_image_ff)
                    ff_beam.rays[:, 3] = numpy.tan(calculation_parameters.dx_convolution) / angle_num
                    ff_beam.rays[:, 4] = 1 / angle_num
                    ff_beam.rays[:, 5] = numpy.tan(angle_perpen) / angle_num

                # NEAR FIELD PROPAGATION
                if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                    nf_beam = image_plane_beam_good.duplicate()

                    nf_beam.rays[:, 0] = copy.deepcopy(calculation_parameters.xx_image_nf)

            if input_parameters.diffraction_plane in [HybridDiffractionPlane.TANGENTIAL, HybridDiffractionPlane.BOTH_2X1D]:
                # FAR FIELD PROPAGATION
                if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
                    if ff_beam is None: ff_beam = image_plane_beam_good.duplicate()

                    angle_perpen = numpy.arctan(calculation_parameters.xp_screen / calculation_parameters.yp_screen)
                    angle_num    = numpy.sqrt(1 + (numpy.tan(angle_perpen)) ** 2 + (numpy.tan(calculation_parameters.dz_convolution)) ** 2)

                    ff_beam.rays[:, 2] = copy.deepcopy(calculation_parameters.zz_image_ff)
                    ff_beam.rays[:, 3] = numpy.tan(angle_perpen) / angle_num
                    ff_beam.rays[:, 4] = 1 / angle_num
                    ff_beam.rays[:, 5] = numpy.tan(calculation_parameters.dz_convolution) / angle_num

                    if image_plane_beam_lost.get_number_of_rays() > 0: ff_beam = S4Beam(array=numpy.concatenate((ff_beam.rays, image_plane_beam_lost.rays), axis=0))

                # NEAR FIELD PROPAGATION
                if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
                    if nf_beam is None: nf_beam = image_plane_beam_good.duplicate()

                    nf_beam.rays[:, 2] = copy.deepcopy(calculation_parameters.zz_image_nf)

        if input_parameters.propagation_type in [HybridPropagationType.FAR_FIELD, HybridPropagationType.BOTH]:
            if image_plane_beam_lost.get_number_of_rays() > 0: ff_beam = S4Beam(array=numpy.concatenate((ff_beam.rays, image_plane_beam_lost.rays), axis=0))

        if input_parameters.propagation_type in [HybridPropagationType.NEAR_FIELD, HybridPropagationType.BOTH]:
            if image_plane_beam_lost.get_number_of_rays() > 0: nf_beam = S4Beam(array=numpy.concatenate((nf_beam.rays, image_plane_beam_lost.rays), axis=0))

        calculation_parameters.ff_beam = None if ff_beam is None else S4HybridBeam(beam=ff_beam)
        calculation_parameters.nf_beam = None if nf_beam is None else S4HybridBeam(beam=nf_beam)

class _S4ApertureHybridScreen(_ShadowOEHybridScreen):
    def _check_oe_displacements(self, input_parameters : HybridInputParameters):
        shadow_oe_element = input_parameters.optical_element.wrapped_optical_element

        if not shadow_oe_element.get_movements() is None: raise Exception("O.E. Movements are not supported for this kind of calculation")

    def _calculate_geometrical_parameters(self, input_parameters: HybridInputParameters):
        geometrical_parameters = AbstractHybridScreen.GeometricalParameters()

        shadow_element = input_parameters.optical_element.wrapped_optical_element
        beam_before      = shadow_element.get_input_beam()
        oe_before        = shadow_element.get_optical_element()
        coordinates      = shadow_element.get_coordinates()

        if oe_before.get_boundary_shape() is None:
            geometrical_parameters.is_infinite = True
        else:
            if oe_before._i_stop: raise Exception("Simple Aperture calculation runs for apertures only")

            beam_at_the_slit = beam_before.duplicate()
            beam_at_the_slit.retrace(coordinates.p())  # TRACE INCIDENT BEAM UP TO THE SLIT

            boundaries = oe_before.get_boundary_shape().get_boundaries()

            geometrical_parameters.max_tangential    = boundaries[3]
            geometrical_parameters.min_tangential    = boundaries[2]
            geometrical_parameters.max_sagittal      = boundaries[1]
            geometrical_parameters.min_sagittal      = boundaries[0]

            ticket_tangential = beam_at_the_slit.histo1(3, nbins=500, nolost=1, ref=23)
            ticket_sagittal   = beam_at_the_slit.histo1(1, nbins=500, nolost=1, ref=23)

            geometrical_parameters.ticket_tangential = {'histogram' : ticket_tangential["histogram"], 'bins' : ticket_tangential["bin_center"]}
            geometrical_parameters.ticket_sagittal   = {'histogram' : ticket_sagittal["histogram"],   'bins' : ticket_sagittal["bin_center"]}

        return geometrical_parameters

class _S4OEWithSurfaceHybridScreen(_ShadowOEHybridScreen):
    def _check_oe_displacements(self, input_parameters: HybridInputParameters):
        beamline_element  = input_parameters.optical_element.wrapped_optical_element
        movements         = beamline_element.get_movements()
        diffraction_plane = input_parameters.diffraction_plane

        if not movements is None:
            if diffraction_plane == HybridDiffractionPlane.SAGITTAL:  # X
                if movements.rotation_x != 0.0 or movements.rotation_z != 0.0: raise Exception("Only rotations around the Y axis are supported for sagittal diffraction plane")
            elif (diffraction_plane == HybridDiffractionPlane.TANGENTIAL or diffraction_plane == HybridDiffractionPlane.BOTH_2D):  # Z
                if movements.rotation_y != 0.0 or movements.rotation_z != 0.0: raise Exception("Only rotations around the X axis are supported for tangential or Both (2D) diffraction planes")
            elif diffraction_plane == HybridDiffractionPlane.BOTH_2X1D:  # Z
                if movements.rotation_z != 0.0:                               raise Exception("Only rotations around the X and Y axis are supported for Both (1D+1D) diffraction planes")

    def _calculate_geometrical_parameters(self, input_parameters: HybridInputParameters):
        geometrical_parameters = AbstractHybridScreen.GeometricalParameters()

        beamline_element = input_parameters.optical_element.wrapped_optical_element.duplicate()
        beam_before      = beamline_element.get_input_beam()
        oe_before        = beamline_element.get_optical_element()

        if oe_before.get_boundary_shape() is None:
            geometrical_parameters.is_infinite = True
        else:
            beam_before.rays = beam_before.rays[numpy.where(beam_before.rays[:, 9] == 1)]  # GOOD ONLY BEFORE THE BEAM

            boundary_shape = oe_before.get_boundary_shape()

            boundaries = boundary_shape.get_boundaries()

            geometrical_parameters.max_tangential = boundaries[3]
            geometrical_parameters.min_tangential = boundaries[2]
            geometrical_parameters.max_sagittal   = boundaries[1]
            geometrical_parameters.min_sagittal   = boundaries[0]

            oe_before._boundary_shape = None # make it infinite to compute the real size of the beam

            _, footprint_beam = beamline_element.trace_beam()

            oe_before._boundary_shape = boundary_shape

            ticket_tangential = footprint_beam.histo1(2, nbins=500, nolost=1, ref=23)
            ticket_sagittal   = footprint_beam.histo1(1, nbins=500, nolost=1, ref=23)

            geometrical_parameters.ticket_tangential = {'histogram' : ticket_tangential["histogram"], 'bins' : ticket_tangential["bin_center"]}
            geometrical_parameters.ticket_sagittal   = {'histogram' : ticket_sagittal["histogram"],   'bins' : ticket_sagittal["bin_center"]}

        return geometrical_parameters

    def _get_footprint_spatial_coordinates(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[numpy.ndarray, numpy.ndarray]:
        footprint_beam   = calculation_parameters.get("footprint_beam")
        good_only_cursor = calculation_parameters.get("good_only_cursor")

        xx_mirr = footprint_beam.rays[:, 0][good_only_cursor]
        yy_mirr = footprint_beam.rays[:, 1][good_only_cursor]

        return xx_mirr, yy_mirr

    def _get_rays_angles(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters) -> Tuple[numpy.ndarray, numpy.ndarray]:
        beamline_element = input_parameters.optical_element.wrapped_optical_element
        good_only_cursor = calculation_parameters.get("good_only_cursor")

        input_beam       = beamline_element.get_input_beam().duplicate()
        input_beam.rays  = input_beam.rays[good_only_cursor]
        optical_element  = beamline_element.get_optical_element().duplicate()

        # put beam in mirror reference system --------------
        #
        p              = beamline_element.get_coordinates().p()
        theta_grazing1 = numpy.pi / 2 - beamline_element.get_coordinates().angle_radial()
        alpha1         = beamline_element.get_coordinates().angle_azimuthal()

        input_beam.rotate(alpha1, axis=2)
        input_beam.rotate(theta_grazing1, axis=1)
        input_beam.translation([0.0, -p * numpy.cos(theta_grazing1), p * numpy.sin(theta_grazing1)])

        # mirror movement:
        movements = beamline_element.get_movements()
        if movements is not None:
            if movements.f_move:
                input_beam.rot_for(OFFX=movements.offset_x,
                                   OFFY=movements.offset_y,
                                   OFFZ=movements.offset_z,
                                   X_ROT=movements.rotation_x,
                                   Y_ROT=movements.rotation_y,
                                   Z_ROT=movements.rotation_z)

        # compute angles in mirror reference system --------------
        #
        v_in = input_beam.get_columns([4, 5, 6])

        if   isinstance(beamline_element, S4MirrorElement):  _, normal = optical_element.apply_mirror_reflection(input_beam)
        elif isinstance(beamline_element, S4GratingElement): _, normal = optical_element.apply_grating_diffraction(input_beam)

        v_out = input_beam.get_columns([4, 5, 6])

        angle_in  = numpy.arccos(v_in[0,:] * normal[0,:] +
                                 v_in[1,:] * normal[1,:] +
                                 v_in[2,:] * normal[2,:])

        angle_out = numpy.arccos(v_out[0,:] * normal[0,:] +
                                 v_out[1,:] * normal[1,:] +
                                 v_out[2,:] * normal[2,:])

        # grazing angles
        incidence_angle  = numpy.absolute((numpy.pi / 2) - angle_in)  # abs is necessary for a bug in S4 to be fixed
        reflection_angle = (numpy.pi / 2) - angle_out

        return incidence_angle, reflection_angle

    def _has_pitch_displacement(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]:
        movements      = input_parameters.optical_element.wrapped_optical_element.get_movements()

        if movements is None: return False, 0.0
        else: return movements.rotation_x != 0.0, movements.rotation_x

    def _has_roll_displacement(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]:
        movements      = input_parameters.optical_element.wrapped_optical_element.get_movements()

        if movements is None: return False, 0.0
        else: return movements.rotation_y != 0.0, movements.rotation_y

    def _has_sagittal_offset(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]:
        movements      = input_parameters.optical_element.wrapped_optical_element.get_movements()

        if movements is None: return False, 0.0
        else: return movements.offset_x != 0.0, movements.offset_x

    def _has_normal_offset(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[bool, float]:
        movements      = input_parameters.optical_element.wrapped_optical_element.get_movements()

        if movements is None: return False, 0.0
        else: return movements.offset_z != 0.0, movements.offset_z

    def _get_optical_element_angles(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[float, float]:
        coordinates    = input_parameters.optical_element.wrapped_optical_element.get_coordinates()
        
        return (numpy.pi / 2 - coordinates.angle_radial()), (numpy.pi / 2 - coordinates.angle_radial_out())
   
    def _get_optical_element_spatial_limits(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> Tuple[float, float, float, float]:
        boundaries = input_parameters.optical_element.wrapped_optical_element.get_optical_element().get_boundary_shape().get_boundaries()

        return boundaries[1], boundaries[0], boundaries[3], boundaries[2]

    def _get_focal_length_from_optical_element(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> float:
        if isinstance(input_parameters.optical_element.wrapped_optical_element, S4AdditionalNumericalMeshMirrorElement):
            surface_shape = input_parameters.optical_element.wrapped_optical_element.get_optical_element().ideal_mirror().get_surface_shape()
        elif isinstance(input_parameters.optical_element.wrapped_optical_element, S4AdditionalNumericalMeshGratingElement):
            surface_shape = input_parameters.optical_element.wrapped_optical_element.get_optical_element().ideal_grating().get_surface_shape()
        else:
            surface_shape  = input_parameters.optical_element.wrapped_optical_element.get_optical_element().get_surface_shape()

        if   (isinstance(surface_shape, Ellipsoid) or isinstance(surface_shape, EllipticalCylinder)) or \
             (isinstance(surface_shape, Hyperboloid) or isinstance(surface_shape, HyperbolicCylinder)): return surface_shape.get_q_focus()
        else: raise ValueError("calculation support for elliptical or hyperbolic elements: TO BE COMPLETED")

class _S4OEWithSurfaceAndErrorHybridScreen(_S4OEWithSurfaceHybridScreen):
    def _get_shadow_optical_element_for_initial_tracing(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters, beamline_element: BeamlineElement):
        shadow_oe = beamline_element.get_optical_element()

        if   isinstance(shadow_oe, S4AdditionalNumericalMeshMirror):  return shadow_oe.ideal_mirror()
        elif isinstance(shadow_oe, S4AdditionalNumericalMeshGrating): return shadow_oe.ideal_grating()
        else: raise ValueError("This should never happen")

    def _get_error_profile(self, input_parameters: HybridInputParameters, calculation_parameters : AbstractHybridScreen.CalculationParameters) -> ScaledMatrix:
        optical_surface = input_parameters.optical_element.wrapped_optical_element.get_optical_element().get_optical_surface_instance()

        x_coords, y_coords = optical_surface.get_mesh_x_y()
        z_values           = optical_surface.get_mesh_z()

        return ScaledMatrix(x_coords, y_coords, z_values)

    def _get_tangential_displacement_index(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        movements      = input_parameters.optical_element.wrapped_optical_element.get_movements()

        error_profile = calculation_parameters.get("error_profile")

        return 0.0 if movements is None else movements.offset_y / error_profile.delta_y()

    def _get_sagittal_displacement_index(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        movements      = input_parameters.optical_element.wrapped_optical_element.get_movements()

        error_profile = calculation_parameters.get("error_profile")

        return 0.0 if movements is None else movements.offset_x / error_profile.delta_x()

class _S4OELensHybridScreen(_S4ApertureHybridScreen):
    def _check_oe_displacements(self, input_parameters : HybridInputParameters): pass

    def _calculate_geometrical_parameters(self, input_parameters: HybridInputParameters):
        geometrical_parameters = AbstractHybridScreen.GeometricalParameters()

        shadow_element = input_parameters.optical_element.wrapped_optical_element
        beam_before    = shadow_element.get_input_beam()
        oe_before      = shadow_element.get_optical_element()
        coordinates    = shadow_element.get_coordinates()

        if oe_before.get_boundary_shape() is None:
            geometrical_parameters.is_infinite = True
        else:
            beam_at_the_first_lens = beam_before.duplicate()
            beam_at_the_first_lens.retrace(coordinates.q())  # TRACE INCIDENT BEAM UP TO THE first lens

            boundary_shape = oe_before.get_boundary_shape()
            boundaries     = boundary_shape.get_boundaries()

            if isinstance(boundary_shape, Circle): # radius, sag center, tang center
                geometrical_parameters.max_tangential    = boundaries[2] + boundaries[0]
                geometrical_parameters.min_tangential    = boundaries[2] - boundaries[0]
                geometrical_parameters.max_sagittal      = boundaries[1] + boundaries[0]
                geometrical_parameters.min_sagittal      = boundaries[1] - boundaries[0]
            else:
                geometrical_parameters.max_tangential    = boundaries[3]
                geometrical_parameters.min_tangential    = boundaries[2]
                geometrical_parameters.max_sagittal      = boundaries[1]
                geometrical_parameters.min_sagittal      = boundaries[0]

            ticket_tangential = beam_at_the_first_lens.histo1(3, nbins=500, nolost=1, ref=23)
            ticket_sagittal   = beam_at_the_first_lens.histo1(1, nbins=500, nolost=1, ref=23)

            geometrical_parameters.ticket_tangential = {'histogram' : ticket_tangential["histogram"], 'bins' : ticket_tangential["bin_center"]}
            geometrical_parameters.ticket_sagittal   = {'histogram' : ticket_sagittal["histogram"],   'bins' : ticket_sagittal["bin_center"]}

        return geometrical_parameters


    def _get_shadow_element(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        compound_oe_element = input_parameters.optical_element.wrapped_optical_element
        boundary_shape      = compound_oe_element.get_optical_element().get_boundary_shape()

        if boundary_shape is None: raise Exception("Calculation not possible: Lens have infinite diameter")

        _, image_plane_distance, _, _, _ = compound_oe_element.get_coordinates().get_positions()

        screen_slit_element = S4ScreenElement(optical_element=S4Screen(name="last element", boundary_shape=boundary_shape),
                                              coordinates=ElementCoordinates(p=-image_plane_distance, q=image_plane_distance, angle_radial=0.0, angle_radial_out=numpy.pi),
                                              input_beam=input_parameters.beam.wrapped_beam)

        shadow_beam, _ = screen_slit_element.trace_beam()

        screen_slit_element.set_input_beam(shadow_beam)

        return screen_slit_element

class _S4OELensAndErrorHybridScreen(_S4OELensHybridScreen):
    def _get_error_profiles(self, input_parameters: HybridInputParameters, calculation_parameters: AbstractHybridScreen.CalculationParameters):
        coords_to_m    = input_parameters.get("crl_coords_to_m")
        thickness_to_m = input_parameters.get("crl_thickness_to_m")

        return [self._read_oasys_surface(thickness_error_file, coords_to_m, thickness_to_m) for thickness_error_file in input_parameters.get("crl_error_profiles")]

    @staticmethod
    def _read_oasys_surface(filename, coords_to_m, thickness_to_m):
        numerical_mesh = S4Mesh()
        numerical_mesh.load_h5file(filename)

        x_coords, y_coords = numerical_mesh.get_mesh_x_y()
        z_values           = numerical_mesh.get_mesh_z()

        return ScaledMatrix(x_coords*coords_to_m, y_coords*coords_to_m, z_values.T*thickness_to_m)

class _S4OEKBMirrorHybridScreen():
    def _modify_image_plane_distance_on_kb_1(self, kb_mirror_1: S4BeamlineElement, kb_mirror_2: S4BeamlineElement):
        total_image_distance = kb_mirror_1.get_coordinates().q() + \
                               kb_mirror_2.get_coordinates().p() + \
                               kb_mirror_2.get_coordinates().q()

        kb_mirror_1.get_coordinates().set_p_and_q(p=kb_mirror_1.get_coordinates().p(), q=total_image_distance)

    def _merge_beams(self, beam_1: S4HybridBeam, beam_2: S4HybridBeam):
        if beam_1 is None or beam_2 is None: return None

        go_beam_1 = numpy.where(beam_1.wrapped_beam.rays[:, 9] == 1)
        go_beam_2 = numpy.where(beam_2.wrapped_beam.rays[:, 9] == 1)

        if   len(go_beam_2[0]) < len(go_beam_1[0]): go_beam_1 = go_beam_1[0][0 : len(go_beam_2[0])]
        elif len(go_beam_2[0]) > len(go_beam_1[0]): go_beam_2 = go_beam_2[0][0 : len(go_beam_1[0])]

        beam_2.wrapped_beam.rays[go_beam_2, 0] = beam_1.wrapped_beam.rays[go_beam_1, 2] # tangential component 1 becomes the sagittal 2
        beam_2.wrapped_beam.rays[go_beam_2, 3] = beam_1.wrapped_beam.rays[go_beam_1, 5]

        return beam_2

# -------------------------------------------------------------
# HYBRID SCREENS IMPLEMENTATION CLASSES
# -------------------------------------------------------------

class S4SimpleApertureHybridScreen(_S4ApertureHybridScreen, AbstractSimpleApertureHybridScreen):
    def __init__(self, wave_optics_provider : HybridWaveOpticsProvider, **kwargs):
        AbstractSimpleApertureHybridScreen.__init__(self, wave_optics_provider, **kwargs)

class S4MirrorOrGratingSizeHybridScreen(_S4OEWithSurfaceHybridScreen, AbstractMirrorOrGratingSizeHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, **kwargs):
        AbstractMirrorOrGratingSizeHybridScreen.__init__(self, wave_optics_provider, **kwargs)

class S4MirrorSizeAndErrorHybridScreen(_S4OEWithSurfaceAndErrorHybridScreen, AbstractMirrorSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, **kwargs):
        AbstractMirrorSizeAndErrorHybridScreen.__init__(self, wave_optics_provider, **kwargs)

class S4GratingSizeAndErrorHybridScreen(_S4OEWithSurfaceAndErrorHybridScreen, AbstractGratingSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, **kwargs):
        AbstractGratingSizeAndErrorHybridScreen.__init__(self, wave_optics_provider, **kwargs)

class S4CRLSizeHybridScreen(_S4OELensHybridScreen, AbstractCRLSizeHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, **kwargs):
        AbstractCRLSizeHybridScreen.__init__(self, wave_optics_provider, **kwargs)

class S4CRLSizeAndErrorHybridScreen(_S4OELensAndErrorHybridScreen, AbstractCRLSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, **kwargs):
        AbstractCRLSizeAndErrorHybridScreen.__init__(self, wave_optics_provider, **kwargs)

class S4KBMirrorSizeHybridScreen(_S4OEKBMirrorHybridScreen, AbstractKBMirrorSizeHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, implementation: str, **kwargs):
        AbstractKBMirrorSizeHybridScreen.__init__(self, wave_optics_provider, implementation, **kwargs)

class S4KBMirrorSizeAndErrorHybridScreen(_S4OEKBMirrorHybridScreen, AbstractKBMirrorSizeAndErrorHybridScreen):
    def __init__(self, wave_optics_provider: HybridWaveOpticsProvider, implementation: str, **kwargs):
        AbstractKBMirrorSizeAndErrorHybridScreen.__init__(self, wave_optics_provider, implementation, **kwargs)


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# SHADOW4 COMPATIBILITY
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

from syned.beamline.optical_element import OpticalElement
from syned.beamline.element_coordinates import ElementCoordinates

from shadow4.beamline.s4_beamline_element import S4BeamlineElement
from shadow4.beam.s4_beam import S4Beam

class S4HybridScreen(OpticalElement):
    def __init__(self, calculation_type : HybridCalculationType):
        self.__hybrid_screen = HybridScreenManager.Instance().create_hybrid_screen_manager(IMPLEMENTATION, calculation_type)
        self.__calculation_type = calculation_type

        super(S4HybridScreen, self).__init__(name=self.__hybrid_screen.__class__.__qualname__,  boundary_shape=None)

    def _get_hybrid_screen(self) -> _ShadowOEHybridScreen: return self.__hybrid_screen
    def _get_calculation_type(self) -> HybridCalculationType: return self.__calculation_type

    def to_python_code(self, **kwargs):
        txt = "\nfrom shadow4_hybrid.s4_hybrid_screen import S4HybridScreen"
        txt += "\n\ncalculation_type=%i" % self.__calculation_type
        txt += "\nhybrid_screen = S4HybridScreen(calculation_type)\n"

        return txt

class S4HybridScreenElement(S4BeamlineElement):
    def __init__(self, hybrid_screen: S4HybridScreen, hybrid_input_parameters : HybridInputParameters):
        self.__hybrid_input_parameters = hybrid_input_parameters

        super(S4HybridScreenElement, self).__init__(optical_element=hybrid_screen,
                                                    coordinates=ElementCoordinates(p=0, q=0, angle_radial=0.0, angle_radial_out=np.pi, angle_azimuthal=0.0),
                                                    movements=None,
                                                    input_beam=self.__hybrid_input_parameters.beam.wrapped_beam)

    def trace_beam(self, **params):
        hybrid_screen    = self.get_optical_element()._get_hybrid_screen()
        input_parameters = self.__hybrid_input_parameters

        return hybrid_screen.run_hybrid_method(input_parameters), None

    def to_python_code(self, **kwargs):
        calculation_type = self.get_optical_element()._get_calculation_type()
        input_parameters = self.__hybrid_input_parameters

        txt = "\n\n# optical element number XX"
        txt += "\nfrom hybrid_methods.coherence.hybrid_screen import StdIOHybridListener"
        txt += "\nfrom shadow4_hybrid.s4_hybrid_screen import S4HybridBeam, S4HybridOE, HybridInputParameters, S4HybridScreenElement"
        txt += self.get_optical_element().to_python_code()

        txt += "\n\nadditional_parameters = {}"

        if calculation_type == HybridCalculationType.CRL_SIZE_AND_ERROR_PROFILE:
            txt += "\nadditional_parameters[\"crl_error_profiles\"] = %s" % input_parameters.get("crl_error_profiles")
            if not input_parameters.get("crl_material") is None:
                txt += "\nadditional_parameters[\"crl_material\"] = '%s'" % input_parameters.get("crl_material")
            else:
                txt += "\nadditional_parameters[\"crl_delta\"] = %g" % input_parameters.get("crl_delta")
            txt += "\nadditional_parameters[\"crl_scaling_factor\"] = %g" % input_parameters.get("crl_scaling_factor")
            txt += "\nadditional_parameters[\"crl_coords_to_m\"]    = %g" % input_parameters.get("crl_coords_to_m")
            txt += "\nadditional_parameters[\"crl_thickness_to_m\"] = %g" % input_parameters.get("crl_thickness_to_m")

        if calculation_type in [HybridCalculationType.KB_SIZE, HybridCalculationType.KB_SIZE_AND_ERROR_PROFILE]:
            txt += "\nkb_mirror_1_element = beamline.get_beamline_element_at(-2)"
            txt += "\nkb_mirror_2_element = beamline.get_beamline_element_at(-1)"
        else:
            txt += "\nhybrid_beamline_element = beamline.get_beamline_element_at(-1)"

        txt += "\ninput_parameters = HybridInputParameters(listener=StdIOHybridListener(),"
        if calculation_type in [HybridCalculationType.KB_SIZE, HybridCalculationType.KB_SIZE_AND_ERROR_PROFILE]:
            txt += "\n                                         beam=S4HybridBeam(beam=[kb_mirror_2_element.get_input_beam(), beam]),"
            txt += "\n                                         optical_element=S4HybridOE(optical_element=[kb_mirror_1_element, kb_mirror_1_element]),"
        else:
            txt += "\n                                         beam=S4HybridBeam(beam=beam),"
            txt += "\n                                         optical_element=S4HybridOE(optical_element=hybrid_beamline_element),"
        txt += "\n                                         diffraction_plane=%i," % input_parameters.diffraction_plane
        txt += "\n                                         propagation_type=%i," % input_parameters.propagation_type
        txt += "\n                                         n_bins_x=%i," % input_parameters.n_bins_x
        txt += "\n                                         n_bins_z=%i," % input_parameters.n_bins_z
        txt += "\n                                         n_peaks=%i," % input_parameters.n_peaks
        txt += "\n                                         fft_n_pts=%i," % input_parameters.fft_n_pts
        txt += "\n                                         analyze_geometry=%r," % input_parameters.analyze_geometry
        txt += "\n                                         random_seed=None,"
        txt += "\n                                         **additional_parameters)"
        txt += "\n\nbeamline_element = S4HybridScreenElement(hybrid_screen=hybrid_screen, hybrid_input_parameters=input_parameters)"

        txt += "\n\nhybrid_result, _ = beamline_element.trace_beam()"
        if input_parameters.propagation_type == HybridPropagationType.FAR_FIELD:
            txt +="\nbeam=hybrid_result.far_field_beam.wrapped_beam"
        elif input_parameters.propagation_type == HybridPropagationType.NEAR_FIELD:
            txt +="\nbeam=hybrid_result.near_field_beam.wrapped_beam"
        else:
            txt +="\nbeam_ff=hybrid_result.far_field_beam.wrapped_beam"
            txt +="\nbeam_nf=hybrid_result.near_field_beam.wrapped_beam"
            txt +="\nbeam=beam_ff #change to nf if desired"
        txt +="\nmirr=None"

        return txt


# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------
# FACTORY METHOD INITIALIZATION
# -------------------------------------------------------------
# -------------------------------------------------------------
# -------------------------------------------------------------

try:
    hsm = HybridScreenManager.Instance()
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4SimpleApertureHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4MirrorOrGratingSizeHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4MirrorSizeAndErrorHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4GratingSizeAndErrorHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4CRLSizeHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4CRLSizeAndErrorHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4KBMirrorSizeHybridScreen)
    hsm.add_hybrid_screen_class(IMPLEMENTATION, S4KBMirrorSizeAndErrorHybridScreen)
except Exception as e:
    print(e)