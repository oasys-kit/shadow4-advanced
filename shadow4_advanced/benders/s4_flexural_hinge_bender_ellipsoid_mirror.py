#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2024. UChicago Argonne, LLC. This software was produced       #
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

from srxraylib.profiles.benders.flexural_hinge_bender_manager import FlexuralHingeStandardBenderManager, FlexuralHingeCalibratedBenderManager, \
    FlexuralHingeBenderStructuralParameters, FlexuralHingeBenderFitParameters, CalibrationParameters, BenderMovement, BenderOuputData, MirrorShape, BenderType

from syned.beamline.shape import EllipticalCylinder, Rectangle
from syned.beamline.element_coordinates import ElementCoordinates

from shadow4.beam.s4_beam import S4Beam
from shadow4.optical_surfaces.s4_mesh import S4Mesh
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirror, S4AdditionalNumericalMeshMirrorElement

class S4FlexuralHingeBenderEllipsoidMirror(S4AdditionalNumericalMeshMirror):
    def __init__(self,
                 ellipsoid_mirror:S4EllipsoidMirror,
                 figure_error_data_file=None,
                 bender_bin_x=10,
                 bender_bin_y=100,
                 E=131000,
                 h=0.01,
                 M1=None,
                 e=None,
                 ratio=None,
                 bender_shape=MirrorShape.TRAPEZIUM,
                 bender_type= BenderType.DOUBLE_MOMENTUM,
                 calibration_parameters: CalibrationParameters=None,
                 fit_to_focus_parameters: FlexuralHingeBenderFitParameters=None,
                 bender_movement: BenderMovement=None):
        assert ellipsoid_mirror is not None

        surface_shape  = ellipsoid_mirror.get_surface_shape()
        boundary_shape = ellipsoid_mirror.get_boundary_shape()

        assert not (surface_shape is None and bender_shape is None)
        if not isinstance(surface_shape, EllipticalCylinder): raise ValueError("Calculation is possible on Elliptical Cylinders only")
        if not isinstance(boundary_shape, Rectangle):         raise ValueError("Calculation is possible on Rectangular/Trapezoidal Elliptical Cylinders only")
        assert (bender_bin_x > 0 and bender_bin_y > 0 and E > 0.0 and h > 0.0)

        self._figure_error_data_file  = figure_error_data_file
        self._bender_bin_x            = bender_bin_x
        self._bender_bin_y            = bender_bin_y
        self._E                       = E
        self._h                       = h
        self._bender_shape            = bender_shape
        self._bender_type             = bender_type
        self._fit_to_focus_parameters = fit_to_focus_parameters
        self._bender_movement         = bender_movement
        self._calibration_parameters  = calibration_parameters

        if not fit_to_focus_parameters is None:
            self._M1    = None
            self._e     = None
            self._ratio = None
        elif not bender_movement is None:
            assert not (M1 is None or e is None or ratio is None)
            assert (0.0 <= e <= 1.0)
            self._M1    = M1
            self._e     = e
            self._ratio = ratio
        else:
            raise ValueError("Specify fit to focus or bender movement")

        x_left, x_right, y_bottom, y_top = boundary_shape.get_boundaries()

        if figure_error_data_file is None:
            self._figure_error_mesh = None
        else:
            figure_error = S4Mesh()
            figure_error.load_h5file(figure_error_data_file)
            xx, yy = figure_error.get_mesh_x_y()
            zz     = figure_error.get_mesh_z()
            self._figure_error_mesh = xx, yy, zz

        grazing_angle = surface_shape.get_grazing_angle()
        p, q          = surface_shape.get_p_q(grazing_angle)

        bender_structural_parameters = FlexuralHingeBenderStructuralParameters(dim_x_minus=-x_left,
                                                                               dim_x_plus=x_right,
                                                                               bender_bin_x=self._bender_bin_x,
                                                                               dim_y_minus=-y_bottom,
                                                                               dim_y_plus=y_top,
                                                                               bender_bin_y=self._bender_bin_y,
                                                                               p=p,
                                                                               q=q,
                                                                               grazing_angle=grazing_angle,
                                                                               E=self._E,
                                                                               h=self._h,
                                                                               M1=self._M1,
                                                                               e=self._e,
                                                                               ratio=self._ratio,
                                                                               figure_error_mesh=self._figure_error_mesh,
                                                                               shape=self._bender_shape,
                                                                               bender_type=self._bender_type,
                                                                               workspace_units_to_m=1.0,
                                                                               workspace_units_to_mm=1000.0)

        if not fit_to_focus_parameters is None:
            bender_manager = FlexuralHingeStandardBenderManager(bender_structural_parameters=bender_structural_parameters)
            bender_data = bender_manager.fit_bender_at_focus_position(fit_to_focus_parameters)
        elif not bender_movement is None:
            if calibration_parameters is None: bender_manager = FlexuralHingeStandardBenderManager(bender_structural_parameters=bender_structural_parameters)
            else:                              bender_manager = FlexuralHingeCalibratedBenderManager(bender_structural_parameters=bender_structural_parameters, calibration_parameters=calibration_parameters)

            bender_data = bender_manager.get_bender_shape_from_movement(bender_movement)
            q           = bender_manager.get_q_ideal_surface(bender_movement)

            # modification of the shape of the mirror with the average q
            ellipsoid_mirror.get_surface_shape().initialize_from_p_q(p, q, grazing_angle)

        S4AdditionalNumericalMeshMirror.__init__(self,
                                                 ideal_mirror=ellipsoid_mirror,
                                                 numerical_mesh_mirror=S4NumericalMeshMirror(boundary_shape=ellipsoid_mirror.get_boundary_shape(),
                                                                                             xx=bender_data.x,
                                                                                             yy=bender_data.y,
                                                                                             zz=bender_data.z_bender_correction.T),
                                                 name=ellipsoid_mirror.get_name())
        self._bender_data = bender_data

    def get_bender_data(self) -> BenderOuputData: return self._bender_data

    def to_python_code(self, **kwargs):
        txt = self.ideal_mirror().to_python_code()

        txt += "\n\nfrom shadow4_advanced.benders.s4_flexural_hinge_bender_ellipsoid_mirror import S4FlexuralHingeBenderEllipsoidMirror"
        txt += "\n\nellipsoid_mirror = optical_element"

        if not self._fit_to_focus_parameters is None:
            txt += "\nfrom srxraylib.profiles.benders.flexural_hinge_bender_manager import FlexuralHingeBenderFitParameters"
            txt += f"\n\nfit_to_focus_parameters = FlexuralHingeBenderFitParameters(optimized_length={self._fit_to_focus_parameters.optimized_length},"
            txt += f"\n                                                          n_fit_steps={self._fit_to_focus_parameters.n_fit_steps},"
            txt += f"\n                                                          M1={self._fit_to_focus_parameters.M1},"
            txt += f"\n                                                          M1_min={self._fit_to_focus_parameters.M1_min},"
            txt += f"\n                                                          M1_max={self._fit_to_focus_parameters.M1_max},"
            txt += f"\n                                                          M1_fixed={self._fit_to_focus_parameters.M1_fixed},"
            txt += f"\n                                                          e={self._fit_to_focus_parameters.e},"
            txt += f"\n                                                          e_min={self._fit_to_focus_parameters.e_min},"
            txt += f"\n                                                          e_max={self._fit_to_focus_parameters.e_max},"
            txt += f"\n                                                          e_fixed={self._fit_to_focus_parameters.e_fixed},"
            txt += f"\n                                                          ratio={self._fit_to_focus_parameters.ratio},"
            txt += f"\n                                                          ratio_min={self._fit_to_focus_parameters.ratio_min},"
            txt += f"\n                                                          ratio_max={self._fit_to_focus_parameters.ratio_max},"
            txt += f"\n                                                          ratio_fixed={self._fit_to_focus_parameters.ratio_fixed})"
            txt += "\nbender_movement = None"
            txt += "\nM1              = None"
            txt += "\ne               = None"
            txt += "\nratio           = None"
        elif not self._bender_movement is None:
            txt += "\nfrom srxraylib.profiles.benders.bender_io import BenderMovement"
            txt += "\n\nfit_to_focus_parameters = None"
            txt += f"\nbender_movement = BenderMovement(position_upstream={str(self._bender_movement.position_upstream)}, position_downstream={str(self._bender_movement.position_downstream)})"
            txt += f"\nM1              = {self._M1}"
            txt += f"\ne               = {self._e}"
            txt += f"\nratio           = {self._ratio}"
            
        if self._calibration_parameters is None:
            txt += "\ncalibration_parameters = None"
        else:
            txt += "\nfrom srxraylib.profiles.benders.flexural_hinge_bender_manager import CalibrationParameters"
            txt += "\n\ncalibration_parameters = CalibrationParameters(parameters_upstream=["+', '.join(['%g']*len(self._calibration_parameters.upstream))+"]," % tuple(self._calibration_parameters.upstream)
            txt += "\n                                               parameters_downstream["+', '.join(['%g']*len(self._calibration_parameters.downstream))+"])" % tuple(self._calibration_parameters.downstream)

        txt += "\n\noptical_element = S4FlexuralHingeBenderEllipsoidMirror(ellipsoid_mirror=ellipsoid_mirror,"
        txt += f"\n                                                       figure_error_data_file={self._figure_error_data_file},"
        txt += f"\n                                                       bender_bin_x={self._bender_bin_x},"
        txt += f"\n                                                       bender_bin_y={self._bender_bin_y},"
        txt += f"\n                                                       E={self._E},"
        txt += f"\n                                                       h={self._h},"
        txt += "\n                                                       M1=M1,"
        txt += "\n                                                       e=e,"
        txt += "\n                                                       ratio=ratio,"
        txt += f"\n                                                       bender_shape={self._bender_shape},"
        txt += f"\n                                                       bender_type={self._bender_movement},"
        txt += "\n                                                       calibration_parameters=calibration_parameters,"
        txt += "\n                                                       fit_to_focus_parameters=fit_to_focus_parameters,"
        txt += "\n                                                       bender_movement=bender_movement)"

        txt += "\n\nbender_data = optical_element.get_bender_data()"

        return txt


class S4FlexuralHingeBenderEllipsoidMirrorElement(S4AdditionalNumericalMeshMirrorElement):
    def __init__(self,
                 optical_element: S4FlexuralHingeBenderEllipsoidMirror=None,
                 coordinates: ElementCoordinates = None,
                 movements: S4BeamlineElementMovements = None,
                 input_beam: S4Beam = None):
        super().__init__(optical_element=optical_element,
                         coordinates=coordinates if coordinates is not None else ElementCoordinates(),
                         movements=movements,
                         input_beam=input_beam)

    def to_python_code(self, **kwargs):
        txt = "\n\n# optical element number XX"
        txt += self.get_optical_element().to_python_code()
        coordinates = self.get_coordinates()
        txt += "\nfrom syned.beamline.element_coordinates import ElementCoordinates"
        txt += "\ncoordinates = ElementCoordinates(p=%g, q=%g, angle_radial=%g, angle_azimuthal=%g, angle_radial_out=%g)" % \
               (coordinates.p(), coordinates.q(), coordinates.angle_radial(), coordinates.angle_azimuthal(), coordinates.angle_radial_out())

        txt += self.to_python_code_movements()

        txt += "\nfrom shadow4_advanced.benders.s4_flexural_hinge_bender_ellipsoid_mirror import S4FlexuralHingeBenderEllipsoidMirrorElement"
        txt += "\nbeamline_element = S4FlexuralHingeBenderEllipsoidMirrorElement(optical_element=optical_element, coordinates=coordinates, movements=movements, input_beam=beam)"
        txt += "\n\nbeam, mirr = beamline_element.trace_beam()"

        return txt