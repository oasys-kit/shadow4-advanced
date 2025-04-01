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

from srxraylib.profiles.benders.fixed_rods_bender_manager import FixedRodsStandardBenderManager, FixedRodsCalibratedBenderManager, \
    FixedRodsBenderStructuralParameters, FixedRodsBenderFitParameters, BenderOuputData, CalibrationParameters, BenderMovement

from syned.beamline.shape import EllipticalCylinder, Rectangle
from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.shape import NumericalMesh

from shadow4.beam.s4_beam import S4Beam
from shadow4.optical_surfaces.s4_mesh import S4Mesh
from shadow4.beamline.s4_beamline_element_movements import S4BeamlineElementMovements
from shadow4.beamline.optical_elements.mirrors.s4_ellipsoid_mirror import S4EllipsoidMirror
from shadow4.beamline.optical_elements.mirrors.s4_numerical_mesh_mirror import S4NumericalMeshMirror
from shadow4.beamline.optical_elements.mirrors.s4_additional_numerical_mesh_mirror import S4AdditionalNumericalMeshMirror, S4AdditionalNumericalMeshMirrorElement

class S4FixedRodsBenderEllipsoidMirror(S4AdditionalNumericalMeshMirror):
    def __init__(self,
                 ellipsoid_mirror:S4EllipsoidMirror,
                 figure_error_data_file=None,
                 bender_bin_x=10,
                 bender_bin_y=100,
                 E=131000,
                 h=0.01,
                 r=0.012,
                 l=0.07,
                 R0=None,
                 eta=None,
                 W2=None,
                 calibration_parameters: CalibrationParameters=None,
                 fit_to_focus_parameters: FixedRodsBenderFitParameters=None,
                 bender_movement: BenderMovement=None):
        assert ellipsoid_mirror is not None
        surface_shape  = ellipsoid_mirror.get_surface_shape()
        boundary_shape = ellipsoid_mirror.get_boundary_shape()

        assert not (surface_shape is None)
        if not isinstance(surface_shape, EllipticalCylinder): raise ValueError("Calculation is possible on Elliptical Cylinders only")
        if not isinstance(boundary_shape, Rectangle):         raise ValueError("Calculation is possible on Rectangular Elliptical Cylinders only")
        assert (bender_bin_x > 0 and bender_bin_y > 0 and E > 0.0 and h > 0.0 and r > 0.0 and l > 0.0)

        self._figure_error_data_file  = figure_error_data_file
        self._bender_bin_x            = bender_bin_x
        self._bender_bin_y            = bender_bin_y
        self._E                       = E
        self._h                       = h
        self._r                       = r
        self._l                       = l

        self._fit_to_focus_parameters = fit_to_focus_parameters
        self._bender_movement         = bender_movement
        self._calibration_parameters  = calibration_parameters

        if not fit_to_focus_parameters is None:
            self._R0  = None
            self._eta = None
            self._W2  = None
        elif not bender_movement is None:
            assert not (R0 is None or eta is None or W2 is None)
            #assert (0.0 <= eta <= 1.0) TODO: verify definition 
            self._R0  = R0
            self._eta = eta
            self._W2  = W2
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

        bender_structural_parameters = FixedRodsBenderStructuralParameters(dim_x_minus=-x_left,
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
                                                                           figure_error_mesh=self._figure_error_mesh,
                                                                           r=self._r,
                                                                           l=self._l,
                                                                           R0=self._R0,
                                                                           eta=self._eta,
                                                                           W2=self._W2,
                                                                           workspace_units_to_m=1.0,
                                                                           workspace_units_to_mm=1000.0)

        if not fit_to_focus_parameters is None:
            bender_manager = FixedRodsStandardBenderManager(bender_structural_parameters=bender_structural_parameters)
            bender_data    = bender_manager.fit_bender_at_focus_position(fit_to_focus_parameters)
        elif not bender_movement is None:
            if calibration_parameters is None: bender_manager = FixedRodsStandardBenderManager(bender_structural_parameters=bender_structural_parameters)
            else:                              bender_manager = FixedRodsCalibratedBenderManager(bender_structural_parameters=bender_structural_parameters, calibration_parameters=calibration_parameters)

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
        self._bender_data      = bender_data
        self._bender_manager   = bender_manager
        self._ellipsoid_mirror = ellipsoid_mirror

    def move_bender(self, bender_movement: BenderMovement):
        if self._bender_movement is None: raise ValueError("This bender has been initialized to fit to focus")

        bender_data = self._bender_manager.get_bender_shape_from_movement(bender_movement)
        q           = self._bender_manager.get_q_ideal_surface(bender_movement)

        surface_shape  = self._ellipsoid_mirror.get_surface_shape()

        grazing_angle = surface_shape.get_grazing_angle()
        p, _          = surface_shape.get_p_q(grazing_angle)

        # modification of the shape of the mirror with the average q
        self._ellipsoid_mirror.get_surface_shape().initialize_from_p_q(p, q, grazing_angle)

        numerical_mesh: NumericalMesh = self.get_surface_shape_instance() # numerical mesh
        numerical_mesh._xx = bender_data.x
        numerical_mesh._yy = bender_data.y
        numerical_mesh._zz = bender_data.z_bender_correction.T

    @property
    def bender_movement(self) -> BenderMovement: return self._bender_movement
    @property
    def calibration_parameters(self) -> CalibrationParameters: return self._calibration_parameters

    def get_bender_data(self) -> BenderOuputData: return self._bender_data

    def to_python_code(self, **kwargs):
        txt = self.ideal_mirror().to_python_code()

        txt += "\n\nfrom shadow4_advanced.benders.s4_fixed_rods_bender_ellipsoid_mirror import S4FixedRodsBenderEllipsoidMirror"
        txt += "\n\nellipsoid_mirror = optical_element"

        if not self._fit_to_focus_parameters is None:
            txt += "\nfrom srxraylib.profiles.benders.fixed_rods_bender_manager import FixedRodsBenderFitParameters"
            txt += f"\n\nfit_to_focus_parameters = FixedRodsBenderFitParameters(optimized_length={self._fit_to_focus_parameters.optimized_length},"
            txt += f"\n                                                          n_fit_steps={self._fit_to_focus_parameters.n_fit_steps},"
            txt += f"\n                                                          R0={self._fit_to_focus_parameters.R0},"
            txt += f"\n                                                          R0_min={self._fit_to_focus_parameters.R0_min},"
            txt += f"\n                                                          R0_max={self._fit_to_focus_parameters.R0_max},"
            txt += f"\n                                                          R0_fixed={self._fit_to_focus_parameters.R0_fixed},"
            txt += f"\n                                                          eta={self._fit_to_focus_parameters.eta},"
            txt += f"\n                                                          eta_min={self._fit_to_focus_parameters.eta_min},"
            txt += f"\n                                                          eta_max={self._fit_to_focus_parameters.eta_max},"
            txt += f"\n                                                          eta_fixed={self._fit_to_focus_parameters.eta_fixed},"
            txt += f"\n                                                          W2={self._fit_to_focus_parameters.W2},"
            txt += f"\n                                                          W2_min={self._fit_to_focus_parameters.W2_min},"
            txt += f"\n                                                          W2_max={self._fit_to_focus_parameters.W2_max},"
            txt += f"\n                                                          W2_fixed={self._fit_to_focus_parameters.W2_fixed})"
            txt += "\nbender_movement = None"
            txt += "\nR0              = None"
            txt += "\neta             = None"
            txt += "\nW2              = None"
        elif not self._bender_movement is None:
            txt += "\nfrom srxraylib.profiles.benders.bender_io import BenderMovement"
            txt += "\n\nfit_to_focus_parameters = None"
            txt += f"\nbender_movement = BenderMovement(position_upstream={str(self._bender_movement.position_upstream)}, position_downstream={str(self._bender_movement.position_downstream)})"
            txt += f"\nR0              = {self._R0}"
            txt += f"\neta             = {self._eta}"
            txt += f"\nW2              = {self._W2}"
            
        if self._calibration_parameters is None:
            txt += "\ncalibration_parameters = None"
        else:
            p0u, p1u = self._calibration_parameters.upstream
            p0d, p1d = self._calibration_parameters.downstream
            txt += "\nfrom srxraylib.profiles.benders.flexural_hinge_bender_manager import CalibrationParameters"
            txt += f"\n\ncalibration_parameters = CalibrationParameters(parameters_upstream=[{p0u}, {p1u}], parameters_downstream=[{p0d}, {p1d}])"

        txt += "\n\noptical_element = S4FixedRodsBenderEllipsoidMirror(ellipsoid_mirror=ellipsoid_mirror,"
        txt += "\n                                                     figure_error_data_file=" + ("None" if self._figure_error_data_file is None else ("'" + self._figure_error_data_file + "'"))   + ","
        txt += f"\n                                                    bender_bin_x={self._bender_bin_x},"
        txt += f"\n                                                    bender_bin_y={self._bender_bin_y},"
        txt += f"\n                                                    E={self._E},"
        txt += f"\n                                                    h={self._h},"
        txt += f"\n                                                    r={self._r},"
        txt += f"\n                                                    l={self._l},"
        txt += "\n                                                     R0=R0,"
        txt += "\n                                                     e=e,"
        txt += "\n                                                     W2=W2,"
        txt += "\n                                                     calibration_parameters=calibration_parameters,"
        txt += "\n                                                     fit_to_focus_parameters=fit_to_focus_parameters,"
        txt += "\n                                                     bender_movement=bender_movement)"

        txt += "\n\nbender_data = optical_element.get_bender_data()"

        return txt


class S4FixedRodsBenderEllipsoidMirrorElement(S4AdditionalNumericalMeshMirrorElement):
    def __init__(self,
                 optical_element: S4FixedRodsBenderEllipsoidMirror=None,
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

        txt += "\nfrom shadow4_advanced.benders.s4_fixed_rods_bender_ellipsoid_mirror import S4FixedRodsBenderEllipsoidMirrorElement"
        txt += "\nbeamline_element = S4FixedRodsBenderEllipsoidMirrorElement(optical_element=optical_element, coordinates=coordinates, movements=movements, input_beam=beam)"
        txt += "\n\nbeam, mirr = beamline_element.trace_beam()"

        return txt