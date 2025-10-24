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
from typing import Tuple
import numpy
import scipy.constants as codata

EtoK = 2.0 * numpy.pi / (codata.h * codata.c / codata.e * 1e2)

from syned.storage_ring.magnetic_structures.undulator import Undulator

from shadow4.sources.s4_electron_beam import S4ElectronBeam
from shadow4.sources.s4_light_source import S4LightSource
from shadow4.beam.s4_beam import S4Beam
from shadow4.sources.source_geometrical.source_geometrical import SourceGeometrical

from hybrid_methods.undulator.hybrid_undulator import (
    HybridUndulatorCalculator, HybridUndulatorInputParameters, HybridUndulatorOutputParameters, HybridUndulatorListener,
    _resonance_energy
)

class S4HybridUndulatorLightSource(S4LightSource, HybridUndulatorCalculator):
    def __init__(self,
                 name = "Undefined",
                 hybrid_input_parameters: HybridUndulatorInputParameters = HybridUndulatorInputParameters(),
                 calculation_listener: HybridUndulatorListener = None):

        if hybrid_input_parameters is None: raise ValueError("hybrid_input_parameters is None")

        electron_beam = S4ElectronBeam(
            energy_in_GeV=hybrid_input_parameters.electron_energy_in_GeV,
            energy_spread=hybrid_input_parameters.electron_energy_spread,
            current=hybrid_input_parameters.ring_current,
            moment_xx=hybrid_input_parameters.electron_beam_size_h**2,
            moment_xxp=0.0,
            moment_xpxp=hybrid_input_parameters.electron_beam_divergence_h**2,
            moment_yy=hybrid_input_parameters.electron_beam_size_v**2,
            moment_yyp=0.0,
            moment_ypyp=hybrid_input_parameters.electron_beam_divergence_v**2
        )
        magnetic_structure = Undulator(
            K_vertical=hybrid_input_parameters.Kv,
            K_horizontal=hybrid_input_parameters.Kh,
            period_length=hybrid_input_parameters.undulator_period,
            number_of_periods=hybrid_input_parameters.number_of_periods
        )

        S4LightSource.__init__(self,
                               name=name,
                               electron_beam=electron_beam,
                               magnetic_structure=magnetic_structure,
                               nrays=hybrid_input_parameters.number_of_rays,
                               seed=hybrid_input_parameters.seed,
                               )
        HybridUndulatorCalculator.__init__(self,
                                           input_parameters=hybrid_input_parameters,
                                           listener=calculation_listener)



    def get_beam(self, **params) -> Tuple[S4Beam, HybridUndulatorOutputParameters]:
        output_beam, output_parameters = self.run_hybrid_undulator_simulation(do_cumulated_calculations=params.get("do_cumulated_calculations", False))

        return output_beam, output_parameters

    def calculate_spectrum(self, **params):
        raise NotImplementedError("This calculation is not implemented for this source.")

    def to_python_code(self, **kwargs):
        """
        Returns the python code for calculating the wiggler source.

        Returns
        -------
        str
            The python code.
        """
        script = ''
        try:    script += self.get_electron_beam().to_python_code()
        except: script += "\n\n#Error retrieving electron_beam code"

        try:    script += self.get_magnetic_structure().to_python_code()
        except: script += "\n\n#Error retrieving magnetic structure code"

        input_parameters = self.get_input_parameters()

        script += "\n\n\n# light source\nfrom shadow4_advanced.s4_hybrid_undulator_light_source import S4HybridUndulatorLightSource"
        script += "\nfrom hybrid_methods.undulator.hybrid_undulator import HybridUndulatorInputParameters, HybridUndulatorOutputParameters"
        script += "\nhybrid_input_parameters = HybridUndulatorInputParameters("
        script += f"\n    number_of_rays                                              = {input_parameters.number_of_rays                                             },"
        script += f"\n    seed                                                        = {input_parameters.seed                                                       },"
        script += f"\n    coherent_beam                                               = {input_parameters.coherent_beam                                              },"
        script += f"\n    phase_diff                                                  = {input_parameters.phase_diff                                                 },"
        script += f"\n    polarization_degree                                         = {input_parameters.polarization_degree                                        },"
        script += f"\n    use_harmonic                                                = {input_parameters.use_harmonic                                               },"
        script += f"\n    harmonic_number                                             = {input_parameters.harmonic_number                                            },"
        script += f"\n    energy                                                      = {input_parameters.energy                                                     },"
        script += f"\n    energy_to                                                   = {input_parameters.energy_to                                                  },"
        script += f"\n    energy_points                                               = {input_parameters.energy_points                                              },"
        script += f"\n    number_of_periods                                           = {input_parameters.number_of_periods                                          },"
        script += f"\n    undulator_period                                            = {input_parameters.undulator_period                                           },"
        script += f"\n    Kv                                                          = {input_parameters.Kv                                                         },"
        script += f"\n    Kh                                                          = {input_parameters.Kh                                                         },"
        script += f"\n    Bh                                                          = {input_parameters.Bh                                                         },"
        script += f"\n    Bv                                                          = {input_parameters.Bv                                                         },"
        script += f"\n    magnetic_field_from                                         = {input_parameters.magnetic_field_from                                        },"
        script += f"\n    initial_phase_vertical                                      = {input_parameters.initial_phase_vertical                                     },"
        script += f"\n    initial_phase_horizontal                                    = {input_parameters.initial_phase_horizontal                                   },"
        script += f"\n    symmetry_vs_longitudinal_position_vertical                  = {input_parameters.symmetry_vs_longitudinal_position_vertical                 },"
        script += f"\n    symmetry_vs_longitudinal_position_horizontal                = {input_parameters.symmetry_vs_longitudinal_position_horizontal               },"
        script += f"\n    horizontal_central_position                                 = {input_parameters.horizontal_central_position                                },"
        script += f"\n    vertical_central_position                                   = {input_parameters.vertical_central_position                                  },"
        script += f"\n    longitudinal_central_position                               = {input_parameters.longitudinal_central_position                              },"
        script += f"\n    electron_energy_in_GeV                                      = {input_parameters.electron_energy_in_GeV                                     },"
        script += f"\n    electron_energy_spread                                      = {input_parameters.electron_energy_spread                                     },"
        script += f"\n    ring_current                                                = {input_parameters.ring_current                                               },"
        script += f"\n    electron_beam_size_h                                        = {input_parameters.electron_beam_size_h                                       },"
        script += f"\n    electron_beam_size_v                                        = {input_parameters.electron_beam_size_v                                       },"
        script += f"\n    electron_beam_divergence_h                                  = {input_parameters.electron_beam_divergence_h                                 },"
        script += f"\n    electron_beam_divergence_v                                  = {input_parameters.electron_beam_divergence_v                                 },"
        script += f"\n    type_of_initialization                                      = {input_parameters.type_of_initialization                                     },"
        script += f"\n    use_stokes                                                  = {input_parameters.use_stokes                                                 },"
        script += f"\n    auto_expand                                                 = {input_parameters.auto_expand                                                },"
        script += f"\n    auto_expand_rays                                            = {input_parameters.auto_expand_rays                                           },"
        script += f"\n    source_dimension_wf_h_slit_gap                              = {input_parameters.source_dimension_wf_h_slit_gap                             },"
        script += f"\n    source_dimension_wf_v_slit_gap                              = {input_parameters.source_dimension_wf_v_slit_gap                             },"
        script += f"\n    source_dimension_wf_h_slit_c                                = {input_parameters.source_dimension_wf_h_slit_c                               },"
        script += f"\n    source_dimension_wf_v_slit_c                                = {input_parameters.source_dimension_wf_v_slit_c                               },"
        script += f"\n    source_dimension_wf_h_slit_points                           = {input_parameters.source_dimension_wf_h_slit_points                          },"
        script += f"\n    source_dimension_wf_v_slit_points                           = {input_parameters.source_dimension_wf_v_slit_points                          },"
        script += f"\n    source_dimension_wf_distance                                = {input_parameters.source_dimension_wf_distance                               },"
        script += f"\n    horizontal_range_modification_factor_at_resizing            = {input_parameters.horizontal_range_modification_factor_at_resizing           },"
        script += f"\n    horizontal_resolution_modification_factor_at_resizing       = {input_parameters.horizontal_resolution_modification_factor_at_resizing      },"
        script += f"\n    vertical_range_modification_factor_at_resizing              = {input_parameters.vertical_range_modification_factor_at_resizing             },"
        script += f"\n    vertical_resolution_modification_factor_at_resizing         = {input_parameters.vertical_resolution_modification_factor_at_resizing        },"
        script += f"\n    waist_position_calculation                                  = {input_parameters.waist_position_calculation                                 },"
        script += f"\n    waist_position                                              = {input_parameters.waist_position                                             },"
        script += f"\n    waist_position_auto                                         = {input_parameters.waist_position_auto                                        },"
        script += f"\n    waist_position_auto_h                                       = {input_parameters.waist_position_auto_h                                      },"
        script += f"\n    waist_position_auto_v                                       = {input_parameters.waist_position_auto_v                                      },"
        script += f"\n    waist_back_propagation_parameters                           = {input_parameters.waist_back_propagation_parameters                          },"
        script += f"\n    waist_horizontal_range_modification_factor_at_resizing      = {input_parameters.waist_horizontal_range_modification_factor_at_resizing     },"
        script += f"\n    waist_horizontal_resolution_modification_factor_at_resizing = {input_parameters.waist_horizontal_resolution_modification_factor_at_resizing},"
        script += f"\n    waist_vertical_range_modification_factor_at_resizing        = {input_parameters.waist_vertical_range_modification_factor_at_resizing       },"
        script += f"\n    waist_vertical_resolution_modification_factor_at_resizing   = {input_parameters.waist_vertical_resolution_modification_factor_at_resizing  },"
        script += f"\n    which_waist                                                 = {input_parameters.which_waist                                                },"
        script += f"\n    number_of_waist_fit_points                                  = {input_parameters.number_of_waist_fit_points                                 },"
        script += f"\n    degree_of_waist_fit                                         = {input_parameters.degree_of_waist_fit                                        },"
        script += f"\n    use_sigma_or_fwhm                                           = {input_parameters.use_sigma_or_fwhm                                          },"
        script += f"\n    waist_position_user_defined                                 = {input_parameters.waist_position_user_defined                                },"
        script += f"\n    kind_of_sampler                                             = {input_parameters.kind_of_sampler                                            },"
        script += f"\n    save_srw_result                                             = {input_parameters.save_srw_result                                            },"
        script += f"\n    source_dimension_srw_file                                   = {input_parameters.source_dimension_srw_file                                  },"
        script += f"\n    angular_distribution_srw_file                               = {input_parameters.angular_distribution_srw_file                              },"
        script += f"\n    x_positions_file                                            = {input_parameters.x_positions_file                                           },"
        script += f"\n    z_positions_file                                            = {input_parameters.z_positions_file                                           },"
        script += f"\n    x_positions_factor                                          = {input_parameters.x_positions_factor                                         },"
        script += f"\n    z_positions_factor                                          = {input_parameters.z_positions_factor                                         },"
        script += f"\n    x_divergences_file                                          = {input_parameters.x_divergences_file                                         },"
        script += f"\n    z_divergences_file                                          = {input_parameters.z_divergences_file                                         },"
        script += f"\n    x_divergences_factor                                        = {input_parameters.x_divergences_factor                                       },"
        script += f"\n    z_divergences_factor                                        = {input_parameters.z_divergences_factor                                       },"
        script += f"\n    combine_strategy                                            = {input_parameters.combine_strategy                                           },"
        script += f"\n    distribution_source                                         = {input_parameters.distribution_source                                        },"
        script += f"\n    energy_step                                                 = {input_parameters.energy_step                                                },"
        script += f"\n    power_step                                                  = {input_parameters.power_step                                                 },"
        script += f"\n    compute_power                                               = {input_parameters.compute_power                                              },"
        script += f"\n    integrated_flux                                             = {input_parameters.integrated_flux                                            },"
        script += f"\n    power_density                                               = {input_parameters.power_density                                              }"
        script += f"\n)"

        script += "\nlight_source = S4HybridUndulatorLightSource(name='%s', hybrid_input_parameters=hybrid_input_parameters)" % (self.get_name())
        script += "\nbeam = light_source.get_beam()"

        return script

    # ABSTRACT METHODS from ###############################################
    #
    def _generate_initial_beam(self):
        input_parameters = self.get_input_parameters()

        energy = input_parameters.energy if input_parameters.use_harmonic != 0 else _resonance_energy(input_parameters, harmonic=input_parameters.harmonic_number)

        source = SourceGeometrical(
            name="Hybrid Undulator Initial Source",
            nrays=self.get_nrays(),
            seed=self.get_seed(),
        )

        source.set_spatial_type_point()
        source.set_angular_distribution_flat(hdiv1=-1.0e-6, hdiv2=1.0e-6, vdiv1= -1.0e-6, vdiv2=1.0e-6)
        source.set_depth_distribution_off()
        source.set_energy_distribution_singleline(value=energy, unit='eV')
        source.set_polarization(polarization_degree=input_parameters.polarization_degree,
                                phase_diff=numpy.radians(input_parameters.phase_diff),
                                coherent_beam=input_parameters.coherent_beam)

        return source.get_beam()

    def _get_rays_from_beam(self, output_beam: S4Beam):                   return output_beam.rays
    def _get_k_from_energy(self, energies: numpy.ndarray):                return energies * EtoK
    def _retrace_output_beam(self, output_beam: S4Beam, distance: float): output_beam.retrace(distance)
    #
    # ###############################################################
