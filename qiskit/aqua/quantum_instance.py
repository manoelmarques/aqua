# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Quantum Instance module """

from typing import Optional, List, Union, Dict, Callable
import warnings

from qiskit.exceptions import QiskitError
from qiskit.utils import QuantumInstance as QQuantumInstance
from qiskit.providers import Backend, BaseBackend
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.layout import Layout
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

try:
    from qiskit.providers.aer.noise import NoiseModel  # pylint: disable=unused-import
except ImportError as ex:
    pass

from .aqua_error import AquaError


class QuantumInstance(QQuantumInstance):
    """Quantum Backend including execution setting."""

    def __init__(self,
                 backend: Union[Backend, BaseBackend],
                 # run config
                 shots: int = 1024,
                 seed_simulator: Optional[int] = None,
                 max_credits: int = 10,
                 # backend properties
                 basis_gates: Optional[List[str]] = None,
                 coupling_map: Optional[Union[CouplingMap, List[List]]] = None,
                 # transpile
                 initial_layout: Optional[Union[Layout, Dict, List]] = None,
                 pass_manager: Optional[PassManager] = None,
                 seed_transpiler: Optional[int] = None,
                 optimization_level: Optional[int] = None,
                 # simulation
                 backend_options: Optional[Dict] = None,
                 noise_model: Optional['NoiseModel'] = None,
                 # job
                 timeout: Optional[float] = None,
                 wait: float = 5.,
                 # others
                 skip_qobj_validation: bool = True,
                 measurement_error_mitigation_cls: Optional[Callable] = None,
                 cals_matrix_refresh_period: int = 30,
                 measurement_error_mitigation_shots: Optional[int] = None,
                 job_callback: Optional[Callable] = None) -> None:
        """
        Quantum Instance holds a Qiskit Terra backend as well as configuration for circuit
        transpilation and execution. When provided to an Aqua algorithm the algorithm will
        execute the circuits it needs to run using the instance.

        Args:
            backend: Instance of selected backend
            shots: Number of repetitions of each circuit, for sampling
            seed_simulator: Random seed for simulators
            max_credits: Maximum credits to use
            basis_gates: List of basis gate names supported by the
                                               target. Defaults to basis gates of the backend.
            coupling_map: Coupling map (perhaps custom) to
                                                      target in mapping
            initial_layout: Initial layout of qubits in mapping
            pass_manager: Pass manager to handle how to compile the circuits
            seed_transpiler: The random seed for circuit mapper
            optimization_level: How much optimization to perform on the circuits.
                Higher levels generate more optimized circuits, at the expense of longer
                transpilation time.
            backend_options: All running options for backend, please refer
                to the provider of the backend for information as to what options it supports.
            noise_model: noise model for simulator
            timeout: Seconds to wait for job. If None, wait indefinitely.
            wait: Seconds between queries for job result
            skip_qobj_validation: Bypass Qobj validation to decrease circuit
                processing time during submission to backend.
            measurement_error_mitigation_cls: The approach to mitigate
                measurement errors. Qiskit Ignis provides fitter classes for this functionality
                and CompleteMeasFitter from qiskit.ignis.mitigation.measurement module can be used
                here. (TensoredMeasFitter is not supported).
            cals_matrix_refresh_period: How often to refresh the calibration
                matrix in measurement mitigation. in minutes
            measurement_error_mitigation_shots: The number of shots number for
                building calibration matrix. If None, the main `shots` parameter value is used.
            job_callback: Optional user supplied callback which can be used
                to monitor job progress as jobs are submitted for processing by an Aqua algorithm.
                The callback is provided the following arguments: `job_id, job_status,
                queue_position, job`
        Raises:
            AqyaError: the shots exceeds the maximum number of shots
            AquaError: set noise model but the backend does not support that
            AquaError: set backend_options but the backend does not support that
        """
        warnings.warn(
            'qiskit.aqua.QuantumInstance is deprecated '
            'as of 0.9.0, and will be '
            'removed no earlier than 3 months after the release date. '
            'Please use qiskit.utils.QuantumInstance.',
            DeprecationWarning)

        try:
            super().__init__(backend,
                             shots,
                             seed_simulator,
                             max_credits,
                             basis_gates,
                             coupling_map,
                             initial_layout,
                             pass_manager,
                             seed_transpiler,
                             optimization_level,
                             backend_options,
                             noise_model,
                             timeout,
                             wait,
                             skip_qobj_validation,
                             measurement_error_mitigation_cls,
                             cals_matrix_refresh_period,
                             measurement_error_mitigation_shots,
                             job_callback)
        except QiskitError as ex:
            raise AquaError(str(ex)) from ex

    def execute(self,
                circuits: Union[QuantumCircuit, List[QuantumCircuit]],
                had_transpiled: bool = False) -> Result:
        """
        A wrapper to interface with quantum backend.
        Args:
            circuits: circuits to execute
            had_transpiled: whether or not circuits had been transpiled
        Returns:
            Result object
        Raises:
            AquaError: execute error
        """
        try:
            return super().execute(circuits, had_transpiled)
        except QiskitError as ex:
            raise AquaError(str(ex)) from ex

    def set_config(self, **kwargs):
        """Set configurations for the quantum instance."""
        try:
            super().set_config(**kwargs)
        except QiskitError as ex:
            raise AquaError(str(ex)) from ex
