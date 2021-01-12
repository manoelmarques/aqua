# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" QEomVQE algorithm """

import logging

from typing import Union, List, Optional, Callable, Dict
import warnings
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers import Backend
from qiskit.algorithms import VQE, MinimumEigensolverResult
from qiskit.aqua.algorithms import VQEResult
from qiskit.aqua.algorithms import VQE as OldVQE
from qiskit.utils.validation import validate_min, validate_in_set
from qiskit.aqua import AquaError
from qiskit.aqua.operators import LegacyBaseOperator as OldLegacyBaseOperator
from qiskit.opflow import LegacyBaseOperator, OperatorBase, I
from qiskit.aqua.operators import OperatorBase as OldOperatorBase
from qiskit.aqua.operators import I as OldI
from qiskit.aqua.components.variational_forms import VariationalForm as OldVariationalForm
from qiskit.algorithms.variational_forms import VariationalForm
from qiskit.aqua.components.optimizers import Optimizer as OldOptimizer
from qiskit.algorithms.optimizers import Optimizer
from qiskit.aqua.operators import Z2Symmetries as OldZ2Symmetries
from qiskit.opflow import Z2Symmetries
from qiskit.utils import QuantumInstance
from qiskit.aqua import aqua_globals
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomVQE(VQE):
    """ QEomVQE algorithm """

    def __init__(self,
                 operator: Optional[Union[OldLegacyBaseOperator,
                                          LegacyBaseOperator]] = None,
                 var_form:
                 Optional[Union[QuantumCircuit,
                                Union[OldVariationalForm,
                                      VariationalForm]]] = None,
                 optimizer: Optional[Union[OldOptimizer,
                                           Optimizer]] = None,
                 num_orbitals: Optional[int] = 0,
                 num_particles: Optional[Union[List[int], int]] = None,
                 initial_point: Optional[np.ndarray] = None,
                 max_evals_grouped: int = 1,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 is_eom_matrix_symmetric: bool = True,
                 active_occupied: Optional[List[int]] = None,
                 active_unoccupied: Optional[List[int]] = None,
                 se_list: Optional[List[List[int]]] = None,
                 de_list: Optional[List[List[int]]] = None,
                 z2_symmetries: Optional[Union[OldZ2Symmetries,
                                               Z2Symmetries]] = None,
                 untapered_op: Optional[Union[OldLegacyBaseOperator,
                                              LegacyBaseOperator]] = None,
                 aux_operators: Optional[List[Union[OldLegacyBaseOperator,
                                                    LegacyBaseOperator]]] = None,
                 quantum_instance: Optional[
                     Union[QuantumInstance,
                           Backend, BaseBackend]] = None) -> None:
        """
        Args:
            operator: qubit operator
            var_form: parameterized variational form.
            optimizer: the classical optimization algorithm.
            num_orbitals:  total number of spin orbitals, has a min. value of 1.
            num_particles: number of particles, if it is a list,
                                              the first number is
                                              alpha and the second number if beta.
            initial_point: optimizer initial point, 1-D vector
            max_evals_grouped: max number of evaluations performed simultaneously
            callback: a callback that can access the intermediate data during
                                 the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            qubit_mapping: qubit mapping type
            two_qubit_reduction: two qubit reduction is applied or not
            is_eom_matrix_symmetric: is EoM matrix symmetric
            active_occupied: list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied: list of unoccupied orbitals to include, indices are
                                      0 to m where m is (num_orbitals - num particles) // 2
            se_list: single excitation list, overwrite the setting in active space
            de_list: double excitation list, overwrite the setting in active space
            z2_symmetries: represent the Z2 symmetries
            untapered_op: if the operator is tapered, we need untapered operator
                                         during building element of EoM matrix
            aux_operators: Auxiliary operators to be evaluated at each eigenvalue
            quantum_instance: Quantum Instance or Backend
        Raises:
            ValueError: invalid parameter
        """
        validate_min('num_orbitals', num_orbitals, 1)
        validate_in_set('qubit_mapping', qubit_mapping,
                        {'jordan_wigner', 'parity', 'bravyi_kitaev'})
        if operator is not None:
            warnings.warn(
                'The operator parameter is deprecated '
                'as of 0.9.0, and will be '
                'removed no earlier than 3 months after the release date.',
                DeprecationWarning)
        if aux_operators is not None:
            warnings.warn(
                'The aux_operators parameter is deprecated '
                'as of 0.9.0, and will be '
                'removed no earlier than 3 months after the release date.',
                DeprecationWarning)
        if isinstance(num_particles, list) and len(num_particles) != 2:
            raise ValueError('Num particles value {}. Number of values allowed is 2'.format(
                num_particles))
        self._old_algorithm = None
        if aqua_globals.deprecated_code:
            self._old_algorithm = OldVQE(operator,
                                         var_form, optimizer, initial_point=initial_point,
                                         max_evals_grouped=max_evals_grouped,
                                         aux_operators=aux_operators,
                                         callback=callback,
                                         quantum_instance=quantum_instance)
        super().__init__(var_form, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped,
                         callback=callback,
                         quantum_instance=quantum_instance)

        self._operator = None
        if operator is not None:
            self.operator = operator

        self.aux_operators = aux_operators

        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles,
                                      qubit_mapping, two_qubit_reduction, active_occupied,
                                      active_unoccupied,
                                      is_eom_matrix_symmetric, se_list, de_list,
                                      z2_symmetries, untapered_op)
        self._num_orbitals = num_orbitals
        self._num_particles = num_particles
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._active_occupied = active_occupied
        self._active_unoccupied = active_unoccupied
        self._is_eom_matrix_symmetric = is_eom_matrix_symmetric
        self._se_list = se_list
        self._de_list = de_list
        self._z2_symmetries = z2_symmetries
        self._untapered_op = untapered_op

    def run(self,
            quantum_instance: Optional[Union[QuantumInstance,
                                             Backend, BaseBackend]] = None,
            **kwargs) -> Dict:
        """Execute the algorithm with selected backend.
        Args:
            quantum_instance: the experimental setting.
            kwargs (dict): kwargs
        Returns:
            dict: results of an algorithm.
        Raises:
            AquaError: If a quantum instance or backend has not been provided
        """
        warnings.warn(
            'The method run is deprecated '
            'as of 0.9.0, and will be '
            'removed no earlier than 3 months after the release date.'
            'Please use compute_minimum_eigenvalue.',
            DeprecationWarning)

        if aqua_globals.deprecated_code:
            ret = self._old_algorithm.run(quantum_instance, **kwargs)
            return self._post_process(ret)
        else:
            if quantum_instance is None and self.quantum_instance is None:
                raise AquaError("A QuantumInstance or Backend "
                                "must be supplied to run the quantum algorithm.")
            if quantum_instance is not None:
                self.quantum_instance = quantum_instance
        return self.compute_minimum_eigenvalue(self.operator, self.aux_operators)

    def compute_minimum_eigenvalue(
                self,
                operator: OperatorBase,
                aux_operators: Optional[List[Optional[OperatorBase]]] = None
    ) -> MinimumEigensolverResult:
        self.qeom = QEquationOfMotion(operator,
                                      self._num_orbitals, self._num_particles,
                                      self._qubit_mapping,
                                      self._two_qubit_reduction, self._active_occupied,
                                      self._active_unoccupied,
                                      self._is_eom_matrix_symmetric,
                                      self._se_list, self._de_list,
                                      self._z2_symmetries, self._untapered_op)
        if isinstance(operator, LegacyBaseOperator):
            operator = operator.to_opflow()
        if aux_operators is not None:
            aux_operators = [o_p.to_opflow()
                             if isinstance(o_p, LegacyBaseOperator)
                             else o_p for o_p in aux_operators]
        ret = super().compute_minimum_eigenvalue(operator,
                                                 aux_operators=aux_operators)
        return self._post_process(ret)

    def _post_process(self, ret: MinimumEigensolverResult) -> 'QEomVQEResult':
        self.quantum_instance.circuit_summary = True
        opt_params = ret.optimal_point
        logger.info("opt params:\n%s", opt_params)
        if aqua_globals.deprecated_code:
            wave_fn = self._old_algorithm.get_optimal_circuit()
        else:
            wave_fn = self.get_optimal_circuit()
        excitation_energies_gap, eom_matrices = self.qeom.calculate_excited_states(
            wave_fn, quantum_instance=self.quantum_instance)
        excitation_energies = excitation_energies_gap + ret.eigenvalue.real
        all_energies = np.concatenate(([ret.eigenvalue.real], excitation_energies))
        self._ret = QEomVQEResult(ret)
        self._ret.energy_gap = excitation_energies_gap
        self._ret.energies = all_energies
        self._ret.eom_matrices = eom_matrices
        return self._ret

    @property
    def random(self):
        """Return a numpy random."""
        return self._old_algorithm.random

    # pylint: disable=arguments-differ
    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """ Returns quantum instance. """
        if aqua_globals.deprecated_code:
            return self._old_algorithm.quantum_instance
        else:
            return super().quantum_instance

    # pylint: disable=no-member
    @quantum_instance.setter
    def quantum_instance(self, value: Union[QuantumInstance,
                                            BaseBackend, Backend]) -> None:
        if aqua_globals.deprecated_code:
            self._old_algorithm.quantum_instance = value
        else:
            super(QEomVQE, self.__class__).quantum_instance.__set__(self, value)

    @property
    def operator(self) -> Optional[Union[OldOperatorBase,
                                         OperatorBase,
                                         OldLegacyBaseOperator,
                                         LegacyBaseOperator]]:
        """ return operator """
        if aqua_globals.deprecated_code:
            return self._old_algorithm.operator
        else:
            return self._operator

    @operator.setter
    def operator(self, operator: Union[OldOperatorBase,
                                       OperatorBase,
                                       OldLegacyBaseOperator,
                                       LegacyBaseOperator]) -> None:
        """ set operator """
        if aqua_globals.deprecated_code:
            self._old_algorithm.operator = operator
            return

        if isinstance(operator, (OldLegacyBaseOperator,
                                 LegacyBaseOperator)):
            operator = operator.to_opflow()
        self._operator = operator
        self._expect_op = None
        self._check_operator_varform(operator)
        # Expectation was not passed by user, try to create one
        if not self._user_valid_expectation:
            self._try_set_expectation_value_from_factory(operator)

    @ property
    def aux_operators(self) -> \
            Optional[List[Optional[Union[OldOperatorBase,
                                         OperatorBase,
                                         OldLegacyBaseOperator,
                                         LegacyBaseOperator]]]]:
        """ Returns aux operators """
        if aqua_globals.deprecated_code:
            return self._old_algorithm.aux_operators
        else:
            return self._aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators: Optional[
                          Union[OldOperatorBase,
                                OperatorBase,
                                OldLegacyBaseOperator,
                                LegacyBaseOperator,
                                List[Optional[Union[OldOperatorBase,
                                                    OperatorBase,
                                                    OldLegacyBaseOperator,
                                                    LegacyBaseOperator]]]]]) -> None:
        """ Set aux operators """
        if aqua_globals.deprecated_code:
            self._old_algorithm.aux_operators = aux_operators
            return

        if aux_operators is None:
            aux_operators = []
        elif not isinstance(aux_operators, list):
            aux_operators = [aux_operators]

        # We need to handle the array entries being Optional i.e. having value None
        self._aux_op_nones = [op is None for op in aux_operators]
        if aux_operators:
            if aqua_globals.deprecated_code:
                zero_op = OldI.tensorpower(self.operator.num_qubits) * 0.0
            else:
                zero_op = I.tensorpower(self.operator.num_qubits) * 0.0
            converted = []
            for op in aux_operators:
                if op is None:
                    converted.append(zero_op)
                elif isinstance(op, (OldLegacyBaseOperator,
                                     LegacyBaseOperator)):
                    converted.append(op.to_opflow())
                else:
                    converted.append(op)

            # For some reason Chemistry passes aux_ops with 0 qubits and paulis sometimes.
            aux_operators = [zero_op if op == 0 else op for op in converted]

        self._aux_operators = aux_operators  # type: List


class QEomVQEResult(VQEResult):
    """The results class for the QEomVQE algorithm."""

    @property
    def energy_gap(self):
        """ returns energy gap value """
        return self.get('energy_gap')

    @energy_gap.setter
    def energy_gap(self, value):
        """ set energy gap value """
        self.data['energy_gap'] = value

    @property
    def energies(self):
        """ returns energies value """
        return self.get('energies')

    @energies.setter
    def energies(self, value):
        """ set energies value """
        self.data['energies'] = value

    @property
    def eom_matrices(self):
        """ returns eom matrices value """
        return self.get('eom_matrices')

    @eom_matrices.setter
    def eom_matrices(self, value):
        """ set eom matrices value """
        self.data['eom_matrices'] = value

    def __getitem__(self, key: object) -> object:
        if key == 'energies':
            return self.data['energies']

        return super().__getitem__(key)
