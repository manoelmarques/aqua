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

""" QEomEE algorithm """

from typing import Union, List, Optional
import warnings
import logging

import numpy as np
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.algorithms import NumPyMinimumEigensolver  \
    as OldNumPyMinimumEigensolver
from qiskit.utils.validation import validate_min, validate_in_set
from qiskit.aqua.operators import LegacyBaseOperator as OldLegacyBaseOperator
from qiskit.opflow import LegacyBaseOperator
from qiskit.aqua.operators import Z2Symmetries as OldZ2Symmetries
from qiskit.opflow import Z2Symmetries, OperatorBase
from qiskit.aqua.algorithms import MinimumEigensolverResult
from qiskit.aqua import aqua_globals
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomEE(NumPyMinimumEigensolver):
    """ QEomEE algorithm (classical) """

    def __init__(self,
                 operator: Optional[Union[OldLegacyBaseOperator,
                                          LegacyBaseOperator]] = None,
                 num_orbitals: Optional[int] = 0,
                 num_particles: Optional[Union[List[int], int]] = None,
                 qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 active_occupied: Optional[List[int]] = None,
                 active_unoccupied: Optional[List[int]] = None,
                 is_eom_matrix_symmetric: bool = True,
                 se_list: Optional[List[List[int]]] = None,
                 de_list: Optional[List[List[int]]] = None,
                 z2_symmetries: Optional[Union[OldZ2Symmetries,
                                               Z2Symmetries]] = None,
                 untapered_op: Optional[Union[OldLegacyBaseOperator,
                                              LegacyBaseOperator]] = None,
                 aux_operators:
                 Optional[List[Union[OldLegacyBaseOperator,
                                     LegacyBaseOperator]]] = None) -> None:
        """
        Args:
            operator: qubit operator
            num_orbitals:  total number of spin orbitals, has a min. value of 1.
            num_particles: number of particles, if it is a list,
                                        the first number is alpha and the second
                                        number if beta.
            qubit_mapping: qubit mapping type
            two_qubit_reduction: two qubit reduction is applied or not
            active_occupied: list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied: list of unoccupied orbitals to include, indices are
                                    0 to m where m is (num_orbitals - num particles) // 2
            is_eom_matrix_symmetric: is EoM matrix symmetric
            se_list: single excitation list, overwrite the setting in active space
            de_list: double excitation list, overwrite the setting in active space
            z2_symmetries: represent the Z2 symmetries
            untapered_op: if the operator is tapered, we need untapered operator
                                         to build element of EoM matrix
            aux_operators: Auxiliary operators to be evaluated at
                                                each eigenvalue
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
        super().__init__()
        self._old_algorithm = OldNumPyMinimumEigensolver(operator, aux_operators)
        self._operator = operator
        self._aux_operators = aux_operators
        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles, qubit_mapping,
                                      two_qubit_reduction, active_occupied, active_unoccupied,
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

    @property
    def random(self):
        """Return a numpy random."""
        return self._old_algorithm.random

    def run(self) -> 'QEomEEResult':
        """Execute the classical algorithm.
        Returns:
            results of an algorithm.
        """
        warnings.warn(
            'The method run is deprecated '
            'as of 0.9.0, and will be '
            'removed no earlier than 3 months after the release date.'
            'Please use compute_minimum_eigenvalue.',
            DeprecationWarning)

        if aqua_globals.deprecated_code:
            ret = self._old_algorithm.run()
            return self._post_process(ret)
        else:
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

    def _post_process(self, ret: MinimumEigensolverResult) -> 'QEomEEResult':
        wave_fn = ret.eigenstate.to_matrix()
        excitation_energies_gap, eom_matrices = self.qeom.calculate_excited_states(wave_fn)
        excitation_energies = excitation_energies_gap + ret.eigenvalue.real
        all_energies = np.concatenate(([ret.eigenvalue.real], excitation_energies))
        self._ret = QEomEEResult(ret)
        self._ret.energy_gap = excitation_energies_gap
        self._ret.energies = all_energies
        self._ret.eom_matrices = eom_matrices
        return self._ret

    @property
    def operator(self) -> Optional[Union[OldLegacyBaseOperator,
                                         LegacyBaseOperator]]:
        """ return operator """
        if aqua_globals.deprecated_code:
            return self._old_algorithm.operator
        else:
            return self._operator

    @operator.setter
    def operator(self, operator: Union[OldLegacyBaseOperator,
                                       LegacyBaseOperator]) -> None:
        """ set operator """
        if aqua_globals.deprecated_code:
            self._old_algorithm.operator = operator
        else:
            self._operator = operator

    @property
    def aux_operators(self) -> \
            Optional[List[Optional[Union[OldLegacyBaseOperator,
                                         LegacyBaseOperator]]]]:
        """ return aux operators """
        if aqua_globals.deprecated_code:
            return self._old_algorithm.aux_operators
        else:
            return self._aux_operators

    @aux_operators.setter
    def aux_operators(self,
                      aux_operators:
                      Optional[List[Optional[Union[OldLegacyBaseOperator,
                                                   LegacyBaseOperator]]]]) -> None:
        """ set aux operators """
        if aqua_globals.deprecated_code:
            self._old_algorithm.aux_operators = aux_operators
        else:
            self._aux_operators = aux_operators


class QEomEEResult(MinimumEigensolverResult):
    """The results class for the QEomEE algorithm."""

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
