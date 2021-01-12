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
"""
This module contains the definition of a base class for
variational forms. Several types of commonly used ansatz.
"""

from typing import Optional, Union, List, Any
import numpy as np
from qiskit import QuantumRegister
from qiskit.algorithms.variational_forms import VariationalForm as VarForm

# pylint: disable=useless-super-delegation


class VariationalForm(VarForm):

    """Base class for VariationalForms."""

    def __init__(self) -> None:
        super().__init__()

    def construct_circuit(self,
                          parameters: Union[List[float], np.ndarray],
                          q: Optional[QuantumRegister] = None) -> Any:
        return super().construct_circuit(parameters, q)
