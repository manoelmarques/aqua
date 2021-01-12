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

""" Aqua Globals """

from typing import Optional
import warnings

import numpy as np
from qiskit.utils import aqua_globals as qiskit_globals


class QiskitAquaGlobals:
    """Aqua class for global properties."""

    CPU_COUNT = qiskit_globals.CPU_COUNT

    _DEPR_MSG = ('The qiskit.aqua.aqua_globals is deprecated'
                 ' as of 0.9.0, and will be'
                 ' removed no earlier than 3 months after the release date.'
                 ' Please use qiskit.utils.aqua_globals.')

    def __init__(self):
        self._deprecated_code = True
        self._showed_deprecation = False

    @property
    def deprecated_code(self) -> bool:
        """Return if should use deprecated code."""
        return self._deprecated_code

    @deprecated_code.setter
    def deprecated_code(self, value: bool) -> None:
        """Set if should use deprecated code."""
        self._deprecated_code = value

    @property
    def random_seed(self) -> Optional[int]:
        """Return random seed."""
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        return getattr(qiskit_globals, 'random_seed')

    @random_seed.setter
    def random_seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        setattr(qiskit_globals, 'random_seed', seed)

    @property
    def num_processes(self) -> int:
        """Return num processes."""
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        return getattr(qiskit_globals, 'num_processes')

    @num_processes.setter
    def num_processes(self, num_processes: Optional[int]) -> None:
        """Set num processes.
           If 'None' is passed, it resets to QiskitAquaGlobals.CPU_COUNT
        """
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        setattr(qiskit_globals, 'num_processes', num_processes)

    @property
    def random(self) -> np.random.Generator:
        """Return a numpy np.random.Generator (default_rng)."""
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        return getattr(qiskit_globals, 'random')

    @property
    def massive(self) -> bool:
        """Return massive to allow processing of large matrices or vectors."""
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        return getattr(qiskit_globals, 'massive')

    @massive.setter
    def massive(self, massive: bool) -> None:
        """Set massive to allow processing of large matrices or  vectors."""
        if not self._showed_deprecation:
            self._showed_deprecation = True
            warnings.warn(QiskitAquaGlobals._DEPR_MSG, DeprecationWarning)
        setattr(qiskit_globals, 'massive', massive)


# Global instance to be used as the entry point for globals.
aqua_globals = QiskitAquaGlobals()  # pylint: disable=invalid-name
