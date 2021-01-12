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

"""Finance Test Case"""

from test import QiskitBaseTestCase
from qiskit.aqua import aqua_globals


class QiskitFinanceTestCase(QiskitBaseTestCase):
    """Finance Test Case"""

    def setUp(self) -> None:
        super().setUp()
        self._class_location = __file__
        aqua_globals.deprecated_code = False

    def tearDown(self) -> None:
        super().tearDown()
        aqua_globals.deprecated_code = True
