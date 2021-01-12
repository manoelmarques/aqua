# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Chemistry Deprecated Test Case"""

from test.chemistry import QiskitChemistryTestCase
from qiskit.aqua import aqua_globals


class QiskitChemistryDeprecatedTestCase(QiskitChemistryTestCase):
    """Chemistry Deprecated Test Case"""

    def setUp(self) -> None:
        super().setUp()
        aqua_globals.deprecated_code = True

    def tearDown(self) -> None:
        super().tearDown()
        aqua_globals.deprecated_code = False
