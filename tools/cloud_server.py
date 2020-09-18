# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Cloud Server """

from typing import Dict
import socketio
import aiohttp
from qiskit.aqua.algorithms import ExpectationComputationFactory


class CloudServer(socketio.AsyncNamespace):
    """Cloud Server"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._clients = {}  # type: Dict

    def on_connect(self, sid, environ):  # pylint: disable=unused-argument
        """ server connect """
        print('server connect client: ', sid)

    def on_disconnect(self, sid):
        """ server disconnect """
        del self._clients[sid]
        print('server disconnect client: ', sid)

    def on_create_client(self, sid, data: Dict):
        """ create client """
        try:
            self._clients[sid] = \
                ExpectationComputationFactory.create_local_instance(data)
            return {
                'status': True,
            }
        except Exception as ex:  # pylint: disable=broad-except
            print('on_create_client exception: ', str(ex))
            return {
                'status': False,
                'error': str(ex)
            }

    def on_compute_expectation_value(self, sid, data):
        """ compute expectation value """
        try:
            params = ExpectationComputationFactory.loads(data)
            parameter_sets, sampled_expect_op, means = \
                self._clients.get(sid).compute_expectation_value(params)
            return {
                'status': True,
                'parameter_sets': ExpectationComputationFactory.dumps(parameter_sets),
                'sampled_expect_op': ExpectationComputationFactory.dumps(sampled_expect_op),
                'means': ExpectationComputationFactory.dumps(means),
            }
        except Exception as ex:  # pylint: disable=broad-except
            print('on_compute_expectation_value exception: ', str(ex))
            return {
                'status': False,
                'error': str(ex)
            }


if __name__ == '__main__':
    _SERVER_SIO = socketio.AsyncServer(async_mode='aiohttp',
                                       ping_timeout=360)
    _APP = aiohttp.web.Application()
    _SERVER_SIO.attach(_APP)
    _SERVER_SIO.register_namespace(CloudServer('/vqe'))

    aiohttp.web.run_app(_APP, host='127.0.0.1', port=8080)
