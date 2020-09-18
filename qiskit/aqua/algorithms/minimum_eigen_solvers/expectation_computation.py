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

""" Expectation Computation """

from typing import Optional, Union, Dict, List, Tuple, Any
from abc import ABC, abstractmethod
import importlib
import logging
import dill
import socketio
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.operators import (CircuitSampler, OperatorBase, ExpectationBase,
                                   CircuitStateFn, LegacyBaseOperator, ExpectationFactory)
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.utils.backend_utils import is_aer_provider
from qiskit.aqua.operators import StateFn

logger = logging.getLogger(__name__)


class ExpectationComputation(ABC):
    """Expectation Computation"""

    @abstractmethod
    def compute_expectation_value(self,
                                  parameters: Union[List[float], np.ndarray]
                                  ) -> Tuple[np.ndarray, OperatorBase, List[float]]:
        """ Computes expectation value"""
        raise NotImplementedError()

    @abstractmethod
    def terminate(self):
        """ Terminate """
        raise NotImplementedError()


class LocalExpectationComputation(ExpectationComputation):
    """Local Expectation Computation"""

    def __init__(self,
                 quantum_instance: QuantumInstance,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 var_form: Optional[Union[QuantumCircuit, VariationalForm]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False) -> None:
        """
        Args:
            quantum_instance: Quantum Instance.
            operator: Qubit operator of the Observable
            var_form: A parameterized circuit used as Ansatz for the wave function.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When ``None`` (the default) an
                :class:`~qiskit.aqua.operators.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
        """
        self._operator = None
        if operator is not None:
            self.operator = operator
        self._var_form = var_form
        self._var_form_params = None
        if var_form is not None:
            self.var_form = var_form

        self._include_custom = include_custom
        self._quantum_instance = quantum_instance
        self._circuit_sampler = None
        self._expectation = expectation
        self._expect_op = None

    @property
    def quantum_instance(self) -> QuantumInstance:
        """ Returns quantum instance. """
        return self._quantum_instance

    @property
    def var_form(self) -> Optional[Union[QuantumCircuit, VariationalForm]]:
        """ Returns variational form """
        return self._var_form

    @var_form.setter
    def var_form(self, var_form: Optional[Union[QuantumCircuit, VariationalForm]]):
        """ Sets variational form """
        if isinstance(var_form, QuantumCircuit):
            # store the parameters
            self._var_form_params = sorted(var_form.parameters, key=lambda p: p.name)
            self._var_form = var_form
        elif isinstance(var_form, VariationalForm):
            self._var_form_params = ParameterVector('θ', length=var_form.num_parameters)
            self._var_form = var_form
        elif var_form is None:
            self._var_form_params = None
            self._var_form = var_form
        else:
            raise ValueError('Unsupported type "{}" of var_form'.format(type(var_form)))

    def _try_set_expectation_value_from_factory(self) -> None:
        if self.operator is not None and self.quantum_instance is not None:
            self._set_expectation(ExpectationFactory.build(operator=self.operator,
                                                           backend=self.quantum_instance,
                                                           include_custom=self._include_custom))

    def _set_expectation(self, exp: ExpectationBase) -> None:
        self._expectation = exp
        self._expect_op = None

    def _check_operator_varform(self):
        """Check that the number of qubits of operator and variational form match."""
        if self.operator is not None and self.var_form is not None:
            if self.operator.num_qubits != self.var_form.num_qubits:
                # try to set the number of qubits on the variational form, if possible
                try:
                    self.var_form.num_qubits = self.operator.num_qubits
                    self._var_form_params = sorted(self.var_form.parameters, key=lambda p: p.name)
                except AttributeError as ex:
                    raise AquaError("The number of qubits of the variational form does not match "
                                    "the operator, and the variational form does not allow setting "
                                    "the number of qubits using `num_qubits`.") from ex

    @property
    def circuit_sampler(self):
        """return circuit sampler"""
        if self._circuit_sampler is None:
            self._circuit_sampler = CircuitSampler(
                self._quantum_instance,
                param_qobj=is_aer_provider(self._quantum_instance.backend))
        return self._circuit_sampler

    def construct_expectation(self,
                              parameter: Union[List[float], List[Parameter], np.ndarray]
                              ) -> OperatorBase:
        r"""
        Generate the ansatz circuit and expectation value measurement, and return their
        runnable composition.

        Args:
            parameter: Parameters for the ansatz circuit.

        Returns:
            The Operator equalling the measurement of the ansatz :class:`StateFn` by the
            Observable's expectation :class:`StateFn`.

        Raises:
            AquaError: If no operator has been provided.
        """
        if self.operator is None:
            raise AquaError("The operator was never provided.")

        # ensure operator and varform are compatible
        self._check_operator_varform()

        if isinstance(self.var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, parameter))  # type: Dict
            wave_function = self.var_form.assign_parameters(param_dict)
        else:
            wave_function = self.var_form.construct_circuit(parameter)

        # Expectation was never created, try to create one
        if self._expectation is None:
            self._try_set_expectation_value_from_factory()

        # If setting the expectation failed, raise an Error:
        if self._expectation is None:
            raise AquaError('No expectation set and could not automatically set one, please '
                            'try explicitly setting an expectation or specify a backend so it '
                            'can be chosen automatically.')

        observable_meas = self._expectation.convert(StateFn(self.operator, is_measurement=True))
        ansatz_circuit_op = CircuitStateFn(wave_function)
        return observable_meas.compose(ansatz_circuit_op).reduce()

    def compute_expectation_value(self,
                                  parameters: Union[List[float], np.ndarray]
                                  ) -> Tuple[np.ndarray, OperatorBase, List[float]]:

        if not self._expect_op:
            self._expect_op = self.construct_expectation(self._var_form_params)

        num_parameters = self.var_form.num_parameters
        if self._var_form.num_parameters == 0:
            raise AquaError('The var_form cannot have 0 parameters.')

        parameter_sets = np.reshape(parameters, (-1, num_parameters))
        # Create dict associating each parameter with the lists of parameterization values for it
        param_bindings = dict(zip(self._var_form_params,
                                  parameter_sets.transpose().tolist()))  # type: Dict

        sampled_expect_op = self.circuit_sampler.convert(self._expect_op, params=param_bindings)
        means = np.real(sampled_expect_op.eval())
        return parameter_sets, sampled_expect_op, means

    def terminate(self):
        pass


class RemoteExpectationComputation(ExpectationComputation):
    """Remote Expectation Computation"""

    def __init__(self,
                 quantum_instance: QuantumInstance,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 var_form: Optional[Union[QuantumCircuit, VariationalForm]] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False) -> None:
        """
        Args:
            quantum_instance: Quantum Instance.
            operator: Qubit operator of the Observable
            var_form: A parameterized circuit used as Ansatz for the wave function.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When ``None`` (the default) an
                :class:`~qiskit.aqua.operators.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.

        Raises:
            AquaError: remote server failed
        """
        operator = ExpectationComputationFactory.dumps(operator)
        var_form = ExpectationComputationFactory.dumps(var_form)
        expectation = ExpectationComputationFactory.dumps(expectation)
        backend = quantum_instance.backend
        backend_name = backend.name()
        clz = backend.provider().__class__
        provider_module_name = clz.__module__
        provider_class_name = clz.__name__
        quantum_instance._backend = None
        _repr_quantum_instance = ExpectationComputationFactory.dumps(quantum_instance)
        quantum_instance._backend = backend
        self._sio = socketio.Client()
        self._sio.connect('http://127.0.0.1:8080',
                          namespaces=['/vqe'])

        result = self._sio.call('create_client',
                                {
                                    'operator': operator,
                                    'var_form': var_form,
                                    'expectation': expectation,
                                    'include_custom': include_custom,
                                    'backend_name': backend_name,
                                    'provider_module_name': provider_module_name,
                                    'provider_class_name': provider_class_name,
                                    'quantum_instance': _repr_quantum_instance,
                                },
                                namespace='/vqe')
        if not result['status']:
            raise AquaError(
                'Call create_client to remote server failed. {}'.format(result['error']))

    def __del__(self):
        self.terminate()

    def compute_expectation_value(self,
                                  parameters: Union[List[float], np.ndarray]
                                  ) -> Tuple[np.ndarray, OperatorBase, List[float]]:
        if self._sio is None:
            raise AquaError(
                'Call compute_expectation_value to remote server failed: Client is disconnected.')

        result = self._sio.call('compute_expectation_value',
                                ExpectationComputationFactory.dumps(parameters),
                                namespace='/vqe')
        if not result['status']:
            raise AquaError(
                'Call compute_expectation_value to remote server failed. {}'.format(
                    result['error']))

        return (ExpectationComputationFactory.loads(result['parameter_sets']),
                ExpectationComputationFactory.loads(result['sampled_expect_op']),
                ExpectationComputationFactory.loads(result['means']))

    def terminate(self):
        """ Terminate """
        if self._sio is not None:
            sio = self._sio
            self._sio = None
            sio.disconnect()


class ExpectationComputationFactory:
    """ Expectation Computation Factory """

    @staticmethod
    def create_instance(remote: bool,
                        quantum_instance: QuantumInstance,
                        operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                        var_form: Optional[Union[QuantumCircuit, VariationalForm]] = None,
                        expectation: Optional[ExpectationBase] = None,
                        include_custom: bool = False) \
            -> ExpectationComputation:
        """creates a ExpectationComputation class"""
        return RemoteExpectationComputation(quantum_instance, operator, var_form,
                                            expectation, include_custom) \
            if remote else LocalExpectationComputation(quantum_instance, operator, var_form,
                                                       expectation, include_custom)

    @staticmethod
    def create_local_instance(data: Dict) -> LocalExpectationComputation:
        """creates a LocalExpectationComputation class"""
        operator = ExpectationComputationFactory.loads(data['operator'])
        var_form = ExpectationComputationFactory.loads(data['var_form'])
        expectation = ExpectationComputationFactory.loads(data['expectation'])
        provider_module = importlib.import_module(data['provider_module_name'])
        provider_class = getattr(provider_module, data['provider_class_name'])
        backend = provider_class().get_backend(data['backend_name'])
        quantum_instance = ExpectationComputationFactory.loads(data['quantum_instance'])
        quantum_instance._backend = backend
        return LocalExpectationComputation(
            quantum_instance=quantum_instance,
            operator=operator,
            var_form=var_form,
            expectation=expectation,
            include_custom=data['include_custom'])

    @staticmethod
    def dumps(arg: Any) -> bytes:
        """ dumps data """
        return dill.dumps(arg, protocol=4)

    @staticmethod
    def loads(arg: bytes) -> Any:
        """ loads data """
        return dill.loads(arg)
