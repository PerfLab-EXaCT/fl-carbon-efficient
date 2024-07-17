# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""


from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import logging
import time
import timeit
import pandas as pd
from logging import DEBUG, INFO
import heapq
from typing import Dict, List, Optional, Tuple, Union
import numpy as np 

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class Server:
    """Flower server."""

    def __init__(
        self, *, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        round_times = []  # List to store round times

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        
        # Previous Aggregated Metrics 
        
        previous_metrics_aggregated = None  # Initialize to None
        previous_results_aggregated = None   
        selected_clients_set = set()
        client_selection_count: Dict[str, int] = {}
        Total_Carbon_footprint = 0 
        
        # df_1000 = pd.read_csv('100clientsintensity.csv')
        # client_intensity_dict = df_1000.set_index('client')['average_intensity'].to_dict()
        client_intensity_dict = []
        
        
        matrix_df = pd.read_csv('reduced_matrix_100_columns.csv')
        carbon_matrix = matrix_df.values


        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
                Total_Carbon_footprint = Total_Carbon_footprint, 
                previous_metrics_aggregated=previous_metrics_aggregated,
                previous_results_aggregated=previous_results_aggregated,
                selected_clients_set_fit = selected_clients_set, 
                client_selection_count_fit = client_selection_count,
                client_intensity_dict = client_intensity_dict, 
                carbon_matrix = carbon_matrix
            )
            
            if res_fit:
                parameters_prime, _, _ , _ , _= res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

                    
                    
            # Update the previous_metrics_aggregated with the metrics from the current round
            _, metrics_aggregated, (results, failures), selected_clients_set_fit_return , Total_Carbon_footprint_fit = res_fit
            previous_metrics_aggregated = metrics_aggregated
            previous_results_aggregated = results
            selected_clients_set.update(selected_clients_set_fit_return)
            Total_Carbon_footprint = Total_Carbon_footprint_fit 
            
            print("Client Selected Dictionary: " , client_selection_count)
            print("Total Carbon Intensity: " , Total_Carbon_footprint)
            
            
            # Calculate round time

            if current_round == 1: 
                round_time = timeit.default_timer() - start_time
                round_times.append(round_time)
            else: 
                previous_time = sum(round_times)
                round_time = timeit.default_timer() - start_time - previous_time
                round_times.append(round_time)
            
            history.add_loss_distributed(
                        server_round=current_round, loss=Total_Carbon_footprint
                    )
            history.add_metrics_distributed(
                        server_round=current_round, metrics=metrics_aggregated
                    )
            
            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    round_time,
                    Total_Carbon_footprint
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        
        # Print total time taken for each round
        for idx, round_time in enumerate(round_times, start=1):
            log(INFO, "Round %s took %s seconds", idx, round_time)
        
        return history



# Asynchronous 

#     def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
#         """Run federated averaging for a number of rounds."""
#         history = History()
#         round_times = []

#         log(INFO, "Initializing global parameters")
#         self.parameters = self._get_initial_parameters(timeout=timeout)
#         log(INFO, "Evaluating initial parameters")
#         res = self.strategy.evaluate(0, parameters=self.parameters)
#         if res is not None:
#             log(
#                 INFO,
#                 "initial parameters (loss, other metrics): %s, %s",
#                 res[0],
#                 res[1],
#             )
#             history.add_loss_centralized(server_round=0, loss=res[0])
#             history.add_metrics_centralized(server_round=0, metrics=res[1])

#         log(INFO, "FL starting")
#         start_time = timeit.default_timer()

#         previous_metrics_aggregated = None
#         previous_results_aggregated = None
#         selected_clients_set = set()
#         client_selection_count = {}
#         Total_Carbon_footprint = 0

#         df_1000 = pd.read_csv('100clientsintensity.csv')
#         client_intensity_dict = df_1000.set_index('client')['average_intensity'].to_dict()

#         for current_round in range(1, num_rounds + 1):
#             res_fit = self.fit_round(
#                 server_round=current_round,
#                 timeout=timeout,
#                 Total_Carbon_footprint=Total_Carbon_footprint,
#                 previous_metrics_aggregated=previous_metrics_aggregated,
#                 previous_results_aggregated=previous_results_aggregated,
#                 selected_clients_set_fit=selected_clients_set,
#                 client_selection_count_fit=client_selection_count,
#                 client_intensity_dict=client_intensity_dict
#             )

#             if res_fit:
#                 parameters_prime, _, _, _, _ = res_fit
#                 if parameters_prime:
#                     self.parameters = parameters_prime

#             _, metrics_aggregated, (results, failures), selected_clients_set_fit_return, Total_Carbon_footprint_fit = res_fit
#             previous_metrics_aggregated = metrics_aggregated
#             previous_results_aggregated = results
#             selected_clients_set.update(selected_clients_set_fit_return)
#             Total_Carbon_footprint = Total_Carbon_footprint_fit

#             print("Client Selected Dictionary:", client_selection_count)
#             print("Total Carbon Intensity:", Total_Carbon_footprint)

#             if current_round == 1:
#                 round_time = timeit.default_timer() - start_time
#                 round_times.append(round_time)
#             else:
#                 previous_time = sum(round_times)
#                 round_time = timeit.default_timer() - start_time - previous_time
#                 round_times.append(round_time)

#             history.add_loss_distributed(server_round=current_round, loss=Total_Carbon_footprint)
#             history.add_metrics_distributed(server_round=current_round, metrics=metrics_aggregated)

#             res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
#             if res_cen is not None:
#                 loss_cen, metrics_cen = res_cen
#                 log(
#                     INFO,
#                     "fit progress: (%s, %s, %s, %s, %s)",
#                     current_round,
#                     loss_cen,
#                     metrics_cen,
#                     round_time,
#                     Total_Carbon_footprint
#                 )
#                 history.add_loss_centralized(server_round=current_round, loss=loss_cen)
#                 history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

#             res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
#             if res_fed:
#                 loss_fed, evaluate_metrics_fed, _ = res_fed
#                 if loss_fed:
#                     history.add_loss_distributed(server_round=current_round, loss=loss_fed)
#                     history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

#         end_time = timeit.default_timer()
#         elapsed = end_time - start_time
#         log(INFO, "FL finished in %s", elapsed)

#         for idx, round_time in enumerate(round_times, start=1):
#             log(INFO, "Round %s took %s seconds", idx, round_time)

#         return history


    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        Total_Carbon_footprint: float, 
        timeout: Optional[float],
        previous_metrics_aggregated: Optional[Dict[str, Scalar]],  # Add this argument
        previous_results_aggregated: List[Tuple[ClientProxy, FitRes]], 
        selected_clients_set_fit: set[ClientProxy], 
        client_selection_count_fit: Dict[str, int], 
        client_intensity_dict: Dict[int, float], 
        carbon_matrix: np.ndarray,
        
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions, selected_clients_set_server_return , Total_Carbon_footprint = self.strategy.configure_fit(
            server_round=server_round,
            Total_Carbon_footprint = Total_Carbon_footprint, 
            parameters=self.parameters,
            client_manager=self._client_manager,
            aggregated_metrics=previous_metrics_aggregated,  # Pass the previous metrics
            aggregated_results=previous_results_aggregated,
            selected_clients_set_configure = selected_clients_set_fit, 
            client_selection_count_configure = client_selection_count_fit,
            client_intensity_dict = client_intensity_dict, 
            carbon_matrix = carbon_matrix,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures) , selected_clients_set_server_return , Total_Carbon_footprint

# Async 
#     def fit_round(
#         self,
#         server_round: int,
#         Total_Carbon_footprint: float,
#         timeout: Optional[float],
#         previous_metrics_aggregated: Optional[Dict[str, Scalar]], 
#         previous_results_aggregated: List[Tuple[ClientProxy, FitRes]],
#         selected_clients_set_fit: set[ClientProxy],
#         client_selection_count_fit: Dict[str, int],
#         client_intensity_dict: Dict[int, float]
#     ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
#         """Perform a single round of federated averaging."""

#         client_instructions, selected_clients_set_server_return, Total_Carbon_footprint = self.strategy.configure_fit(
#             server_round=server_round,
#             Total_Carbon_footprint=Total_Carbon_footprint,
#             parameters=self.parameters,
#             client_manager=self._client_manager,
#             aggregated_metrics=previous_metrics_aggregated,
#             aggregated_results=previous_results_aggregated,
#             selected_clients_set_configure=selected_clients_set_fit,
#             client_selection_count_configure=client_selection_count_fit,
#             client_intensity_dict=client_intensity_dict
#         )

#         if not client_instructions:
#             log(INFO, "fit_round %s: no clients selected, cancel", server_round)
#             return None
#         log(
#             DEBUG,
#             "fit_round %s: strategy sampled %s clients (out of %s)",
#             server_round,
#             len(client_instructions),
#             self._client_manager.num_available(),
#         )

#         results, failures = fit_clients(
#             client_instructions=client_instructions,
#             max_workers=self.max_workers,
#             timeout=timeout,
#         )

#         aggregated_result: Tuple[
#             Optional[Parameters],
#             Dict[str, Scalar],
#         ] = self.strategy.aggregate_fit(server_round, results, failures)

#         parameters_aggregated, metrics_aggregated = aggregated_result
#         return parameters_aggregated, metrics_aggregated, (results, failures), selected_clients_set_server_return, Total_Carbon_footprint



    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client , selected_clients_set_initial_return, Total_Carbon_footprint = self._client_manager.sample(1 , 1 , 2)
        random_client = random_client[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures




# Assuming FitRes includes a status code and possibly other relevant information
# FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[Union[Tuple[ClientProxy, FitRes], BaseException]]]

# def fit_clients(
#     client_instructions: List[Tuple[ClientProxy, FitIns]],
#     max_workers: Optional[int],
#     timeout: Optional[float],
#     n: int = 100,  # number of clients you wish to wait for before proceeding
# ) -> FitResultsAndFailures:
#     """Refine parameters concurrently on all selected clients, considering results from the first n."""
#     results: List[Tuple[ClientProxy, FitRes]] = []
#     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_client = {
#             executor.submit(fit_client, client_proxy, ins, timeout): client_proxy
#             for client_proxy, ins in client_instructions
#         }

#         completed = 0
#         for future in as_completed(future_to_client):
#             client_proxy = future_to_client[future]

#             try:
#                 res = future.result()  # Get the result from the future
#                 if res[1].status.code == Code.OK:
#                     results.append(res)
#                     completed += 1
#                     if completed >= n:  # If we have results from n clients, break
#                         break
#                 else:
#                     failures.append(res)
#             except Exception as e:
#                 logging.exception(f"Failed processing for client {client_proxy}: {e}")
#                 failures.append((client_proxy, e))
            
#         # After breaking out of the loop, we might want to cancel remaining futures
#         # to avoid wasting resources. However, this must be done carefully to avoid
#         # interrupting any operations that might be in a critical state.
#         for future in future_to_client:
#             if not future.done():
#                 future.cancel()  # Attempt to cancel the future

#     return results, failures



# Async 
# def fit_clients(
#     client_instructions: List[Tuple[ClientProxy, FitIns]],
#     max_workers: Optional[int],
#     timeout: Optional[float]
# ) -> FitResultsAndFailures:
#     """Refine parameters asynchronously on all selected clients."""
#     results: List[Tuple[ClientProxy, FitRes]] = []
#     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_client = {
#             executor.submit(fit_client, client_proxy, ins, timeout): client_proxy
#             for client_proxy, ins in client_instructions
#         }

#         for future in concurrent.futures.as_completed(future_to_client):
#             client_proxy = future_to_client[future]

#             try:
#                 res = future.result()
#                 results.append(res)
#             except Exception as e:
#                 logging.exception(f"Failed processing for client {client_proxy}: {e}")
#                 failures.append((client_proxy, e))

#     return results, failures




def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    clientprox, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
