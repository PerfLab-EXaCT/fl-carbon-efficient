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
"""Abstract class for criterion sampling."""


from abc import ABC, abstractmethod
from .client_proxy import ClientProxy
import random
from typing import Dict, List, Optional, Tuple
from flwr.common import Scalar
import numpy as np
import pandas as pd
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np 

class Criterion(ABC):
    """Abstract class which allows subclasses to implement criterion
    sampling."""
    
    @abstractmethod
    def select(self) -> bool:
        """Decide whether a client should be eligible for sampling or not."""        
        

        
        
class simple_criterion(Criterion):
    def __init__(self , clients={}) -> None:
        self.clients: Dict[str, ClientProxy] = clients
        
    
    def select(
        self,
        round_server_criteria: int,
        Total_Carbon_footprint: float,
        carbon_matrix: np.ndarray,
        client_selection_count_criteria: Dict[str, int] , 
        client_intensity_dict: Dict[int, float], 
        selected_clients_set_criteria: set[ClientProxy] = {},
        aggregated_metrics_criteria: Optional[Dict[str, Scalar]] = None,
        aggregated_results_criteria: List[Tuple[ClientProxy, FitRes]] = None,
    ) -> List[str]:
        """Decide which clients should be eligible for sampling."""

        
        
        
        """      ***************     Client Selection based on Cost     ******************   """
        
        # selected_clients_set_criteria = {}
        # cid_client = list(self.clients)
        
        # sorted_clients = sorted(client_intensity_dict.items(), key=lambda item: item[1])
        # selected_clients_sorted = sorted_clients[:4]
        # print("Sorted Dictionary: " , selected_clients_sorted)
        # selected_clients_idx = [client[0] for client in selected_clients_sorted]

        
        # for client_idx in selected_clients_idx:
        #     selected_clients.append(cid_client[client_idx])
            
#         for client in selected_clients:
# #             client_selection_count_criteria[client] = client_selection_count_criteria.get(client, 0) + 1
#         selected_clients_set_criteria = {}
#         cid_client = list(self.clients)
#         selected_clients = []
#         selected_clients_intensities = []

#         # Function to find the indices of the smallest elements in each row
#         def find_smallest_indices(matrix):
#             smallest_indices = np.argsort(matrix, axis=1)
#             return smallest_indices

#         # Get the indices of the smallest elements in each row
#         smallest_indices = find_smallest_indices(carbon_matrix)
#         low_carbon_clients_indices = smallest_indices[round_server_criteria]

#         # Select clients while considering the selection count
#         for client_idx in low_carbon_clients_indices:
#             client_id = cid_client[client_idx]
#             if client_selection_count_criteria.get(client_idx, 0) <= 10:  # 2 refers to how many times a client can be selected in the whole training process
#                 selected_clients.append(client_id)
#                 selected_clients_intensities.append(carbon_matrix[round_server_criteria][client_idx])
#                 client_selection_count_criteria[client_idx] = client_selection_count_criteria.get(client_idx, 0) + 1
#             if len(selected_clients) >= 10: 
#                 break

#         total_intensity = sum(selected_clients_intensities)
#         Total_Carbon_footprint += total_intensity
#         print("Total Carbon intensity for round ", round_server_criteria, "is: ", total_intensity) 
        

        """       ***************     For random add/drop of clients     ******************   """
        
        selected_clients_set_criteria = {}
        cid_client = list(self.clients)
        selected_clients = []
        selected_clients_intensities = []

        # Select clients randomly while considering the selection count
        client_indices = list(range(len(cid_client)))
        random.shuffle(client_indices)

        for client_idx in client_indices:
            client_id = cid_client[client_idx]
            if client_selection_count_criteria.get(client_idx, 0) <= 10:  # 10 refers to how many times a client can be selected in the whole training process
                selected_clients.append(client_id)
                client_selection_count_criteria[client_idx] = client_selection_count_criteria.get(client_idx, 0) + 1
            if len(selected_clients) >= 10:
                break

        print("Selected clients for round ", round_server_criteria, "are: ", selected_clients)

        return selected_clients, selected_clients_set_criteria, Total_Carbon_footprint

        
        # return selected_clients , selected_clients_set_criteria , Total_Carbon_footprint
    
    
    
    
    
    



