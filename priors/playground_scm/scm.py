"""
This file defines all functionality for Structural Causal Models.
"""

from typing import Tuple, Any, Callable, Dict, List
import networkx as nx
import matplotlib.pyplot as plt
import random
import dill
import torch
import numpy as np
from priors.playground_scm.utils_playground_scm import torch_random_choice

class StructuralCausalModel:
    """
    Class defining a Structural Causal Model. We define an SCM as a tuple
    .. math::
        \mathcal{M} = (\mathcal{X}, \mathcal{F}, \mathcal{U}, \mathcal{P})
    where :math:`\mathcal{X}` is the set of endogenous variables, :math:`\mathcal{F}` is the set of assignment
    functions, :math:`\mathcal{U}` the set of exogenous variables, and :math:`\mathcal{P}` is the set of probability
    distributions associated with the exogenous variables.
    """

    endogenous_vars: Dict[str, Any]
    """Dictionary storing the value of each endogenous variable."""
    functions: Dict[str, Tuple[Callable, dict]]
    """Functional assignments of the endogenous variables. for each endogenous variables, the functional assignments are
    stored as well as a dictionary mapping the parameters of the callable (key) to the name of the causes (values)."""
    exogenous_vars: Dict[str, Any]
    """Dictionary storing the values of the exogenous variables."""
    exogenous_distributions: Dict[str, Tuple[Callable, dict]]
    """Dictionary storing the distribution for each exogenous variable. The values of the dict contain the callable 
    representing the distribution and it's kwargs as a tuple."""
    saved_functions: Dict[str, Tuple[Callable, dict]]
    """Contains a backup of the function of each endogenous variable to be able to restore them after intervention."""
    def __init__(self):
        self.endogenous_vars = {}
        self.exogenous_vars = {}
        self.functions = {}
        self.exogenous_distributions = {}
        self.saved_functions = {}

    def add_endogenous_var(self, name: str, function: Callable, param_varnames: dict):
        """
        Adds an endogenous variable to the SCM.

        :param name: name of the endogenous variable
        :param function: callable that returns a value given the causes of the endogenous variables.
        :param param_varnames: dict that maps names of the parameters in the function to the name of the parent node.

        Example:
        >>> scm.add_endogenous_var('A', lambda noise: noise + 5, {'noise': 'U'})
        """
        # all names are uppercase
        name = name.upper()
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.endogenous_vars[name] = None
        self.functions[name] = (function, param_varnames)

    def add_endogenous_vars(self, vars: List[Tuple[str, Callable, dict]]):
        """
        Adds a list of endogenous variables to the SCM.

        :param vars: list of endogenous variables definitions as defined in the 'add_endogenous_var' function.
        """
        [self.add_endogenous_var(v[0], v[1], v[2]) for v in vars]

    def add_exogenous_var(self, name: str, distribution: Callable, distribution_kwargs: dict):
        """
        Add an exogenous variable to the SCM.

        :param name: name of the exogenous variable.
        :param distribution: distribution of the exogenous variable.
        :param distribution_kwargs: kwargs for the distribution

        Example:
        >>> scm.add_exogenous_var('U', random.randint, {'a': 3, 'b': 8})
        """
        # all names are uppercase
        name = name.upper()
        # ensure unique names
        assert name not in self.exogenous_vars.keys(), 'Variable already exists'
        assert name not in self.endogenous_vars.keys(), 'Variable already exists in endogenous vars'

        self.exogenous_vars[name] = None
        self.exogenous_distributions[name] = (distribution, distribution_kwargs)

    def add_exogenous_vars(self, vars: List[Tuple[str, Any, Callable, dict]]):
        """
        Adds a list of exogenous variables to the SCM.

        :param vars: list of exogenous variables definitions as defined in the 'add_exogenous_var' function.
        """
        [self.add_exogenous_var(v[0], v[1], v[2]) for v in vars]

    def set_binarization_params(self, treatment):
        self.t_threshs, self.t1s, self.t2s = [], [], []
        for b in range(treatment.shape[0]):

            treatment[b] = torch.nan_to_num(treatment[b])
            not_min_max = (treatment[b] > treatment[b].min()) & (treatment[b] < treatment[b].max())

            if (not_min_max == False).all():
                self.t_threshs.append(treatment[b][0]), self.t1s.append(treatment[b][0]), self.t2s.append(treatment[b][0])
                continue

            treatment[b] *= self.t_scaling_factor if hasattr(self, 't_scaling_factor') else 1
            
            if self.binary_strategy == 'extreme':
                t_thresh = np.random.choice(treatment[b][not_min_max])  # the threshold cannot be the min or max value of the treatment
                t1 = torch_random_choice(treatment[b][treatment[b] < t_thresh])
                t2 = torch_random_choice(treatment[b][treatment[b] > t_thresh])

            elif self.binary_strategy == 'mean':
                t_thresh = (treatment[b][not_min_max]).mean()
                t1 = (treatment[b][treatment[b] < t_thresh]).mean()
                t2 = (treatment[b][treatment[b] > t_thresh]).mean()

            assert t1 != t2, f'Treatment values are equal! Got {t1} and {t2}. Variance in treatment is {torch.var(treatment[b])}'

            self.t_threshs.append(t_thresh), self.t1s.append(t1), self.t2s.append(t2)
        
        self.t_threshs, self.t1s, self.t2s = torch.Tensor(self.t_threshs), torch.Tensor(self.t1s), torch.Tensor(self.t2s)

    def get_binarized_treatment(self, treatment):

        for b in range(treatment.shape[0]):
            lt_map = treatment[b] < self.t_threshs[b]
            treatment[b][lt_map] = self.t1s[b]
            treatment[b][~lt_map] = self.t2s[b]

        return treatment
    

    def get_zero_one_treatment(self, treatment):

        for b in range(treatment.shape[1]):
            treatment[:, b] = (treatment[:, b] < treatment[:, b].mean()).float()

        return treatment


    def get_next_sample(self, seed=0, exogenous_vars=None, binarize=False, graph = None) -> Tuple[Dict, Dict]:
        """
        Generates an ancestral sample from the joint distribution of endogenous variables by generating a sample of the
        exogenous variables
        .. math::
            u \sim P(\mathcal{U})
        and determining the value of variables :math:`\mathcal{X}` by applying the functions in :math:`\mathcal{F}`
        according to the topological ordering in the causal graph resulting in a sample of the endogenous variables
        .. math::
            x \sim P(\mathcal{X})
        :param seed: seed for the random number generator
        :param exogenous_vars: dictionary of exogenous variables to use instead of sampling them from the distribution
        :param binarize: whether to binarize the treatment variable or not
        :param graph: the causal graph to use if it is already created. If None, the graph will be created from the SCM.



        :return: a sample of endogenous :math:`x` and exogenous variables :math:`u`.
        """
        random.seed(seed)
        # update exogenous vars
        if exogenous_vars is None:
            for key in self.exogenous_vars:
                dist = self.exogenous_distributions[key]
                res = dist[0](**dist[1])
                self.exogenous_vars[key] = res
        else:
            self.exogenous_vars = exogenous_vars

        if binarize and self.t_key in self.exogenous_vars.keys() and exogenous_vars is None:
            self.set_binarization_params(self.exogenous_vars[self.t_key])
            self.exogenous_vars[self.t_key] = self.get_binarized_treatment(self.exogenous_vars[self.t_key])

        # update endogenous vars
        if graph is None:
            structure_model = self.create_graph()
        else:
            structure_model = graph

        for node in nx.topological_sort(structure_model):
            if node in self.exogenous_vars.keys():  # skip exogenous vars
                continue
            # get the values for the parameters needed in the functions
            """
            params = {}
            for param in self.functions[node][1]:  # parameters of functions
                if self.functions[node][1][param] in self.endogenous_vars.keys():
                    params[param] = self.endogenous_vars[self.functions[node][1][param]]
                else:
                    params[param] = self.exogenous_vars[self.functions[node][1][param]]
            # Update variable according to its function and parameters
            """

            var_lookup = {**self.exogenous_vars, **self.endogenous_vars}
            # Get the mapping from parameters to variable names
            param_map = self.functions[node][1]
            # Build the final params dict with a comprehension
            params = {param: var_lookup[param_map[param]] for param in param_map}

            self.endogenous_vars[node] = self.functions[node][0](**params)

            if binarize and self.t_key == node:
                self.set_binarization_params(self.endogenous_vars[node])
                self.endogenous_vars[node] = self.get_binarized_treatment(self.endogenous_vars[node])

        return dict(self.endogenous_vars), dict(self.exogenous_vars)

    def do_interventions(self, interventions: List[Tuple[str, Tuple[Callable, dict]]]):
        """
        Replaces the functions of the scm with the given interventions per endogenous variable. E.g. the intervention
        :math:`do(X_0 = 5, X_1 = X_0+1)` can be implemented with
        >>> scm.do_interventions([("X0", (lambda: 5, {})), ("X1", (lambda x0: x0+1, {'X0':'x0'})])

        :param interventions: List of tuples where every tuple contains the name of the variable to intervene on and a
        callable that represents the new causal function for this variable given its parents as a dict. The dict maps
        the parameters of the callable (key) to the names of the parents in the SCM (value).
        """
        random.seed()
        self.saved_functions = {}
        for interv in interventions:
            if interv:  # interv not None
                self.saved_functions[interv[0]] = self.functions[interv[0]]
                self.functions[interv[0]] = interv[1]

    def undo_interventions(self):
        """
        Restores all functional relations that were deleted in the previous call of `do_interventions`.
        """
        for key, value in self.saved_functions.items():
            self.functions[key] = value
        self.saved_functions.clear()

    def get_intervention_targets(self) -> List[str]:
        """
        Returns a list containing the names of the variables that are currently being intervened on.

        :return: List of intervention targets.
        """
        return list(self.saved_functions.keys())

    def create_graph(self) -> nx.DiGraph:
        """
        Returns the DAG that corresponds to the functional structure of this SCM.

        :return: A causal graph.
        """
        graph = nx.DiGraph()

        # create nodes
        [graph.add_node(var.upper(), type='endo') for var in self.endogenous_vars]
        [graph.add_node(var.upper(), type='exo') for var in self.exogenous_vars]

        for var in self.functions:
            for parent in self.functions[var][1].values():
                if parent.lower() in self.endogenous_vars or parent.upper() in self.endogenous_vars\
                        or parent.lower() in self.exogenous_vars or parent.upper() in self.exogenous_vars:
                    graph.add_edge(parent.upper(), var.upper())

        return graph

    def draw_graph(self):
        """
        Draws the causal graph.
        """
        graph = self.create_graph()
        values = dict(self.endogenous_vars), dict(self.exogenous_vars)
        values = dict(values[0], **values[1])
        nx.draw(graph, arrowsize=20, with_labels=True, node_size=3000, font_size=10,
                labels={key: str(key) + ':\n' + str(values[key]) for key in values}, pos=nx.planar_layout(graph))
        plt.show()

    def save(self, filepath: str, verbose: int = 0) -> None:
        """
        Save the structural causal model to a file

        :param filepath: path to file for saving the SCM
        :param verbose: verbosity level, 0=silent, 1=print to console
        """
        try:
            with open(filepath, 'wb') as file:
                dill.dump(self, file)
            message = f"SCM successfully saved to {filepath}"
        except IOError as e:
            message = f"Error saving object to file: {str(e)}"
        if verbose == 1:
            print(message)

    @staticmethod
    def load(filepath: str, verbose: int = 0) -> "StructuralCausalModel":
        """
        Load an object from a file using dill deserialization.

        :param filepath: The path and filename of the file to load the object from.
        :param verbose: verbosity level, 0=silent, 1=print to console

        :return: the loaded SCM
        """
        try:
            with open(filepath, 'rb') as file:
                obj = dill.load(file)
            return obj
        except IOError as e:
            if verbose == 1:
                print(f"Error loading object from file: {str(e)}")
        except dill.UnpicklingError as e:
            if verbose == 1:
                print(f"Error deserializing object: {str(e)}")
