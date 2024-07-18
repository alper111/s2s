import numpy as np
import torch


def dict_to_transition(state: dict, next_state: dict) -> tuple[dict, dict, list]:
    """
    Given a (state, next_state) tuple where states are stored
    in a dictionary of modalities where each modality is a dictionary
    of entities, convert it to a dictionary of tensors.

    The non-trivial thing is the addition and deletion of objects. Right now,
    we use skolem arrays (e.g., torch.zeros) to represent the absence of an object
    (i.e., if it's not in the dictionary).

    Parameters
    ----------
    state : dict
        The state dictionary.
    next_state : dict
        The next state dictionary.

    Returns
    -------
    state_v : dict[torch.Tensor]
        The state tensor dictionary.
    next_state_v : dict[torch.Tensor]
        The next state tensor dictionary.
    key_order : list
        The order of keys in the state tensor.
    """
    modalities = [m for m in state.keys() if m != "dimensions"]
    state_dict = {}
    next_state_dict = {}
    key_order = []
    for mode in modalities:
        all_entities = state[mode].keys() | next_state[mode].keys()
        state_arr = []
        next_state_arr = []
        for key in all_entities:
            key_order.append(key)
            if key in state[mode] and state[mode][key] is not None:
                state_arr.append(torch.tensor(state[mode][key], dtype=torch.float32))
            else:
                skolem = torch.zeros(state["dimensions"][mode], dtype=torch.float32)
                state_arr.append(skolem)
            if key in next_state[mode] and next_state[mode][key] is not None:
                next_state_arr.append(torch.tensor(next_state[mode][key], dtype=torch.float32))
            else:
                skolem = torch.zeros(next_state["dimensions"][mode], dtype=torch.float32)
                next_state_arr.append(skolem)
        state_dict[mode] = torch.stack(state_arr)
        next_state_dict[mode] = torch.stack(next_state_arr)
    return state_dict, next_state_dict, key_order


def dictarray_to_transition(state: np.ndarray, next_state: np.ndarray):
    """
    Given an array of (state, next_state) dictionaries, convert it to
    a dictionary of state tensors with shape (n_batch, n_entity, n_dim)
    for each modality, together with the respective mask vector.

    Parameters
    ----------
    state : np.ndarray
        The state array.
    next_state : np.ndarray
        The next state array.

    Returns
    -------
    state_v : dict[torch.Tensor]
        The state tensor dictionary.
    next_state_v : dict[torch.Tensor]
        The next state tensor dictionary.
    """
    max_entity = {}
    modalities = [m for m in state[0].keys() if m != "dimensions"]
    for mode in modalities:
        max_entity[mode] = max([len(s[mode]) for s in state])
    state_dict = {m: [] for m in modalities}
    state_dict["masks"] = {m: [] for m in modalities}
    next_state_dict = {m: [] for m in modalities}
    next_state_dict["masks"] = {m: [] for m in modalities}
    for s_d, sn_d in zip(state, next_state):
        s_v, sn_v = dict_to_transition(s_d, sn_d)
        for mode in modalities:
            n_entity = len(s_v[mode])
            rem_entities = max_entity[mode] - n_entity
            if rem_entities > 0:
                s_v[mode] = torch.nn.functional.pad(s_v[mode], (0, 0, 0, rem_entities), mode="constant")
                sn_v[mode] = torch.nn.functional.pad(sn_v[mode], (0, 0, 0, rem_entities), mode="constant")
            mask = torch.zeros(max_entity[mode], dtype=bool)
            mask[:n_entity] = True
            state_dict[mode].append(s_v[mode])
            state_dict["masks"][mode].append(mask)
            next_state_dict[mode].append(sn_v[mode])
            next_state_dict["masks"][mode].append(mask)
    for mode in modalities:
        state_dict[mode] = torch.stack(state_dict[mode])
        state_dict["masks"][mode] = torch.stack(state_dict["masks"][mode])
        next_state_dict[mode] = torch.stack(next_state_dict[mode])
        next_state_dict["masks"][mode] = torch.stack(next_state_dict["masks"][mode])
    return state_dict, next_state_dict
