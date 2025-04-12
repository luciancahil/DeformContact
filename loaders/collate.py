import torch
from torch_geometric.data import Batch

def crystal_collate(batch):
    """
    Custom collate function to handle batching of PyTorch Geometric Data objects
    along with None values.

    Args:
        batch (list): A list of tuples, where each tuple contains elements returned by __getitem__.

    Returns:
        tuple: A tuple where:
            - Tensor elements are batched (if any),
            - PyTorch Geometric Data objects are batched into a Batch object,
            - None elements are kept as None.
    """
    # Initialize lists to hold batched elements
    batched_none_0 = None  # First element is always None
    batched_data1 = []
    batched_data2 = []
    batched_none_3 = None  # Fourth element is always None
    batched_none_4 = None  # Fifth element is always None

    # Iterate over each sample in the batch
    for sample in batch:
        # Unpack the tuple
        data1, data2 = sample

        # Append Data objects to their respective lists
        batched_data1.append(data1)
        batched_data2.append(data2)

        # No action needed for None elements as they are consistently None

    # Batch the Data objects using PyTorch Geometric's Batch.from_data_list
    batched_data1 = Batch.from_data_list(batched_data1)
    batched_data2 = Batch.from_data_list(batched_data2)

    # Return the batched tuple
    return (batched_none_0, batched_data1, batched_data2, batched_none_3, batched_none_4)


def collate_fn(batch):
    obj_names, soft_rest_graphs, soft_def_graphs, metas, rigid_graphs = zip(*batch)
    
    obj_names = [name for name in obj_names]

    tensor_meta_keys = [key for key in metas[0].keys() if isinstance(metas[0][key], torch.Tensor)]
    scalar_meta_keys = [key for key in metas[0].keys() if not isinstance(metas[0][key], torch.Tensor)]

    collated_meta = {key: torch.stack([meta[key] for meta in metas]) for key in tensor_meta_keys}
    for key in scalar_meta_keys:
        collated_meta[key] = [meta[key] for meta in metas]

    return obj_names, soft_rest_graphs, soft_def_graphs, collated_meta, rigid_graphs
