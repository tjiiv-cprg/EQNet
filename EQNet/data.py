import yaml
import random


def get_argoverse_v2_root(machine:str):
    with open("EQNet/env_config.yml", "r") as f:
        config = yaml.safe_load(f)
    if machine not in config['machines']:
        raise ValueError(f"Machine {machine} not found in EQNet/env_config.yml")
    ARGOVERSE_V2_ROOT = config['machines'][machine]['argoverse_v2_dataroot']
    return ARGOVERSE_V2_ROOT


def get_argoverse_v2_dataset_for_qcnet(machine:str, split:str):
    
    from qcnet_55cacb4.datasets.argoverse_v2_dataset import ArgoverseV2Dataset
    from qcnet_55cacb4.transforms.target_builder import TargetBuilder

    return ArgoverseV2Dataset(
        root = get_argoverse_v2_root(machine),
        split = split,
        transform=TargetBuilder(50, 60),
        processed_dir=None,
    )


def get_argoverse_v2_val_batch(machine:str, random_=False, batch_size=4):
    from torch_geometric.data.collate import collate
    dataset = get_argoverse_v2_dataset_for_qcnet(machine, split="val")
    try:
        if random_:
            data_indices = [random.randrange(len(dataset)) for _ in range(batch_size)]
        else:
            data_indices = [1073, 1977, 1496, 318, 638, 1379, 1418, 133, 1590, 184, 117, 1301, 1676, 1884, 765, 483, 801, 1213, 928, 1620, 114, 86, 1051, 1872, 694, 1242, 12, 90, 264, 1566, 1090, 1176][:batch_size]
        data_list = [dataset[i] for i in data_indices]
    except IndexError:
        raise "Please redo data preprocessing."

    data, slices, _ = collate(
        dataset[0].__class__,
        data_list=data_list,
    )

    data['agent']['hist_end'] = 50
    data['agent']['num_pred_steps'] = 60

    return data, dataset
