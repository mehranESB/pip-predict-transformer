from torch.utils.data import Dataset as torch_Dataset
from ..utils.transform import mirror_reflect, add_uniform_score


class Dataset(torch_Dataset):
    """
    Dataset class for applying transformations and preparing data for training.
    This class wraps around an existing dataset and applies a series of transformations
    to the data. It is designed to preprocess the data and extract specific input and
    target features required for training machine learning models.
    """

    def __init__(self, dataset, trans_cfg: dict):
        self.dataset = dataset
        self.trans_cfg = trans_cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # retrive data from dataset
        pip_data = self.dataset[index]

        # apply mirror reflect transformation
        param = self.trans_cfg.get("mirror_reflect", None)
        if param:
            pip_data = mirror_reflect(pip_data, **param)

        # add uniform distribution of dist into data
        param = self.trans_cfg.get("add_uniform_score", None)
        if param:
            pip_data = add_uniform_score(pip_data, **param)

        # extract data that are necessary for process
        input_data = pip_data[["Open", "High", "Low", "Close", "X"]].to_numpy()
        target_data = pip_data[["dist", "hilo", "iter", "udist"]].to_numpy()

        retrive_data = {
            "input": input_data,
            "dist": target_data[:, 0],
            "hilo": target_data[:, 1],
            "iter": target_data[:, 2],
            "udist": target_data[:, 3],
        }

        return retrive_data
