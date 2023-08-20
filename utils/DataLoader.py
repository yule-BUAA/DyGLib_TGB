import os
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray,
                 labels: np.ndarray, interact_types: np.ndarray = None, node_label_times: np.ndarray = None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        :param interact_types: ndarray, its element can be "train", "validate", "test" or "just_update"
        :param node_label_times: ndarray, record the labeled time of nodes (nodes without labels are noted by the interact time)
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.interact_types = interact_types
        self.node_label_times = node_label_times
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


data_num_nodes_map = {
    "tgbl-wiki": 9227,
    "tgbl-review": 352637,
    "tgbl-coin": 638486,
    "tgbl-comment": 994790,
    "tgbl-flight": 18143,
    "tgbn-trade": 255,
    "tgbn-genre": 992,
    "tgbn-reddit": 11068
}

data_num_edges_map = {
    "tgbl-wiki": 157474,
    "tgbl-review": 4873540,
    "tgbl-coin": 22809486,
    "tgbl-comment": 44314507,
    "tgbl-flight": 67169570,
    "tgbn-trade": 507497,
    "tgbn-genre": 17858395,
    "tgbn-reddit": 27174118
}


def get_link_prediction_tgb_data(dataset_name: str):
    """
    generate tgb data for link prediction task
    :param dataset_name: str, dataset name
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object), eval_neg_edge_sampler, eval_metric_name
    """
    # Load data and train val test split
    dataset = LinkPropPredDataset(name=dataset_name, root="datasets", preprocess=True)
    data = dataset.full_data

    src_node_ids = data['sources'].astype(np.longlong)
    dst_node_ids = data['destinations'].astype(np.longlong)
    node_interact_times = data['timestamps'].astype(np.float64)
    edge_ids = data['edge_idxs'].astype(np.longlong)
    labels = data['edge_label']
    edge_raw_features = data['edge_feat'].astype(np.float64)
    # deal with edge features whose shape has only one dimension
    if len(edge_raw_features.shape) == 1:
        edge_raw_features = edge_raw_features[:, np.newaxis]
    # currently, we do not consider edge weights
    # edge_weights = data['w'].astype(np.float64)

    num_edges = edge_raw_features.shape[0]
    assert num_edges == data_num_edges_map[dataset_name], 'Number of edges are not matched!'

    # union to get node set
    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    assert num_nodes == data_num_nodes_map[dataset_name], 'Number of nodes are not matched!'

    assert src_node_ids.min() == 0 or dst_node_ids.min() == 0, "Node index should start from 0!"
    assert edge_ids.min() == 0 or edge_ids.min() == 1, "Edge index should start from 0 or 1!"
    # we notice that the edge id on the datasets (except for tgbl-wiki) starts from 1, so we manually minus the edge ids by 1
    if edge_ids.min() == 1:
        print(f"Manually minus the edge indices by 1 on {dataset_name}")
        edge_ids = edge_ids - 1
    assert edge_ids.min() == 0, "After correction, edge index should start from 0!"

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    eval_neg_edge_sampler = dataset.negative_sampler
    dataset.load_val_ns()
    dataset.load_test_ns()
    eval_metric_name = dataset.eval_metric

    # note that in our data preprocess pipeline, we add an extra node and edge with index 0 as the padded node/edge for convenience of model computation,
    # therefore, for TGB, we also manually add the extra node and edge with index 0
    src_node_ids = src_node_ids + 1
    dst_node_ids = dst_node_ids + 1
    edge_ids = edge_ids + 1

    MAX_FEAT_DIM = 172
    if 'node_feat' not in data.keys():
        node_raw_features = np.zeros((num_nodes + 1, 1))
    else:
        node_raw_features = data['node_feat'].astype(np.float64)
        # deal with node features whose shape has only one dimension
        if len(node_raw_features.shape) == 1:
            node_raw_features = node_raw_features[:, np.newaxis]

    # add feature of padded node and padded edge
    node_raw_features = np.vstack([np.zeros(node_raw_features.shape[1])[np.newaxis, :], node_raw_features])
    edge_raw_features = np.vstack([np.zeros(edge_raw_features.shape[1])[np.newaxis, :], edge_raw_features])

    assert MAX_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'
    assert MAX_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask], edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name


def get_node_classification_tgb_data(dataset_name: str):
    """
    generate tgb data for node classification task
    :param dataset_name: str, dataset name
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object), eval_metric_name, num_classes
    """
    # Load data and train val test split
    dataset = PyGNodePropPredDataset(name=dataset_name, root="datasets")
    data = dataset.dataset.full_data

    src_node_ids = data['sources'].astype(np.longlong)
    dst_node_ids = data['destinations'].astype(np.longlong)
    node_interact_times = data['timestamps'].astype(np.float64)
    edge_ids = data['edge_idxs'].astype(np.longlong)
    edge_raw_features = data['edge_feat'].astype(np.float64)
    # deal with edge features whose shape has only one dimension
    if len(edge_raw_features.shape) == 1:
        edge_raw_features = edge_raw_features[:, np.newaxis]
    # TODO: the number of edges or nodes are mismatched with the TGB paper
    num_edges = edge_raw_features.shape[0]
    # assert num_edges == data_num_edges_map[dataset_name], 'Number of edges are not matched!'

    # union to get node set
    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    # assert num_nodes == data_num_nodes_map[dataset_name], 'Number of nodes are not matched!'

    assert src_node_ids.min() == 0 or dst_node_ids.min() == 0, "Node index should start from 0!"
    assert edge_ids.min() == 0 or edge_ids.min() == 1, "Edge index should start from 0 or 1!"
    # we notice that the edge id on the datasets (except for tgbl-wiki) starts from 1, so we manually minus the edge ids by 1
    if edge_ids.min() == 1:
        print(f"Manually minus the edge indices by 1 on {dataset_name}")
        edge_ids = edge_ids - 1
    assert edge_ids.min() == 0, "After correction, edge index should start from 0!"

    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    eval_metric_name = dataset.eval_metric
    num_classes = dataset.num_classes

    # in TGB, each node tends to be assigned with a label after a number of interactions, which is different from our setting
    # we add interact_types property in Data to mark each interaction with value "train", "validate", "test" or "just_update"
    # set node label, interact_type, and node label time for each interaction
    labels = np.zeros((num_edges, num_classes))
    interact_types = np.array(["just_update" for _ in range(num_edges)])
    node_label_times = copy.deepcopy(node_interact_times)

    # dictionary, key is interact time, value is a dictionary, whose key is node id, value is node label, which is a ndarray with shape (num_classes, )
    label_dict = dataset.dataset.label_dict

    # dictionary, key is a tuple (label time, node id), value is the node label, which is a ndarray (each element is float64 type) with shape (num_classes, )
    converted_label_dict = {}
    for node_label_time in tqdm(label_dict.keys()):
        for src_node_id in label_dict[node_label_time].keys():
            # the type of each node label is
            converted_label_dict[(node_label_time, src_node_id)] = label_dict[node_label_time][src_node_id]

    if os.path.exists(f"./saved_labeled_node_interaction_indices/{dataset_name}.npy"):
        # use to_list() to get the dictionary
        # dictionary, key is a tuple (interact time, node id), value is the interaction index
        labeled_node_interaction_indices = np.load(f"./saved_labeled_node_interaction_indices/{dataset_name}.npy", allow_pickle=True).tolist()
    else:
        # dictionary, key is a tuple (interact time, node id), value is the interaction index
        # for the first time, we need to compute labeled_node_interaction_indices and store it
        labeled_node_interaction_indices = {}
        for (node_label_time, src_node_id) in tqdm(converted_label_dict.keys(), desc=f"data preprocessing for {dataset_name}"):
            # record the most recent interaction index (i.e., edge id) of node src_node_id
            # ndarray, shape (num_edges, )
            nodes_historical_interactions_mask = (src_node_ids == src_node_id) & (node_interact_times <= node_label_time)
            if len(edge_ids[nodes_historical_interactions_mask]) > 0:
                nodes_most_recent_interaction_idx = edge_ids[nodes_historical_interactions_mask][-1]
                assert nodes_most_recent_interaction_idx == np.where(nodes_historical_interactions_mask)[0][-1], \
                    "Mismatched interaction index with edge id!"
            else:
                nodes_most_recent_interaction_idx = 0
                print("Warning: a labeled node is not matched, use the first interaction to match it")
            # use the index of the most recent interaction of the labeled node (deal with mismatched issue with the exact interaction time in label_dict)
            labeled_node_interaction_indices[(node_label_time, src_node_id)] = nodes_most_recent_interaction_idx
        assert len(converted_label_dict.keys()) == len(labeled_node_interaction_indices.keys()), "Mismatched dictionary keys!"

        os.makedirs(f"./saved_labeled_node_interaction_indices", exist_ok=True)
        np.save(f"./saved_labeled_node_interaction_indices/{dataset_name}.npy", labeled_node_interaction_indices)

    # set labels and interact_types
    min_val_time = node_interact_times[val_mask].min()
    min_test_time = node_interact_times[test_mask].min()
    assert min_val_time > node_interact_times[train_mask].max(), "Train data and validation data are mixed!"
    assert min_test_time > node_interact_times[val_mask].max(), "Validation data and test data are mixed!"

    for (node_label_time, src_node_id) in tqdm(converted_label_dict.keys()):
        interaction_idx = labeled_node_interaction_indices[(node_label_time, src_node_id)]
        labels[interaction_idx] = converted_label_dict[(node_label_time, src_node_id)]
        node_label_times[interaction_idx] = node_label_time
        if min_val_time <= node_label_time < min_test_time:
            interact_types[interaction_idx] = "validate"
        elif node_label_time >= min_test_time:
            interact_types[interaction_idx] = "test"
        else:
            interact_types[interaction_idx] = "train"

    # note that in our data preprocess pipeline, we add an extra node and edge with index 0 as the padded node/edge for convenience of model computation,
    # therefore, for TGB, we also manually add the extra node and edge with index 0
    src_node_ids = src_node_ids + 1
    dst_node_ids = dst_node_ids + 1
    edge_ids = edge_ids + 1

    MAX_FEAT_DIM = 172
    if 'node_feat' not in data.keys():
        node_raw_features = np.zeros((num_nodes + 1, 1))
    else:
        node_raw_features = data['node_feat'].astype(np.float64)
        # deal with node features whose shape has only one dimension
        if len(node_raw_features.shape) == 1:
            node_raw_features = node_raw_features[:, np.newaxis]

    # add feature of padded node and padded edge
    node_raw_features = np.vstack([np.zeros(node_raw_features.shape[1])[np.newaxis, :], node_raw_features])
    edge_raw_features = np.vstack([np.zeros(edge_raw_features.shape[1])[np.newaxis, :], edge_raw_features])

    assert MAX_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'
    assert MAX_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {MAX_FEAT_DIM}!'

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids,
                     labels=labels, interact_types=interact_types, node_label_times=node_label_times)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask], node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], interact_types=interact_types[train_mask], node_label_times=node_label_times[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask], node_interact_times=node_interact_times[val_mask],
                    edge_ids=edge_ids[val_mask], labels=labels[val_mask], interact_types=interact_types[val_mask], node_label_times=node_label_times[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask], node_interact_times=node_interact_times[test_mask],
                     edge_ids=edge_ids[test_mask], labels=labels[test_mask], interact_types=interact_types[test_mask], node_label_times=node_label_times[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_metric_name, num_classes
