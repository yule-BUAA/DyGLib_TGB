import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset


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


for dataset_name in ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-comment", "tgbl-flight"]:
    dataset = LinkPropPredDataset(name=dataset_name, root="datasets")
    data = dataset.full_data

    src_node_ids = data['sources'].astype(np.longlong)
    dst_node_ids = data['destinations'].astype(np.longlong)
    node_interact_times = data['timestamps'].astype(np.float64)
    edge_ids = data['edge_idxs'].astype(np.longlong)
    labels = data['edge_label']
    edge_raw_features = data['edge_feat'].astype(np.float64)

    print("==========================")
    print(f"statistics on {dataset_name}:")
    print(f"minimal edge index: {edge_ids.min()}")
    print(f"maximal edge index: {edge_ids.max()}")
    print(f"minimal node index: {min(src_node_ids.min(), dst_node_ids.min())}")
    print(f"maximal node index: {max(src_node_ids.max(), dst_node_ids.max())}")

    num_edges = edge_raw_features.shape[0]
    print(f"actual number of edges: {num_edges}", )
    print(f"reported number of edges:  {data_num_edges_map[dataset_name]}")
    assert num_edges == data_num_edges_map[dataset_name], 'Number of edges are not matched!'
    # union to get node set
    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    print(f"actual number of nodes: {num_nodes}")
    print(f"reported number of nodes: {data_num_nodes_map[dataset_name]}")
    assert num_nodes == data_num_nodes_map[dataset_name], 'Number of nodes are not matched!'

    print(f"shape of edge features: {edge_raw_features.shape}")
    if 'node_feat' in data.keys():
        node_raw_features = data['node_feat'].astype(np.float64)
        print(f"shape of node features: {edge_raw_features.shape}")

for dataset_name in ["tgbn-trade", "tgbn-genre", "tgbn-reddit"]:
    dataset = PyGNodePropPredDataset(name=dataset_name, root="datasets")
    data = dataset.dataset.full_data

    src_node_ids = data['sources'].astype(np.longlong)
    dst_node_ids = data['destinations'].astype(np.longlong)
    node_interact_times = data['timestamps'].astype(np.float64)
    edge_ids = data['edge_idxs'].astype(np.longlong)
    labels = data['edge_label']
    edge_raw_features = data['edge_feat'].astype(np.float64)

    print("==========================")
    print(f"statistics on {dataset_name}:")
    print(f"minimal edge index: {edge_ids.min()}")
    print(f"maximal edge index: {edge_ids.max()}")
    print(f"minimal node index: {min(src_node_ids.min(), dst_node_ids.min())}")
    print(f"maximal node index: {max(src_node_ids.max(), dst_node_ids.max())}")

    num_edges = edge_raw_features.shape[0]
    print(f"actual number of edges: {num_edges}", )
    print(f"reported number of edges:  {data_num_edges_map[dataset_name]}")
    # assert num_edges == data_num_edges_map[dataset_name], 'Number of edges are not matched!'
    # union to get node set
    num_nodes = len(set(src_node_ids) | set(dst_node_ids))
    print(f"actual number of nodes: {num_nodes}")
    print(f"reported number of nodes: {data_num_nodes_map[dataset_name]}")
    # assert num_nodes == data_num_nodes_map[dataset_name], 'Number of nodes are not matched!'

    print(f"shape of edge features: {edge_raw_features.shape}")
    if 'node_feat' in data.keys():
        node_raw_features = data['node_feat'].astype(np.float64)
        print(f"shape of node features: {edge_raw_features.shape}")

# ==========================
# statistics on tgbl-wiki:
# minimal edge index: 0
# maximal edge index: 157473
# minimal node index: 0
# maximal node index: 9226
# actual number of edges: 157474
# reported number of edges:  157474
# actual number of nodes: 9227
# reported number of nodes: 9227
# shape of edge features: (157474, 172)
# ==========================
# statistics on tgbl-review:
# minimal edge index: 1
# maximal edge index: 4873540
# minimal node index: 0
# maximal node index: 352636
# actual number of edges: 4873540
# reported number of edges:  4873540
# actual number of nodes: 352637
# reported number of nodes: 352637
# shape of edge features: (4873540, 1)
# ==========================
# statistics on tgbl-coin:
# minimal edge index: 1
# maximal edge index: 22809486
# minimal node index: 0
# maximal node index: 638485
# actual number of edges: 22809486
# reported number of edges:  22809486
# actual number of nodes: 638486
# reported number of nodes: 638486
# shape of edge features: (22809486, 1)
# ==========================
# statistics on tgbl-comment:
# minimal edge index: 1
# maximal edge index: 44314507
# minimal node index: 0
# maximal node index: 994789
# actual number of edges: 44314507
# reported number of edges:  44314507
# actual number of nodes: 994790
# reported number of nodes: 994790
# shape of edge features: (44314507, 2)
# ==========================
# statistics on tgbl-flight:
# minimal edge index: 1
# maximal edge index: 67169570
# minimal node index: 0
# maximal node index: 18142
# actual number of edges: 67169570
# reported number of edges:  67169570
# actual number of nodes: 18143
# reported number of nodes: 18143
# shape of edge features: (67169570, 16)
# ==========================
# statistics on tgbn-trade:
# minimal edge index: 1
# maximal edge index: 468245
# minimal node index: 0
# maximal node index: 254
# actual number of edges: 468245
# reported number of edges:  507497
# actual number of nodes: 255
# reported number of nodes: 255
# shape of edge features: (468245,)
# ==========================
# statistics on tgbn-genre:
# minimal edge index: 1
# maximal edge index: 17858395
# minimal node index: 0
# maximal node index: 1504
# actual number of edges: 17858395
# reported number of edges:  17858395
# actual number of nodes: 1505
# reported number of nodes: 992
# shape of edge features: (17858395,)
# ==========================
# statistics on tgbn-reddit:
# minimal edge index: 1
# maximal edge index: 27174118
# minimal node index: 0
# maximal node index: 11765
# actual number of edges: 27174118
# reported number of edges:  27174118
# actual number of nodes: 11766
# reported number of nodes: 11068
# shape of edge features: (27174118,)
