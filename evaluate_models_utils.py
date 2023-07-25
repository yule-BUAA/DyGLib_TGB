import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
from collections import defaultdict
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator as LinkPredictionEvaluator
from tgb.nodeproppred.evaluate import Evaluator as NodeClassificationEvaluator

from models.EdgeBank import edge_bank_link_prediction
from models.PersistentForecast import PersistentForecast
from models.MovingAverage import MovingAverage
from utils.utils import set_random_seed
from utils.utils import NeighborSampler
from utils.DataLoader import Data


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, eval_stage: str,
                                   eval_metric_name: str, evaluator: LinkPredictionEvaluator, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler from TGB
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage to generate negative edges, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: LinkPredictionEvaluator, link prediction evaluator
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1, pos_dst=batch_dst_node_ids - 1,
                                                                                pos_timestamp=batch_node_interact_times, split_mode=eval_stage)

            # ndarray, shape (batch_size, num_negative_samples_per_node)
            # we should pay attention to the mappings of node ids, add 1 to convert to the mapped node ids in our implementation
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0)
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (node_1_interact_time, node_1_interact_time, ..., node_2_interact_time, node_2_interact_time, ...)
            repeated_batch_node_interact_times = np.repeat(batch_node_interact_times, repeats=num_negative_samples_per_node, axis=0)

            # follow our previous implementation, we compute for positive and negative edges respectively
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, node_feat_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN']:
                # note that negative nodes do not change the memories while the positive nodes change the memories,
                # we need to first compute the embeddings of negative nodes for memory-based models
                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, node_feat_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      edge_ids=None,
                                                                      edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, node_feat_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                src_node_embeddings, dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size * num_negative_samples_per_node, node_feat_dim)
                neg_src_node_embeddings, neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=repeated_batch_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids.flatten(),
                                                                      node_interact_times=repeated_batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # get positive probabilities, Tensor, shape (batch_size, )
            positive_probabilities = model[1](input_1=src_node_embeddings, input_2=dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()
            # get negative probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            negative_probabilities = model[1](input_1=neg_src_node_embeddings, input_2=neg_dst_node_embeddings).squeeze(dim=-1).sigmoid().cpu().numpy()

            for sample_idx in range(len(batch_src_node_ids)):
                # compute metric
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[sample_idx * num_negative_samples_per_node: (sample_idx + 1) * num_negative_samples_per_node],
                    "eval_metric": [eval_metric_name],
                }
                evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch')

    return evaluate_metrics


def evaluate_model_node_classification(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                       evaluate_data: Data, eval_stage: str, eval_metric_name: str, evaluator: NodeClassificationEvaluator,
                                       loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the node classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_data: Data, data to be evaluated
    :param eval_stage: str, specifies the evaluate stage, can be 'val' or 'test'
    :param eval_metric_name: str, name of the evaluation metric
    :param evaluator: NodeClassificationEvaluator, node classification evaluator
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        # store the results for each timeslot, and finally compute the metric for each timeslot
        # dictionary of list, key is the timeslot, value is a list, where each element is a prediction, np.ndarray with shape (num_classes, )
        evaluate_predicts_per_timeslot_dict = defaultdict(list)
        # dictionary of list, key is the timeslot, value is a list, where each element is a label, np.ndarray with shape (num_classes, )
        evaluate_labels_per_timeslot_dict = defaultdict(list)
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_labels, batch_interact_types, batch_node_label_times = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices], \
                evaluate_data.labels[evaluate_data_indices], evaluate_data.interact_types[evaluate_data_indices], \
                evaluate_data.node_label_times[evaluate_data_indices]

            # split the batch data based on interaction types
            # train_idx = torch.tensor(np.where(batch_interact_types == 'train')[0])
            if eval_stage == 'val':
                eval_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                # other_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
            else:
                assert eval_stage == 'test', f"Wrong setting of eval_stage {eval_stage}!"
                eval_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
                # other_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
            # just_update_idx = torch.tensor(np.where(batch_interact_types == 'just_update')[0])
            # assert len(train_idx) == len(other_idx) == 0 and len(eval_idx) + len(just_update_idx) == len(batch_interact_types), "The data are mixed!"

            # for memory-based models, we should use all the interactions to update memories (including eval_stage and 'just_update'),
            # while other memory-free methods only need to compute on eval_stage
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                # get temporal embedding of source and destination nodes, note that the memories are updated during the forward process
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            else:
                if len(eval_idx) > 0:
                    if model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=num_neighbors)

                    elif model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times,
                                                                              num_neighbors=num_neighbors,
                                                                              time_gap=time_gap)
                    elif model_name in ['DyGFormer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, node_feat_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                              dst_node_ids=batch_dst_node_ids,
                                                                              node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {model_name}!")
                else:
                    batch_src_node_embeddings = None

            if len(eval_idx) > 0:
                # get predicted probabilities, shape (batch_size, num_classes)
                predicts = model[1](x=batch_src_node_embeddings).squeeze(dim=-1)
                labels = torch.from_numpy(batch_labels).float().to(predicts.device)

                loss = loss_func(input=predicts[eval_idx], target=labels[eval_idx])

                evaluate_losses.append(loss.item())
                # append the predictions and labels to evaluate_predicts_per_timeslot_dict and evaluate_labels_per_timeslot_dict
                for idx in eval_idx:
                    evaluate_predicts_per_timeslot_dict[batch_node_label_times[idx]].append(predicts[idx].softmax(dim=0).cpu().detach().numpy())
                    evaluate_labels_per_timeslot_dict[batch_node_label_times[idx]].append(labels[idx].cpu().detach().numpy())

                evaluate_idx_data_loader_tqdm.set_description(f'{eval_stage} for the {batch_idx + 1}-th batch, loss: {loss.item()}')

        # compute the evaluation metric for each timeslot
        for time_slot in tqdm(evaluate_predicts_per_timeslot_dict):
            time_slot_predictions = np.stack(evaluate_predicts_per_timeslot_dict[time_slot], axis=0)
            time_slot_labels = np.stack(evaluate_labels_per_timeslot_dict[time_slot], axis=0)
            # compute metric
            input_dict = {
                "y_true": time_slot_labels,
                "y_pred": time_slot_predictions,
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

    return evaluate_losses, evaluate_metrics


def evaluate_edge_bank_link_prediction(args: argparse.Namespace, train_data: Data, val_data: Data, test_data: Data,
                                       val_idx_data_loader: DataLoader, test_idx_data_loader: DataLoader,
                                       evaluate_neg_edge_sampler: NegativeEdgeSampler, eval_metric_name: str, dataset_name: str,):
    """
    evaluate the EdgeBank model for link prediction
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_data: Data, test data
    :param val_idx_data_loader: DataLoader, validate index data loader
    :param test_idx_data_loader: DataLoader, test index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler from TGB
    :param eval_metric_name: str, name of the evaluation metric
    :param dataset_name: str, dataset name
    :return:
    """
    # generate the train_validation split of the data: needed for constructing the memory for EdgeBank
    train_val_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids], axis=0),
                          dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids], axis=0),
                          node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times], axis=0),
                          edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids], axis=0),
                          labels=np.concatenate([train_data.labels, val_data.labels], axis=0))

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'eval_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # evaluate EdgeBank
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        evaluator = LinkPredictionEvaluator(name=dataset_name)

        val_metrics, test_metrics = [], []
        val_idx_data_loader_tqdm = tqdm(val_idx_data_loader, ncols=120)
        test_idx_data_loader_tqdm = tqdm(test_idx_data_loader, ncols=120)

        for batch_idx, val_data_indices in enumerate(val_idx_data_loader_tqdm):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                val_data.src_node_ids[val_data_indices], val_data.dst_node_ids[val_data_indices], \
                val_data.node_interact_times[val_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1, pos_dst=batch_dst_node_ids - 1,
                                                                                pos_timestamp=batch_node_interact_times, split_mode="val")

            # ndarray, shape (batch_size, num_negative_samples_per_node)
            # we should pay attention to the mappings of node ids, add 1 to convert to the mapped node ids in our implementation
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0)

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (repeated_batch_src_node_ids, batch_neg_dst_node_ids.flatten())

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_data.src_node_ids, val_data.src_node_ids[: val_data_indices[0]]], axis=0),
                                dst_node_ids=np.concatenate([train_data.dst_node_ids, val_data.dst_node_ids[: val_data_indices[0]]], axis=0),
                                node_interact_times=np.concatenate([train_data.node_interact_times, val_data.node_interact_times[: val_data_indices[0]]], axis=0),
                                edge_ids=np.concatenate([train_data.edge_ids, val_data.edge_ids[: val_data_indices[0]]], axis=0),
                                labels=np.concatenate([train_data.labels, val_data.labels[: val_data_indices[0]]], axis=0))

            # perform link prediction for EdgeBank
            # positive_probabilities, Tensor, shape (batch_size, )
            # negative_probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=0.15)

            for sample_idx in range(len(batch_src_node_ids)):
                # compute MRR
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[sample_idx * num_negative_samples_per_node: (sample_idx + 1) * num_negative_samples_per_node],
                    "eval_metric": [eval_metric_name],
                }
                val_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            val_idx_data_loader_tqdm.set_description(f'validate for the {batch_idx + 1}-th batch')

        for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times = \
                test_data.src_node_ids[test_data_indices], test_data.dst_node_ids[test_data_indices], \
                test_data.node_interact_times[test_data_indices]

            # batch_neg_dst_node_ids_list: a list of list, where each internal list contains the ids of negative destination nodes for a positive source node
            # contain batch lists, each list with length num_negative_samples_per_node (20 in the TGB evaluation)
            # we should pay attention to the mappings of node ids, reduce 1 to convert to the original node ids
            batch_neg_dst_node_ids_list = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1, pos_dst=batch_dst_node_ids - 1,
                                                                                pos_timestamp=batch_node_interact_times, split_mode="test")

            # ndarray, shape (batch_size, num_negative_samples_per_node)
            # we should pay attention to the mappings of node ids, add 1 to convert to the mapped node ids in our implementation
            batch_neg_dst_node_ids = np.array(batch_neg_dst_node_ids_list) + 1

            num_negative_samples_per_node = batch_neg_dst_node_ids.shape[1]
            # ndarray, shape (batch_size * num_negative_samples_per_node, ),
            # value -> (src_node_1_id, src_node_1_id, ..., src_node_2_id, src_node_2_id, ...)
            repeated_batch_src_node_ids = np.repeat(batch_src_node_ids, repeats=num_negative_samples_per_node, axis=0)

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (repeated_batch_src_node_ids, batch_neg_dst_node_ids.flatten())

            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([train_val_data.src_node_ids, test_data.src_node_ids[: test_data_indices[0]]], axis=0),
                                dst_node_ids=np.concatenate([train_val_data.dst_node_ids, test_data.dst_node_ids[: test_data_indices[0]]], axis=0),
                                node_interact_times=np.concatenate([train_val_data.node_interact_times, test_data.node_interact_times[: test_data_indices[0]]], axis=0),
                                edge_ids=np.concatenate([train_val_data.edge_ids, test_data.edge_ids[: test_data_indices[0]]], axis=0),
                                labels=np.concatenate([train_val_data.labels, test_data.labels[: test_data_indices[0]]], axis=0))

            # perform link prediction for EdgeBank
            # positive_probabilities, Tensor, shape (batch_size, )
            # negative_probabilities, Tensor, shape (batch_size * num_negative_samples_per_node, )
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=history_data,
                                                                                       positive_edges=positive_edges,
                                                                                       negative_edges=negative_edges,
                                                                                       edge_bank_memory_mode=args.edge_bank_memory_mode,
                                                                                       time_window_mode=args.time_window_mode,
                                                                                       time_window_proportion=0.15)

            for sample_idx in range(len(batch_src_node_ids)):
                # compute MRR
                input_dict = {
                    # use slices instead of index to keep the dimension of y_pred_pos
                    "y_pred_pos": positive_probabilities[sample_idx: sample_idx + 1],
                    "y_pred_neg": negative_probabilities[sample_idx * num_negative_samples_per_node: (sample_idx + 1) * num_negative_samples_per_node],
                    "eval_metric": [eval_metric_name],
                }
                test_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            test_idx_data_loader_tqdm.set_description(f'test for the {batch_idx + 1}-th batch')

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')


def evaluate_parameter_free_node_classification(args: argparse.Namespace, train_data: Data, val_data: Data, test_data: Data,
                                                train_idx_data_loader: DataLoader, val_idx_data_loader: DataLoader,
                                                test_idx_data_loader: DataLoader, eval_metric_name: str, num_classes: int):
    """
    evaluate parameter-free models (PersistentForecast and MovingAverage) on the node classification task
    :param args: argparse.Namespace, configuration
    :param train_data: Data, train data
    :param val_data: Data, validation data
    :param test_data: Data, test data
    :param train_idx_data_loader: DataLoader, train index data loader
    :param val_idx_data_loader: DataLoader, validate index data loader
    :param test_idx_data_loader: DataLoader, test index data loader
    :param eval_metric_name: str, name of the evaluation metric
    :param num_classes: int, number of label classes
    :return:
    """
    train_metric_all_runs, val_metric_all_runs, test_metric_all_runs = [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_result_name = f'eval_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'PersistentForecast':
            model = PersistentForecast(num_classes=num_classes)
        else:
            assert args.model_name == 'MovingAverage', f"Wrong setting of model_name {args.model_name}!"
            model = MovingAverage(num_classes=num_classes, window_size=args.moving_average_window_size)
        logger.info(f'model -> {model}')

        loss_func = nn.CrossEntropyLoss()
        evaluator = NodeClassificationEvaluator(name=args.dataset_name)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        def inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader: DataLoader, evaluate_data: Data, stage: str):
            """
            inner function to evaluate parameter-free models (PersistentForecast and MovingAverage) on the node classification task,
            note that we need compute on the train data because it can modify the memory to improve validation and test performance
            :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
            :param evaluate_data: Data, data to be evaluated
            :param stage: str, specifies the stage, can be 'train', 'val' or 'test'
            :return:
            """
            # store evaluate losses and metrics
            evaluate_losses, evaluate_metrics = [], []
            # store the results for each timeslot, and finally compute the metric for each timeslot
            # dictionary of list, key is the timeslot, value is a list, where each element is a prediction, np.ndarray with shape (num_classes, )
            evaluate_predicts_per_timeslot_dict = defaultdict(list)
            # dictionary of list, key is the timeslot, value is a list, where each element is a label, np.ndarray with shape (num_classes, )
            evaluate_labels_per_timeslot_dict = defaultdict(list)
            evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
            for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
                batch_src_node_ids, batch_labels, batch_interact_types, batch_node_label_times = \
                    evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.labels[evaluate_data_indices], \
                    evaluate_data.interact_types[evaluate_data_indices], evaluate_data.node_label_times[evaluate_data_indices]

                # split the batch data based on interaction types
                if stage == 'train':
                    eval_idx = torch.tensor(np.where(batch_interact_types == 'train')[0])
                    # other_idx_1 = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                    # other_idx_2 = torch.tensor(np.where(batch_interact_types == 'test')[0])
                elif stage == 'val':
                    eval_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                    # other_idx_1 = torch.tensor(np.where(batch_interact_types == 'train')[0])
                    # other_idx_2 = torch.tensor(np.where(batch_interact_types == 'test')[0])
                else:
                    assert stage == 'test', f"Wrong setting of stage {stage}!"
                    eval_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
                    # other_idx_1 = torch.tensor(np.where(batch_interact_types == 'train')[0])
                    # other_idx_2 = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                # just_update_idx = torch.tensor(np.where(batch_interact_types == 'just_update')[0])
                # assert len(other_idx_1) == len(other_idx_2) == 0 and len(eval_idx) + len(just_update_idx) == len(batch_interact_types), "The data are mixed!"

                # only use the interactions in stage to update memories, since the labels of 'just_update' are meaningless
                if len(eval_idx) > 0:
                    predicts, label_times, labels = [], [], []
                    for idx in eval_idx:
                        predict = model.get_memory(node_id=batch_src_node_ids[idx])
                        predicts.append(predict)
                        label_times.append(batch_node_label_times[idx])
                        labels.append(batch_labels[idx])
                        model.update_memory(node_id=batch_src_node_ids[idx], node_label=batch_labels[idx])
                    predicts = torch.from_numpy(np.stack(predicts, axis=0)).float()
                    labels = torch.from_numpy(np.stack(labels, axis=0)).float()

                    loss = loss_func(input=predicts, target=labels)

                    evaluate_losses.append(loss.item())
                    # append the predictions and labels to evaluate_predicts_per_timeslot_dict and evaluate_labels_per_timeslot_dict
                    for predict, label_time, label in zip(predicts, label_times, labels):
                        evaluate_predicts_per_timeslot_dict[label_time].append(predict.numpy())
                        evaluate_labels_per_timeslot_dict[label_time].append(label.numpy())

                    evaluate_idx_data_loader_tqdm.set_description(f'{stage} for the {batch_idx + 1}-th batch, loss: {loss.item()}')

            # compute the evaluation metric for each timeslot
            for time_slot in tqdm(evaluate_predicts_per_timeslot_dict):
                time_slot_predictions = np.stack(evaluate_predicts_per_timeslot_dict[time_slot], axis=0)
                time_slot_labels = np.stack(evaluate_labels_per_timeslot_dict[time_slot], axis=0)
                # compute metric
                input_dict = {
                    "y_true": time_slot_labels,
                    "y_pred": time_slot_predictions,
                    "eval_metric": [eval_metric_name],
                }
                evaluate_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            return evaluate_losses, evaluate_metrics

        train_losses, train_metrics = inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader=train_idx_data_loader,
                                                                                        evaluate_data=train_data,
                                                                                        stage='train')

        val_losses, val_metrics = inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader=val_idx_data_loader,
                                                                                    evaluate_data=val_data,
                                                                                    stage='val')

        test_losses, test_metrics = inner_evaluate_parameter_free_node_classification(evaluate_idx_data_loader=test_idx_data_loader,
                                                                                      evaluate_data=test_data,
                                                                                      stage='test')

        # store the evaluation metrics at the current run
        train_metric_dict, val_metric_dict, test_metric_dict = {}, {}, {}

        logger.info(f'train loss: {np.mean(train_losses):.4f}')
        for metric_name in train_metrics[0].keys():
            average_train_metric = np.mean([train_metric[metric_name] for train_metric in train_metrics])
            logger.info(f'train {metric_name}, {average_train_metric:.4f}')
            train_metric_dict[metric_name] = average_train_metric

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        train_metric_all_runs.append(train_metric_dict)
        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "train metrics": {metric_name: f'{train_metric_dict[metric_name]:.4f}' for metric_name in train_metric_dict},
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
        with open(save_result_path, 'w') as file:
            file.write(result_json)
        logger.info(f'save results at {save_result_path}')

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in train_metric_all_runs[0].keys():
        logger.info(f'train {metric_name}, {[train_metric_single_run[metric_name] for train_metric_single_run in train_metric_all_runs]}')
        logger.info(f'average train {metric_name}, {np.mean([train_metric_single_run[metric_name] for train_metric_single_run in train_metric_all_runs]):.4f} '
                    f'± {np.std([train_metric_single_run[metric_name] for train_metric_single_run in train_metric_all_runs], ddof=1):.4f}')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
