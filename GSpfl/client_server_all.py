import warnings

warnings.filterwarnings('ignore')

import os
import pickle
import torch.nn.functional as F
from sklearn.metrics import f1_score
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
import itertools as it
from flwr.common import Metrics
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Optional, Union
from torch_geometric.nn import GCNConv, GATv2Conv
import copy

# flower imports for custom strategy
from typing import Callable, Dict, List, Optional, Tuple, Union
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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg, aggregate_median, aggregate_krum, \
    aggregate_trimmed_avg, aggregate_bulyan

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 113
BATCH_SIZE = 32
time_frequency = 60 * 24
chunk_size = 10


def datetime_range(start, end, delta):
    current = start
    while current <= end:
        yield current
        current += delta


def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])


def choose_target_generate_fllist(sheroaks_crime):
    crime_type = 8
    start_time_so = '2018-01-01'
    end_time_so = '2018-12-31'
    format_string = '%Y-%m-%d'
    start_time_so = datetime.strptime(start_time_so, format_string)
    end_time_so = datetime.strptime(end_time_so, format_string)
    time_list_so = [dt.strftime('%Y-%m-%d') for dt in
                    datetime_range(start_time_so, end_time_so, timedelta(minutes=time_frequency))]
    x_ = list(moving_window(time_list_so, chunk_size))
    final_list_so = []
    label_list_so = []
    for i in range(len(x_)):
        feature_time_frame = x_[i][:chunk_size - 1]
        feature_list = []
        for index_fea in range(len(feature_time_frame) - 1):
            start_so = feature_time_frame[index_fea]
            end_so = feature_time_frame[index_fea + 1]
            df_so_middle = sheroaks_crime.loc[
                (sheroaks_crime['date_occ'] >= start_so) & (sheroaks_crime['date_occ'] < end_so)]
            crime_record = np.zeros(crime_type)
            for index, row in df_so_middle.iterrows():
                crime_record[int(row["crime_type_id"])] = 1
            feature_list.append(crime_record)
        final_list_so.append(feature_list)

        label_time_frame = x_[i][chunk_size - 2:]
        label_time_slots = sheroaks_crime.loc[
            (sheroaks_crime['date_occ'] >= label_time_frame[0]) & (sheroaks_crime['date_occ'] < label_time_frame[1])]
        crime_record = np.zeros(crime_type)
        for index_label, row_label in label_time_slots.iterrows():
            crime_record[int(row_label["crime_type_id"])] = 1
        label_list_so.append(crime_record)

    # print("the shape of feature list is {}, and the shape of label list is {} ".format(np.shape(final_list_so), np.shape(label_list_so)))
    return final_list_so, label_list_so


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label


def load_datasets():
    PATH = 'C:\\Users\\cbhum\\OneDrive\\Documents\\Datasets\\crime\\'
    df = pd.read_csv(PATH + 'processed_crime.csv')

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    testloaders = []
    globaltestloaders = []
    x_global_test = []
    target_global_test = []
    global NUM_CLIENTS

    # Group the DataFrame by 'neighborhood_id' and calculate the size of each group
    group_partition = df.groupby('neighborhood_id')
    NUM_CLIENTS = len(group_partition)

    for partition_id, partition_df in group_partition:
            feature, label = choose_target_generate_fllist(partition_df)
            num_samples = len(feature)
            # Calculate the sizes for training, validation, test, and global test sets
            num_train = round(num_samples * 0.5417)  # 65% for training
            num_val = round(num_samples * 0.0417)  # 5% for validation
            num_test = round(num_samples * 0.0833)  # 10% for testing
            # num_global_test = num_samples - num_train - num_val - num_test  # Remaining for global testing

            # Training set
            x_train, target_train = feature[:num_train], label[:num_train]
            # Validation set
            x_val, target_val = feature[num_train:num_train + num_val], label[num_train:num_train + num_val]
            # Test set
            x_test, target_test = feature[num_train + num_val: num_train + num_val + num_test], label[
                                                                                                num_train + num_val: num_train + num_val + num_test]
            # Global test
            x_global_test.extend(x_test)
            target_global_test.extend(target_test)

            train_dataset = CustomDataset(x_train, target_train)
            val_dataset = CustomDataset(x_val, target_val)
            test_dataset = CustomDataset(x_test, target_test)

            trainloaders.append(DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False))
            valloaders.append(DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False))
            testloaders.append(DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False))

    global_test_dataset = CustomDataset(x_global_test, target_global_test)
    globaltestloaders.append(DataLoader(global_test_dataset, batch_size=BATCH_SIZE, shuffle=False))

    return trainloaders, valloaders, testloaders, globaltestloaders


trainloaders, valloaders, testloaders, globaltestloaders = load_datasets()


class Net(nn.Module):
    def __init__(self, num_inputs=8, hidden_units=16):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=8)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.relu(hn[0])
        out = self.linear(out)
        out = self.sigmoid(out)
        return out


def sd_matrixing_agcn(weights, cid):
    concat_weights = []

    for i, (client_params, num_examples) in enumerate(weights):
        temp_weights = []
        for param_idx, param in enumerate(client_params):
            reshaped_layer = param.reshape(-1)
            torch_reshaped_layer = torch.from_numpy(reshaped_layer)
            temp_weights.append(torch_reshaped_layer)
        client_row = torch.cat(temp_weights, dim=0)
        concat_weights.append(client_row)

    feature_matrix = torch.stack(concat_weights, dim=0)
    client_matrix = feature_matrix[int(cid), :]
    return client_matrix


def read_weights_agcn(cid):
    file_path = os.path.join("../GSpfl/store_weights", f"weights_after_gcn.pkl")
    with open(file_path, "rb") as f:
        loaded_weights = pickle.load(f)
        concat_weights = sd_matrixing_agcn(loaded_weights, cid)
        return concat_weights


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs, targets):
        # Flatten tensors to 2D (batch_size x num_classes)
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        # Calculate Tversky numerator and denominator
        true_positives = torch.sum(outputs * targets, dim=1)
        false_positives = torch.sum(outputs * (1 - targets), dim=1)
        false_negatives = torch.sum((1 - outputs) * targets, dim=1)
        tversky_num = true_positives + self.smooth
        tversky_denom = true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth
        # Calculate Tversky loss
        tversky_loss = 1.0 - (tversky_num / tversky_denom)
        # Apply focal loss
        focal_loss = torch.pow(tversky_loss, self.gamma)
        # Average the focal loss over all samples
        focal_tversky_loss = focal_loss.mean()

        return focal_tversky_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


def train(net, personalize_param, global_param, trainloader, learning_rate, server_round, cid, epochs: int):
    """Train the network on the training set."""
    all_predictions, all_labels = [], []
    # criterion = torch.nn.BCELoss()
    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    # criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1, smooth=1e-6)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(criterion.parameters()), lr=learning_rate,
                                 weight_decay=1e-4)
    correct_train, total_train, epoch_loss_train = 0, 0, 0.0
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            x, labels = batch[0], batch[1]
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, labels)
            # reg terms
            if server_round != 1:
                reg1, reg2 = 0.0, 0.0
                for param, p_param, g_param in zip(net.parameters(), personalize_param, global_param):
                    reg1 += torch.nn.functional.pairwise_distance(param.view(-1), p_param.view(-1)).sum()
                    reg2 += torch.nn.functional.pairwise_distance(param.view(-1), g_param.view(-1)).sum()
                loss = loss + reg1 + reg2
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
            total_train += labels.size(0)
            predicted_labels = (outputs > 0.5).float()
            all_predictions.extend(predicted_labels.tolist())
            all_labels.extend(labels.tolist())
            # Compute accuracy for each sample and each label
            correct_train += ((predicted_labels == labels).float().sum(dim=1) == labels.size(1)).sum().item()

        epoch_loss_train /= len(trainloader.dataset)
        epoch_acc = correct_train / total_train
        epoch_loss_train /= len(trainloader)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        micro_f1 = f1_score(all_labels, all_predictions, average='micro')
        print(
            f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}, macro F1 {macro_f1}, micro F1 {micro_f1}")
        return epoch_loss_train, micro_f1, macro_f1


def val(net, testloader):
    """Evaluate the network on the entire test set."""
    all_predictions, all_labels, macro, micro = [], [], [], []
    # criterion = torch.nn.BCELoss()
    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    # criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1, smooth=1e-6)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0], batch[1]
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted_labels = (outputs > 0.5).float()
            all_predictions.extend(predicted_labels.tolist())
            all_labels.extend(labels.tolist())
            # Compute accuracy for each sample and each label
            correct += ((predicted_labels == labels).float().sum(dim=1) == labels.size(1)).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    return loss, accuracy, micro_f1, macro_f1


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    all_predictions, all_labels, macro, micro = [], [], [], []
    # criterion = torch.nn.BCELoss()
    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
    # criterion = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1, smooth=1e-6)
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            features, labels = batch[0], batch[1]
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted_labels = (outputs > 0.5).float()
            all_predictions.extend(predicted_labels.tolist())
            all_labels.extend(labels.tolist())
            correct += ((predicted_labels == labels).float().sum(dim=1) == labels.size(1)).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    # class_report = classification_report(all_labels, all_predictions)
    return loss, accuracy, micro_f1, macro_f1


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, testloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        learning_r = config["lr"]
        server_round = config["round"]
        global_param = [copy.deepcopy(param) for param in self.net.parameters()]
        if server_round != 1:
            personalize_param = read_weights_agcn(self.cid)
        else:
            personalize_param = None
        train_loss, train_micro, train_macro = train(self.net, personalize_param, global_param, self.trainloader,
                                                     learning_r, server_round, self.cid, epochs=5)
        return get_parameters(self.net), len(self.trainloader), {"loss": float(train_loss), "micro": float(train_micro),
                                                                 "macro": float(train_macro)}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy, micro, macro = test(self.net, self.testloader)
        print(f" Evaluate Test loss {loss}, Test accuracy {accuracy}, micro F1 {micro}, macro F1 {macro}")
        return float(loss), len(self.testloader), {"micro": float(micro), "macro": float(macro)}


def numpyclient_fn(cid) -> FlowerNumPyClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    testloader = testloaders[int(cid)]
    return FlowerNumPyClient(cid, net, trainloader, valloader, testloader)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    micros = [num_examples * m["micro"] for num_examples, m in metrics]
    macros = [num_examples * m["macro"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"micro": sum(micros) / sum(examples), "macro": sum(macros) / sum(examples)}


def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: dict[str, fl.common.Scalar], ):
    net = Net().to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    losses, accuracys, micros, macros = [], [], [], []
    for batch_idx, globaltestloader in enumerate(globaltestloaders):
        loss, accuracy, micro, macro = test(net, globaltestloader)
        losses.append(loss)
        accuracys.append(accuracy)
        micros.append(micro)
        macros.append(macro)

    loss = sum(losses) / len(losses)
    accuracy = sum(accuracys) / len(accuracys)
    micro = sum(micros) / len(micros)
    macro = sum(macros) / len(macros)

    print(f"Server-side evaluation loss {loss}, accuracy {accuracy}, micro {micro}, macro {macro}")
    return loss, {"micro": micro, "macro": macro}


def gram_matrix(X):
    """
    Compute the Gram matrix for input matrix X.
    """
    return X @ X.T


def center_gram_matrix(K):
    """
    Center the Gram matrix K.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def cka_similarity(X, Y):
    """
    Compute the CKA similarity between two matrices X and Y.
    """
    # Compute Gram matrices
    K = gram_matrix(X)
    L = gram_matrix(Y)
    # Center Gram matrices
    K_centered = center_gram_matrix(K)
    L_centered = center_gram_matrix(L)
    # Compute the numerator and denominators for CKA
    numerator = np.trace(K_centered @ L_centered)
    denominator = np.sqrt(np.trace(K_centered @ K_centered) * np.trace(L_centered @ L_centered))

    return numerator / denominator


def create_graph_from_CKA_sim(self, weights):
    count = 0
    for layers in zip(*weights):
        layer_np = np.asarray(layers)
        new_shape = (layer_np.shape[0], -1)
        reshaped_layer = layer_np.reshape(new_shape)
        num_clients = len(reshaped_layer)
        similarity_matrix = np.zeros((num_clients, num_clients))
        if count in [0, 1, 4]:
            # print(" I am count ", count)
            for i in range(num_clients):
                for j in range(num_clients):
                    # similarity_matrix[i, j] = euclidean(reshaped_layer[i], reshaped_layer[j])    # Euclidean Distance
                    # similarity_matrix[i, j] = np.linalg.norm(reshaped_layer[i] - reshaped_layer[j])   # Frobenius Norm
                    # similarity_matrix[i, j] = wasserstein_distance(reshaped_layer[i], reshaped_layer[j])  # Euclidean Distance
                    # similarity_matrix[i, j] = euclidean(reshaped_layer[i], reshaped_layer[j])  # Euclidean Distance
                    similarity_matrix[i, j] = cka_similarity(layer_np[i], layer_np[j])

            min_val = np.min(similarity_matrix)
            max_val = np.max(similarity_matrix)

            if max_val != min_val:  # Avoid division by zero
                normalized_similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)
            else:
                normalized_similarity_matrix = np.zeros_like(similarity_matrix)
        count += 1
    return normalized_similarity_matrix


def create_adj_from_graph_CKA_sim(self, graph, epsilon=0.50, sigma=0.1):
    n = len(graph)
    adj = np.zeros((n, n))
    for ci in range(n):
        adj[ci][ci] = 0
        for cj in range(ci + 1, n):
            if graph[ci][cj] > epsilon:
                adj[ci][cj] = adj[cj][ci] = graph[ci][cj]
            else:
                adj[ci][cj] = adj[cj][ci] = 0
    return adj


class FedGCNStrategy(fl.server.strategy.Strategy):
    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 20,
            min_evaluate_clients: int = 20,
            min_available_clients: int = NUM_CLIENTS,
            evaluate_fn=evaluate,  # Pass the evaluation function
            evaluate_metrics_aggregation_fn=weighted_average,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        return "FedGCNStrategy"

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Create custom configs
        config = {"lr": 0.01, "round": server_round}
        fit_configurations = []

        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        client_weights = [weights for weights, _ in weights_results]

        # flatten client weights
        concat_weights = []
        for layers in zip(*client_weights):
            layer_np = np.asarray(layers)
            new_shape = (layer_np.shape[0], -1)
            reshaped_layer = layer_np.reshape(new_shape)
            torch_reshaped_layer = torch.from_numpy(reshaped_layer)
            concat_weights.append(torch_reshaped_layer)

        feature_matrix = torch.cat(concat_weights, dim=1)

        # Construct Adjacency Matrix
        graph_matrix = create_graph_from_CKA_sim(self, client_weights)
        adj_matrix = create_adj_from_graph_CKA_sim(self, graph_matrix)
        adj_matrix_u = np.triu(adj_matrix)  # Upper triangular part
        np.fill_diagonal(adj_matrix_u, 0)

        # Create adjacency matrix as edge indices for PyTorch Geometric
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

        class GCN(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.conv1 = GCNConv(input_dim, output_dim)
                self.relu = nn.RReLU(lower=0.1, upper=0.3)  # works little good

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = self.relu(x)
                return x

        gcn = GCN(input_dim=feature_matrix.shape[1], output_dim=feature_matrix.shape[1])
        updated_features = gcn(feature_matrix, edge_index)

        # construct model from flatten weights (We can save the keys and values if run without flower)
        original_layer_shapes = [(64, 8), (64, 16), (64,), (64,), (8, 16), (8,)]
        # Calculate the flattened sizes of each layer
        layer_sizes = [torch.tensor(shape).prod().item() for shape in original_layer_shapes]
        # Split the resultant matrix into individual layers
        split_layers = torch.split(updated_features, layer_sizes, dim=1)
        # Reshape each split layer to its original shape
        reconstructed_layers = [
            layer.view(113, *shape) for layer, shape in zip(split_layers, original_layer_shapes)
        ]

        num_examples_list = [fit_res.num_examples for _, fit_res in results]
        new_weights_results = []
        for client_idx in range(113):  # Iterate over clients
            client_layers = [layer[client_idx] for layer in reconstructed_layers]  # Extract client-specific layers
            client_params = [tensor.detach().numpy() for tensor in client_layers]
            new_weights_results.append((client_params, num_examples_list[client_idx]))

        #  save personalize weights
        os.makedirs("../GSpfl/store_weights", exist_ok=True)
        file_path = os.path.join("../GSpfl/store_weights", f"weights_after_gcn.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(new_weights_results, f)

        # num_malicious = 2
        # # For each client, take the n-f-2 closest parameters vectors
        # num_closest = max(1, len(adj_matrix) - num_malicious - 2)
        # closest_indices = []
        # for distance in adj_matrix:
        #     closest_indices.append(
        #         np.argsort(distance)[1: num_closest + 1].tolist()  # noqa: E203
        #     )
        #
        #     # Compute the score for each client, that is the sum of the distances
        #     # of the n-f-2 closest parameters vectors
        # scores = [
        #     np.sum(adj_matrix[i, closest_indices[i]])
        #     for i in range(len(adj_matrix))
        # ]
        # to_keep = len(adj_matrix) - num_malicious
        # best_indices = np.argsort(scores)[::-1][len(scores) - to_keep:]  # noqa: E203
        # best_results = [new_weights_results[i] for i in best_indices]

        # Iterate over results
        agg_macro, agg_micro = [], []
        for client, eval_res in results:
            metrics = eval_res.metrics
            agg_macro.append(metrics['macro'])
            agg_micro.append(metrics['micro'])

        macro = sum(agg_macro) / len(agg_macro)
        micro = sum(agg_micro) / len(agg_micro)
        metrics_aggregated = {'macro': macro, 'micro': micro}

        # convert arrays to flower return parameters type
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        agg_macro, agg_micro = [], []
        for client, eval_res in results:
            metrics = eval_res.metrics
            agg_macro.append(metrics['macro'])
            agg_micro.append(metrics['micro'])

        macro = sum(agg_macro) / len(agg_macro)
        micro = sum(agg_micro) / len(agg_micro)

        metrics_aggregated = {'macro': macro, 'micro': micro}

        return loss_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


client_resources = {"num_cpus": 1, "num_gpus": 0.0}

# Start simulation
history = fl.simulation.start_simulation(
    client_fn=numpyclient_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=FedGCNStrategy(),
    client_resources=client_resources,
)

# Save object to file
with open('../data.pickle', 'wb') as f:
    pickle.dump(history, f)

