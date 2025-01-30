import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import grpc
import json
import datetime
import csv
import os
import time
import uuid
import io
import argparse
import random

import fl_proto_pb2 as fl_pb2
import fl_proto_pb2_grpc as fl_pb2_grpc

from fashion_mnist_model import FashionMNISTCNN
# Hyperparameters
local_lr = 0.01
lr_decay = 0.997
local_epochs = 3
target_accuracy = 80.0
# momentum = 0.9
number_of_rounds = 300
mu = 0.8
base_data_source = 'FASHION-MNIST'
session_id = str(uuid.uuid4())

# Set random seeds
def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# MNIST Normalization
def base_transform():
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion MNIST stats
    ]

# Augmentations for different scenarios
def fs_low():
    return [
        transforms.RandomRotation(5)
    ]

def fs_medium():
    return [
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

def fs_high():
    return [
        transforms.RandomRotation(15),  # Reduced from 30°
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Mild translation
        # transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Simulate writing variations
        transforms.RandomAutocontrast(p=0.5)

    ]

# Load full MNIST dataset
def load_full_fmnist(train=True):
    return datasets.FashionMNIST(root='./data_fmnist', train=train, download=True)

# Dirichlet Distribution
def dirichlet_distribution(num_clients, num_classes, alpha):
    return np.random.dirichlet([alpha] * num_classes, num_clients)

# Create Label Skew Indices
def create_label_skew_indices(client_id, total_samples=60000, alpha_label=1, alpha_quantity=2.5, seed=42):
    client_num = int(client_id.split("client")[-1])
    np.random.seed(seed + client_num)
    random.seed(seed + client_num)
    
    dataset = load_full_fmnist(train=True)
    all_targets = np.array(dataset.targets)
    num_classes = 10
    
    # Explicit 5-client setup
    total_clients = 5
    quantity_proportions = np.random.dirichlet([alpha_quantity] * total_clients)
    client_total = int(quantity_proportions[client_num - 1] * total_samples)
    
    # Class distribution
    class_proportions = np.random.dirichlet([alpha_label] * num_classes)
    class_allocations = (class_proportions * client_total).astype(int)
    class_allocations = np.maximum(class_allocations, 1)
    
    # Redistribution logic
    current_total = np.sum(class_allocations)
    if current_total != client_total:
        diff = client_total - current_total
        class_allocations[-1] += diff
        if class_allocations[-1] < 1:
            deficit = 1 - class_allocations[-1]
            class_allocations[-1] = 1
            largest_class = np.argmax(class_allocations[:-1])
            class_allocations[largest_class] -= deficit

    # Collect indices
    indices = []
    for class_idx in range(num_classes):
        class_indices = np.where(all_targets == class_idx)[0]
        np.random.shuffle(class_indices)
        needed = min(class_allocations[class_idx], len(class_indices))
        indices.extend(class_indices[:needed].tolist())
    
    # Ensure minimum samples
    if len(indices) < 100:  # Example minimum
        needed = 100 - len(indices)
        extra = np.random.choice(np.arange(60000), needed, replace=False)
        indices.extend(extra.tolist())
    
    np.random.shuffle(indices)
    return indices

# Client Dataset
class ClientDataset(torch.utils.data.Dataset):
    def __init__(self, indices, transform):
        self.full_dataset = datasets.FashionMNIST(root='./data_fmnist', train=True, download=True)
        self.indices = indices
        self.transform = transform  
            
    def __getitem__(self, idx):
        img, target = self.full_dataset[self.indices[idx]]
        return self.transform(img), target
            
    def __len__(self):
        return len(self.indices)

# Hybrid DataLoader for Federated Learning
def build_hybrid_loader(client_id, scenario):
    scenario = scenario.upper()
    config = {
        'LOW': {'alpha_label': 1.0, 'alpha_quantity': 5, 'transform': fs_low()},
        'MEDIUM': {'alpha_label': 0.5, 'alpha_quantity': 1, 'transform': fs_medium()},
        'HIGH': {'alpha_label': 0.5, 'alpha_quantity': 1.0, 'transform': fs_high()}
    }[scenario]

    indices = create_label_skew_indices(
        client_id=client_id,
        alpha_label=config['alpha_label'],
        alpha_quantity=config['alpha_quantity']
    )

    transform = transforms.Compose(config['transform'] + base_transform())
    dataset = ClientDataset(indices, transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    test_transform = transforms.Compose(base_transform())
    test_dataset = datasets.FashionMNIST(root='./data_fmnist', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2, shuffle=False)
    
    return train_loader, test_loader

# FedProx Training & Evaluation
def fedprox_train_and_evaluate(model, global_params, device, train_loader, test_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=local_lr)
    global_params = {k: v.to(device).detach().clone() for k, v in global_params.items()}

    model.train()
    total_loss = 0.0
    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            ce_loss = torch.nn.functional.cross_entropy(outputs, target)

            prox_term = 0.0
            for n, p in model.named_parameters():
                prox_term += torch.norm(p - global_params[n], p=2) ** 2
            
            loss = ce_loss + (mu / 2) * prox_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch+1}/{local_epochs} | Loss: {avg_epoch_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / local_epochs
    return accuracy, avg_loss, 0.0
# -----------------------------
# gRPC Communication
# -----------------------------
def request_global_model(stub, client_id, session_id, metadata):
    try:
        response = stub.RequestGlobalModel(fl_pb2.ModelRequest(
            client_id=client_id,
            session_id=session_id,
            metadata=metadata
        ))
        return response.model_data
    except grpc.RpcError as e:
        print(f"[{client_id}] Model request failed: {e.code().name()}")
        return None


def send_model_parameters(stub, model, accuracy, avg_loss, client_id, scenario, order_id, train_loader, round_num):
    # Serialize model
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    
    # Create metadata
    metadata = fl_pb2.ModelMetadata(
        data_source=base_data_source,
        data_quality_score=len(train_loader.dataset)/60000,
        total_parameters=sum(p.numel() for p in model.parameters()),
        timestamp=datetime.datetime.now().isoformat(),
        model_performance={"accuracy": float(accuracy), "average_loss": float(avg_loss)},
        model_structure="FashionMNISTCNN",
        **simulate_network_parameters(),
        current_round=round_num 
    )
    
    # Create message
    message = fl_pb2.ModelParameters(
        client_id=client_id,
        model_data=buffer.getvalue(),
        metadata=metadata,
        session_id=session_id
    )
    
    for attempt in range(5):
        try:
            ack = stub.SendModelParameters(message, timeout=30)
            print(f"Server ACK: {ack.message}")
            return True, ack  # Return both success and Ack
        except grpc.RpcError as e:
            wait_time = 2 ** attempt
            print(f"Attempt {attempt+1} failed: {e.code().name}")
            time.sleep(wait_time)
    print("Failed to send model parameters after 5 attempts.")
    return False, None  # Return failure state

def simulate_network_parameters():
    return {
        'latency': float(np.random.uniform(50, 150)),
        'bandwidth': float(np.random.uniform(1, 10)),
        'reliability': float(np.random.uniform(0.8, 1.0)),
        'cpu_usage': float(np.random.uniform(30, 80)),
        'memory_consumption': float(np.random.uniform(200, 1000))
    }


def save_training_metrics(client_id, scenario, order_id, round_num, accuracy, avg_loss):
    filename = f"training_metrics_fmnist_hybrid_{scenario.lower()}_{client_id}_order_{order_id}.csv"
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Round', 'Accuracy', 'Average Loss', 'Timestamp'])
        writer.writerow([
            round_num,
            f"{accuracy:.2f}%",
            f"{avg_loss:.4f}",
            datetime.datetime.now().isoformat()
        ])


def main():
    parser = argparse.ArgumentParser(description="FedProx MNIST Client")
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True, choices=['LOW', 'MEDIUM', 'HIGH'])
    parser.add_argument("--order_id", type=int, required=True)
    args = parser.parse_args()

    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Client {args.client_id} starting on {device} ===")

    # Data loading
    train_loader, test_loader = build_hybrid_loader(args.client_id, args.scenario)
    
    # Model setup
    model = FashionMNISTCNN().to(device)
    
    # gRPC setup
    channel = grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_send_message_length', 256*1024*1024),
        ('grpc.max_receive_message_length', 256*1024*1024),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.http2.max_pings_without_data', 0),
        ('grpc.keepalive_permit_without_calls', 1)
    ])
    stub = fl_pb2_grpc.FederatedLearningServiceStub(channel)
    
    # Initial metadata
    initial_metadata = fl_pb2.ModelMetadata(
        data_source=base_data_source,
        data_quality_score=len(train_loader.dataset)/60000,
        total_parameters=sum(p.numel() for p in model.parameters()),
        timestamp=datetime.datetime.now().isoformat()
    )

    client_round = 0
    accuracy = 0.0
    while client_round < number_of_rounds:
        # Get current server round with retry logic
        server_round = None
        for attempt in range(5):
            try:
                response = stub.RequestGlobalModel(fl_pb2.ModelRequest(
                    client_id=args.client_id,
                    session_id=session_id,
                    metadata=initial_metadata
                ))
                server_round = response.current_round
                break
            except grpc.RpcError as e:
                if attempt == 4:
                    print("Critical: Failed to connect to server after 5 attempts")
                    return
                time.sleep(2 ** attempt)

        if server_round is None:
            print("Could not get server status. Retrying...")
            time.sleep(5)
            continue

        # Termination check
        if server_round >= number_of_rounds:
            print("Server completed all rounds. Exiting.")
            break

        # Synchronization checks
        if server_round > client_round:
            print(f"Server ahead (Round {server_round}), client syncing!")
            client_round = server_round
            continue

        if server_round < client_round:
            print(f"Client ahead (Client: {client_round}, Server: {server_round}), waiting!")
            time.sleep(5)
            continue

        # Load global model
        if response.model_data:
            try:
                global_state = torch.load(io.BytesIO(response.model_data), map_location=device)
                model.load_state_dict(global_state)
            except Exception as e:
                print(f"Error loading global model: {str(e)}")
                continue

        # Local training
        print(f"\n=== Participating in Round {server_round + 1}/{number_of_rounds} ===")
        initial_global_params = {k: v.clone() for k, v in model.state_dict().items()}
        try:
            accuracy, avg_loss, _ = fedprox_train_and_evaluate(
                model, initial_global_params, device, train_loader, test_loader
            )
        except Exception as e:
            print(f"Training failed: {str(e)}")
            continue

        print(f"\n=== Client {args.client_id} Results ===")
        print(f"Round {server_round} | Accuracy: {accuracy:.2f}% | Loss: {avg_loss:.4f}")

        # Save metrics
        save_training_metrics(
            client_id=args.client_id,
            scenario=args.scenario,
            order_id=args.order_id,
            round_num=server_round,
            accuracy=accuracy,
            avg_loss=avg_loss
        )

        # Send update with verification
        success = False
        for send_attempt in range(5):
            try:
                # Verify server round hasn't changed
                current_response = stub.RequestGlobalModel(fl_pb2.ModelRequest(
                    client_id=args.client_id,
                    session_id=session_id,
                    metadata=initial_metadata
                ))
                if current_response.current_round != server_round:
                    print(f"Server advanced to round {current_response.current_round} during training")
                    break

                # Send parameters
                send_success, ack = send_model_parameters(
                    stub=stub,
                    model=model,
                    accuracy=accuracy,
                    avg_loss=avg_loss,
                    client_id=args.client_id,
                    scenario=args.scenario,
                    order_id=args.order_id,
                    train_loader=train_loader,
                    round_num=server_round
                )

                if send_success and ack:
                    if "aggregation successful" in ack.message.lower():
                        print("Server aggregated successfully")
                        success = True
                        break
                    elif "already received" in ack.message.lower():
                        print("Server already processed this update")
                        success = True
                        break
                    else:
                        print(f"Server status: {ack.message}")
                        time.sleep(2)

            except grpc.RpcError as e:
                print(f"Attempt {send_attempt+1} failed: {e.code().name}")
                time.sleep(2 ** send_attempt)

        if success:
            client_round += 1
            global local_lr
            local_lr *= lr_decay
            print(f"\n[Client {args.client_id}] LR decayed to: {local_lr:.4f}")
        else:
            print("Failed to send update. Re-syncing with server.")

    print(f"\n=== Client {args.client_id} completed all rounds ===")


if __name__ == "__main__":
    main()