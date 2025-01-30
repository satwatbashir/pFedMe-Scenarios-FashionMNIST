# simulate_parameters.py

import random

def simulate_network_parameters():
    """Simulate network parameters and resource availability."""
    latency = random.uniform(50, 200)  # milliseconds
    bandwidth = random.uniform(10, 100)  # Mbps
    reliability = random.uniform(0.9, 1.0)  # Probability
    cpu_usage = random.uniform(10, 90)  # Percentage
    memory_consumption = random.uniform(500, 2000)  # MB

    return {
        'latency': latency,
        'bandwidth': bandwidth,
        'reliability': reliability,
        'cpu_usage': cpu_usage,
        'memory_consumption': memory_consumption
    }
