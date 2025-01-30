import multiprocessing
import subprocess
import sys
import time

def run_client(script_name, client_id, scenario, order_id):
    """Run a client script with specified arguments."""
    subprocess.run([
        sys.executable, script_name,
        "--client_id", client_id,
        "--scenario", scenario,
        "--order_id", str(order_id)
    ])

if __name__ == '__main__':
    scenario_set = ['HIGH', 'HIGH', 'HIGH', 'HIGH', 'HIGH'] 
    num_clients = len(scenario_set)
    order_id = 1  

    processes = []
    for client_index, scenario in enumerate(scenario_set, start=1):
        client_id = f"client{client_index}"
        p = multiprocessing.Process(
            target=run_client,
            args=('fmnist_clients_H.py', client_id, scenario, order_id)
        )
        processes.append(p)
        p.start()
        time.sleep(1)  

    # Wait for all clients to finish
    for p in processes:
        p.join()

    print(f"Completed scenario: {scenario_set}")