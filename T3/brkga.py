###############################################################################
# main_minimal.py: minimal script for calling BRKGA algorithms to solve
#                  instances of the Traveling Salesman Problem.
#
# (c) Copyright 2019, Carlos Eduardo de Andrade.
# All Rights Reserved.
#
# This code is released under LICENSE.md.
#
# Created on:  Nov 18, 2019 by ceandrade
# Last update: Jul 17, 2025 by gustavognz
###############################################################################

import sys
import os

from brkga_mp_ipr.enums import Sense
from brkga_mp_ipr.types_io import load_configuration
from brkga_mp_ipr.algorithm import BrkgaMpIpr
from tsp_instance import TSPInstance
from tsp_decoder import TSPDecoder

###############################################################################

def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python main_minimal.py <seed> <config-file> "
              "<num-generations> <tsp-instance-file>")
        sys.exit(1)

    ########################################
    # Read the command-line arguments and the instance
    ########################################

    seed = int(sys.argv[1])
    configuration_file = sys.argv[2]
    num_generations = int(sys.argv[3])
    instance_file = sys.argv[4]

    nome_instancia = os.path.splitext(os.path.basename(instance_file))[0]

    print("Reading data...")
    instance = TSPInstance(instance_file)

    ########################################
    # Define calado por instância
    ########################################

    calado_burma14 = [
        33.224206677231756, 28.224206677231756, 28.224206677231756,
        33.224206677231756, 28.224206677231756, 28.224206677231756,
        33.224206677231756, 33.224206677231756, 33.224206677231756,
        28.224206677231756, 33.224206677231756, 28.224206677231756,
        28.224206677231756, 33.224206677231756
    ]

    calado_gr17 = [
        39.06315011727191, 34.06315011727191, 34.06315011727191,
        34.06315011727191, 34.06315011727191, 34.06315011727191,
        39.06315011727191, 39.06315011727191, 39.06315011727191,
        34.06315011727191, 39.06315011727191, 39.06315011727191,
        34.06315011727191, 39.06315011727191, 34.06315011727191,
        39.06315011727191, 34.06315011727191
    ]

    calado_gr21 = [
        48.78271708766145, 43.78271708766145, 43.78271708766145,
        43.78271708766145, 43.78271708766145, 48.78271708766145,
        48.78271708766145, 48.78271708766145, 43.78271708766145,
        43.78271708766145, 48.78271708766145, 43.78271708766145,
        43.78271708766145, 43.78271708766145, 48.78271708766145,
        48.78271708766145, 48.78271708766145, 43.78271708766145,
        43.78271708766145, 48.78271708766145, 48.78271708766145
    ]

    calado_ulysses16 = [
        36.95466563135283, 31.95466563135283, 31.95466563135283,
        31.95466563135283, 31.95466563135283, 31.95466563135283,
        36.95466563135283, 36.95466563135283, 36.95466563135283,
        36.95466563135283, 31.95466563135283, 31.95466563135283,
        31.95466563135283, 36.95466563135283, 36.95466563135283,
        36.95466563135283
    ]

    calado_ulysses22 = [
        50.56170480896554, 45.56170480896554, 45.56170480896554,
        45.56170480896554, 45.56170480896554, 50.56170480896554,
        50.56170480896554, 50.56170480896554, 45.56170480896554,
        45.56170480896554, 50.56170480896554, 45.56170480896554,
        45.56170480896554, 50.56170480896554, 50.56170480896554,
        50.56170480896554, 50.56170480896554, 45.56170480896554,
        45.56170480896554, 50.56170480896554, 50.56170480896554,
        45.56170480896554
    ]

    calado_por_instancia = {
        "burma14": calado_burma14,
        "gr17": calado_gr17,
        "gr21": calado_gr21,
        "ulysses16": calado_ulysses16,
        "ulysses22": calado_ulysses22
    }

    calado_portos = calado_por_instancia.get(
        nome_instancia,
        [float("inf")] * instance.num_nodes  # fallback: sem restrição
    )

    ########################################
    # Read algorithm parameters
    ########################################

    print("Reading parameters...")
    brkga_params, _ = load_configuration(configuration_file)

    ########################################
    # Generate demands similar to PLI code
    ########################################
    
    import random
    
    # Generate demands like in your PLI code (1.5 to 3.5)
    random.seed(42)
    demanda_portos = [0.0]  # depot has no demand
    for i in range(1, instance.num_nodes):
        demanda = random.uniform(1.5, 3.5)
        demanda_portos.append(demanda)
    
    print(f"Generated demands: {demanda_portos}")

    ########################################
    # Build the BRKGA data structures and initialize
    ########################################

    print("Building BRKGA data and initializing...")

    decoder = TSPDecoder(instance, calado_portos=calado_portos, demanda_portos=demanda_portos)

    brkga = BrkgaMpIpr(
        decoder=decoder,
        sense=Sense.MINIMIZE,
        seed=seed,
        chromosome_size=instance.num_nodes,
        params=brkga_params
    )

    # NOTE: don't forget to initialize the algorithm.
    brkga.initialize()

    ########################################
    # Find good solutions / evolve
    ########################################

    print(f"Evolving {num_generations} generations...")
    brkga.evolve(num_generations)

    best_cost = brkga.get_best_fitness()
    print(f"Best cost: {best_cost}")

###############################################################################

if __name__ == "__main__":
    main()
