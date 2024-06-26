#!/bin/bash
poetry run python app/federated_learning_sweep.py --json-path "cifar10_sweep_parameters.json" --sweep-name "cifar10"

