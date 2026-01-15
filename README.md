# EADGN：Traffic Forecasting Model

This repository contains the official implementation of the EADGN.

### Repository Content
This codebase includes the essential components required for experimental verification:
* **Raw Datasets**: The original traffic flow data.
* **Processed Datasets**: Traffic speed matrices fused with event data.
* **Preprocessing Scripts**: Specific scripts used for data matching and alignment.
* **Core Model Implementation**: The key implementation of the EADGN model layers.

> **Note:** While this repository provides the core components necessary for reproduction and verification, the complete suite of core modules is currently withheld to maintain confidentiality and anonymity during the review process. The full codebase will be completely open-sourced immediately upon the paper's publication.

## Requirements
Python 3.6、Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser


## Run
```bash
python train.py --config configs/bj500.yaml
```
