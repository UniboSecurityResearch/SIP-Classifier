# SIP-Classifier

This repository contains the dataset and supporting scripts used in the paper
"SIP-Classifier: Classifying SIP Signaling Sequences in IMS Using Transformers and Clustering".

**Repository Structure**

- `sip_dataset.csv`: CSV dataset of SIP signaling flows. Columns:
	- `Signaling Flow`: a string describing the SIP signaling sequence.
	- `Class`: numeric label for the sequence (`0 = benign`, `1 = anomalous`, `2 = unknown`).

- `sota/`: folder containing baseline model implementations and results used for comparison in experiments.
	- `CNN.py`: Convolutional Neural Network baseline from the paper [Detection of abnormal SIP signaling patterns: a deep learning comparison (Computers, 2022)](https://www.mdpi.com/2073-431X/11/2/27)
	- `corr.py`: Correlation-based baseline from the paper [Correlation-Based Abnormal SIP Dialog Identification: A Performance Comparison with Bayesian and Deep Learning Approaches (IEEE Access, 2025)](https://ieeexplore.ieee.org/abstract/document/11098849).
	- `HMM.py`: Hidden Markov Model baseline from the paper [A Machine Learning Approach for Prediction of Signaling SIP Dialogs (IEEE Access, 2021)](https://ieeexplore.ieee.org/abstract/document/9376867).
	- `LSTM1.py` and `LSTM2.py`: LSTM baseline implementations from the paper [Classification of Abnormal Signaling SIP Dialogs Through Deep Learning (IEEE Access, 2021)](https://ieeexplore.ieee.org/abstract/document/9648193)
	- `results/`: directory containing the results obtained from running the baseline models on the `sip_dataset.csv`.

- `util/`: utility scripts and small CSVs used for dataset splits, stats, and labeled subsets.
	- `anomalous.csv`: subset of the `sip_dataset.csv` containing only anomalous signaling flows, without duplicates.
	- `benign.csv`: subset of the `sip_dataset.csv` containing only benign signaling flows, without duplicates.
	- `split.py`: helper script to extract the `anomalous.csv` / `benign.csv` subsets.
	- `stats.py`: script to compute statistics over the dataset `sip_dataset.csv`.

## Cite us
If you find this work interesting and use it in your academic research, please cite our paper!

[![DOI](https://zenodo.org/badge/924094381.svg)](https://doi.org/10.5281/zenodo.15655738)
