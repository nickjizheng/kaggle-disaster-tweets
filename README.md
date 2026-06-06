# Disaster Tweet Classification

This repository contains the research project I worked on for detecting disaster-related tweets in the Kaggle Natural Language Processing with Disaster Tweets competition.

The project write-up is available here: [Kaggle.pdf](Kaggle.pdf).

Training and inference scripts are included for several ELECTRA-based model variants.

## Scripts

- `baseline_train.py` / `baseline_test.py`: simplified ELECTRA baseline
- `MC_train.py` / `MC_Test.py`: multi-channel ELECTRA model
- `Aux.py` / `Aux_Test.py`: auxiliary hierarchical ELECTRA model
- `CEMN.py` / `CEMN_Test.py`: catastrophic event memory network model
- `Aux+MC_train.py` / `Aux+MC_Test.py`: auxiliary plus multi-channel model
- `Aux+MC+CEMN_train.py` / `Aux+MC+CEMN_Test.py`: auxiliary plus multi-channel plus CEMN model
- `M-C + Aux_Train.py` / `M-C + Aux_Test.py`: original combined multi-channel plus auxiliary scripts

## Inference

Each test script loads `kaggle/test.csv`, finds the latest matching checkpoint in `/root/autodl-fs/model` by default, and writes Kaggle submission files into `kaggle/`.

```bash
python baseline_test.py
python MC_Test.py --model-path /root/autodl-fs/model/multichannel_only_f1_0.8123.pth
```

Use `--electra-path`, `--test-file`, `--model-dir`, or `--output-dir` to override the default paths.
