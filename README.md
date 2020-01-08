# Name-entity Recognition
This is a project to demo how a neural network can be used to tag entities.

## Train
```bash
python train.py --log_dir logs01 --data_paths ..\data
```

## Eval
```bash
python eval.py --checkpoint_path logs01\NER_2020-01-08_23-04-23 --data_paths ..\data
```
