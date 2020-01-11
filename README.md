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
## Tagger
``` bash
python tagger.py --checkpoint_path logs01\NER_2020-01-08_23-04-23
```
Tagger output
``` bash
(NER) C:\Users\Warrior\Code\Practise\NER-English\Code>curl -X GET http://localhost:8991/gettag?text="Multiple%20Fatalities%20reported%20in%20Washington%20Navy%20Yard%shooting,via%20nytimes%20http://nyti.ms/1dipdeA"
[["Multiple", "O"], ["Fatalities", "O"], ["reported", "O"], ["in", "O"], ["Washington", "B-facility"], ["Navy", "I-facility"], ["Yard%shooting", "O"], [",via", "O"], ["nytimes", "O"], ["http", "O"], ["://nyti.ms/1dipdeA", "O"]]
```
