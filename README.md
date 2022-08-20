Codebase for our AAAI paper [A Joint Model for Definition Extraction with Syntactic Connection and Semantic Consistency](https://ojs.aaai.org/index.php/AAAI/article/download/6444/6300).

## Requirements

- Python 3
- PyTorch
- tqdm
- sklearn
- pytorch-crf

## Training

To train the model, run:
```
bash train.sh model_name
```

Model checkpoints and logs will be saved to `./saved_models/model_name`.

For the complete list of parameters see `train.py`.

## Evaluation

To run evaluation on the test set, run:
```
python eval.py saved_models/model_name --dataset test
```

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

The evaluation script will print results of different metrics. We use the macro-f1 from sklearn as our main metric to compare the models.

## Retrain

Reload a pretrained model and finetune it, run:
```
python train.py --load --model_file saved_models/model_name/best_model.pt --optim sgd --lr 0.001
```

# License

CC BY-NC-SA 4.0.

# Citation

If you use the code released in this repo, please cite our paper:

```
@inproceedings{veyseh2020joint,
  title={A joint model for definition extraction with syntactic connection and semantic consistency},
  author={Veyseh, Amir and Dernoncourt, Franck and Dou, Dejing and Nguyen, Thien},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={9098--9105},
  year={2020}
}
```
