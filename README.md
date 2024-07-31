# hybridgnn

- [Overleaf Latex link](https://www.overleaf.com/8255131161fxgzwccqftmz#5676c1)

- [HybridGNN blogpost link](https://docs.google.com/document/d/1kcGl9zk_pHuZ5xE9HBBVCmJa6iLiZ_yOjX9eFPpHiXw/edit)

- [Spreadsheet of results](https://docs.google.com/spreadsheets/d/1bnNurVKLCgWjgvd9fCO-NexCgU75Xql9erfn6h3Wooo/edit?usp=sharing)


## How to Run

run [`benchmark/relbench_link_prediction_benchmark.py`](https://github.com/kumo-ai/hybridgnn/blob/master/benchmark/relbench_link_prediction_benchmark.py)

```
python relbench_link_prediction_benchmark.py --dataset rel-trial --task site-sponsor-run --model hybridgnn

```


run [`examples/relbench_example.py`](https://github.com/kumo-ai/hybridgnn/blob/master/examples/relbench_example.py)

```
python relbench_example.py --dataset rel-trial --task site-sponsor-run --model hybridgnn

python relbench_example.py --dataset rel-trial --task condition-sponsor-run --model hybridgnn
```


## Install Instruction

```
pip install .
pip install relbench[example]
pip install optuna
```
