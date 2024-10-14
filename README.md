# contextgnn

- [Overleaf Latex link](https://www.overleaf.com/8255131161fxgzwccqftmz#5676c1)

- [contextgnn blogpost link](https://docs.google.com/document/d/1kcGl9zk_pHuZ5xE9HBBVCmJa6iLiZ_yOjX9eFPpHiXw/edit)

- [Spreadsheet of results](https://docs.google.com/spreadsheets/d/1bnNurVKLCgWjgvd9fCO-NexCgU75Xql9erfn6h3Wooo/edit?usp=sharing)


## How to Run

Run [`benchmark/relbench_link_prediction_benchmark.py`](https://github.com/kumo-ai/contextgnn/blob/master/benchmark/relbench_link_prediction_benchmark.py)

```sh
python relbench_link_prediction_benchmark.py --dataset rel-trial --task site-sponsor-run --model contextgnn
```


Run [`examples/relbench_example.py`](https://github.com/kumo-ai/contextgnn/blob/master/examples/relbench_example.py)

```sh
python relbench_example.py --dataset rel-trial --task site-sponsor-run --model contextgnn
python relbench_example.py --dataset rel-trial --task condition-sponsor-run --model contextgnn
```


## Install Instruction

```sh
pip install -e .

# to run examples and benchmarks
pip install -e '.[full]'
```
