## How to Run

We run our experiments on NVIDIA L40S Tensor Core GPU with 44.7 GB of memory.
If you want to run with smaller GPU memory, please set `num_layers=2` in all the scripts, unless you are running on `rel-trial` for RelBench, in this case, please use `num_layers=4`.

To reproduce results on RelBench, run `benchmark/relbench_link_prediction_benchmark.py`.

```sh
python relbench_link_prediction_benchmark.py --model contextgnn
```

To reproduce results on IJCAI-Contest, run `benchmark/tgt_ijcai_benchmark.py`.

```
python tgt_ijcai_benchmark.py --model contextgnn
```

To run ContextGNN without optuna tuning, run

```sh
python relbench_example.py --dataset rel-trial --task site-sponsor-run --model contextgnn
```


## Install Instruction

```sh
pip install -e .

# to run examples and benchmarks
pip install -e '.[full]'
```
