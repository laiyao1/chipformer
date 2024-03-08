## ChiPFormer: Transferable Chip Placement via Offline Decision Transformer

An offline RL placement method ChiPFormer, significantly improving the design quality and efficiency.

###  Publication
Yao Lai, Jinxin Liu, Zhentao Tang, Bin Wang, Jianye Hao, Ping Luo. "ChiPFormer: Transferable Chip Placement via Offline Decision Transformer." International Conference on Machine Learning, ICML (2023): 18346-18364.

[paper](https://arxiv.org/pdf/2306.14744.pdf) | [dataset](https://drive.google.com/drive/folders/1F7075SvjccYk97i2UWhahN_9krBvDCmr) | [website](https://sites.google.com/view/chipformer/home) | [video(English)](https://www.youtube.com/watch?v=9-EQmDjRLHQ) | [video(Mandarin)](https://www.bilibili.com/video/BV1ym4y177CC/)


### Usage

#### Download the offline placement dataset

Download the offline placement dataset from [Google Drive](https://drive.google.com/drive/folders/1F7075SvjccYk97i2UWhahN_9krBvDCmr). (We provide the placement data including 8 benchmarks: adaptec1-4, bigblue1-4, but you can just download a single benchmark file for training.)

For quick start, you can directly unzip the adaptec1_small.pkl data.

```
tar -zxvf adaptec1_small.pkl.tar.gz
```

#### Download placement benchmark files

Our test dataset including ISPD05 and ICCAD04 dataset. We provide the *adaptec1* dataset for quick start. For all benchmarks, they can also be downloaded from the *placement_bench.zip* file in [Google Drive](https://drive.google.com/drive/folders/1F7075SvjccYk97i2UWhahN_9krBvDCmr) .

#### Quick start

Our ChiPFormer includes the pretraining and finetuning two parts.

- For pretraining:

```
python run_dt_place.py
```

The dataset file for training can be modified in *create_dataset.py*. 
The saved models are in the folder *save_models*.

- For finetuning:

```
python odt.py --benchmark=adaptec1
```
The model path for funetuning can be modified in *odt.py*. Typically, the optimal model files obtained from pre-training are used to fine-tune.

### Parameter

For *run_dt_place.py*:

- **seed** Random seed.
- **context_length** Maximum length of decision transformer.
- **epochs** Maximum training epochs.
- **batch_size** Batch size.
- **cuda** GPU label for use.
- **is_eval_only** Whether to evaluate function. (In evaluation function, it will place all macros rather than the maximum length number of macros)
- **test_all_macro** Whether to place all existing macros.

For *odt.py*:
- **replay_size** Replay buffer size for finetuning.
- **traj_len** Maximum length of decision transformer.
- **batch_size** Batch size.
- **benchmark** Circuit benchmark for finetuning.
- **max_online_iters** Maximum number of iterations for finetuning.
- **eval_interval** Evaluation every how many iterations.
- **exploration_rtg** Return-to-go value for exploration.
- **is_fifo** Whether to use fifo buffer or priority queue buffer. 
- **cuda** GPU label for use.

### Dependency
- [Python](https://www.python.org/) >= 3.9

- [Pytorch](https://pytorch.org/) >= 1.10

  - Other versions may also work, but not tested

- [tqdm](https://tqdm.github.io/)


### Reference code
The code refers to the following open source repos:
- [decision-transformer](https://github.com/kzl/decision-transformer)
- [online-dt](https://github.com/facebookresearch/online-dt)

### Citation
If you find our paper/code useful in your research, please cite

```
@inproceedings{lai2023chipformer,
  author       = {Lai, Yao and Liu, Jinxin and Tang, Zhentao and Wang, Bin and Hao, Jianye and Luo, Ping},
  title        = {ChiPFormer: Transferable Chip Placement via Offline Decision Transformer},
  booktitle    = {International Conference on Machine Learning, {ICML} 2023, 23-29 July
                  2023, Honolulu, Hawaii, {USA}},
  series       = {Proceedings of Machine Learning Research},
  volume       = {202},
  pages        = {18346--18364},
  publisher    = {{PMLR}},
  year         = {2023},
  url          = {https://proceedings.mlr.press/v202/lai23c.html},
}
```

