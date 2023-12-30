import os
import logging
from mingpt.utils import set_seed
import numpy as np
import torch
from torch.utils.data import Dataset
from mingpt.model_placement import GPT, GPTConfig
from mingpt.trainer_placement import Trainer, TrainerConfig
from mingpt.trainer_placement import circuit_feas
import torch
import argparse
from create_dataset import create_dataset

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', type=str, default='1,2,3')
parser.add_argument('--is_eval_only', action='store_true')
parser.add_argument('--test_all_macro', action='store_true')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

set_seed(args.seed)
grid = 84
seq_len = args.context_length

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, 
            timesteps, meta_data = None, obss_wire = None, obss_mask = None, benchmarks = None,
            stepwise_returns = None, lengths = None):
        assert block_size % 3 == 0
    
        self.block_size = block_size
        self.seq_len = self.block_size // 3
        self.vocab_size = int((grid) ** 2)
        self.data = data
        self.actions = actions
        print("data raw shape", data.shape)
        self.done_idxs = done_idxs
        self.meta_data = meta_data
        print("meta_data raw shape", meta_data.shape)
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.obss_wire = obss_wire
        self.obss_mask = obss_mask
        self.benchmarks = benchmarks
        self.stepwise_returns = stepwise_returns
        self.lengths = lengths
    
    def __len__(self):
        return len(self.data)//self.seq_len

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        idx = idx * self.seq_len
        done_idx = idx + self.seq_len
        if self.obss_mask is None:
            states = torch.tensor(np.array(self.data[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        else:
            tmp_obss = torch.tensor(np.array(self.data[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1)
            tmp_obss_wire = torch.tensor(np.array(self.obss_wire[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1)
            tmp_obss_mask = torch.tensor(np.array(self.obss_mask[idx:done_idx]), 
                dtype=torch.float32).reshape(block_size, -1)
            states = torch.cat((tmp_obss, tmp_obss_wire, tmp_obss_mask), dim=1)

        meta_states = torch.tensor(np.array(self.meta_data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        benchmarks = torch.tensor(self.benchmarks[idx:done_idx], dtype=torch.int64).unsqueeze(1)
        stepwise_returns = torch.tensor(self.stepwise_returns[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        benchmark_id = int(self.benchmarks[idx][0])
        circuit_feas_for_benchmark = torch.tensor(circuit_feas[benchmark_id], dtype = torch.float32)
        length = torch.zeros((block_size,), dtype=torch.bool)
        length[:int(self.lengths[idx][0])] = 1
        return states, actions, rtgs, timesteps, meta_states, \
            benchmarks, stepwise_returns, circuit_feas_for_benchmark, length


# load the pickle file and build the dataset
obss, actions, returns, done_idxs, rtgs, \
    timesteps, meta_data, obss_wire, obss_mask, benchmarks, \
    stepwise_returns, lengths = \
    create_dataset(0, 0, 0, 
    0, 0, args.is_eval_only)

print("create dataset finish.")
print("obss shape", len(obss), len(obss[0]))
print("actions", actions)
print("actions shape", len(actions))
print("returns shape", len(returns))
print("done_idxs shape", len(done_idxs))
print("rtgs", rtgs)
print("rtgs shape", len(rtgs))

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
print("lengths shape", len(lengths))
train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, 
    done_idxs, rtgs, timesteps, meta_data, obss_wire, 
    obss_mask, benchmarks, stepwise_returns, lengths)

print("!!!! max(timesteps)", max(timesteps))
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, 
                  model_type="reward_conditioned", max_timestep=max(timesteps))
model = GPT(mconf)

model_path = "save_models/trained_model.pkl" 

if model_path is not None:
    state_dict = torch.load(model_path)
    for k,v in state_dict.items():
        if "module." in k:
            state_dict[k.split('.', 1)[1]] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict, strict = True)
model.eval()
get_parameter_number(model)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type="reward_conditioned", max_timestep=max(timesteps),
                      draw_placement = True, is_eval_only = args.is_eval_only,
                      test_all_macro = args.test_all_macro)
print("trainerconfig finish")
trainer = Trainer(model, train_dataset, None, tconf)
print("trainer build finish")
trainer.train()