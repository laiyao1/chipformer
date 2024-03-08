"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import time
import logging
import copy
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from mingpt.prim import prim_real


logger = logging.getLogger(__name__)

from mingpt.utils import sample
import torch
# import matplotlib.pyplot as plt

from mingpt.place_db import PlaceDB

seq_len = 256

benchmark_to_id = {'adaptec1': 0, 'adaptec2': 1, 'adaptec3': 2, 'adaptec4': 3,
            'bigblue1': 4, 'bigblue2': 5, 'bigblue3': 6, 'bigblue4': 7,
            'ibm01': 8, 'ibm02': 9, 'ibm03': 10, 'ibm04': 11,
            'ibm05': 12, 'ibm06': 13, 'ibm08': 14, 'ibm09': 15,
            'ibm10': 16, 'ibm11': 17, 'ibm12': 18}
benchmark_id_to_name = {v: k for k, v in benchmark_to_id.items()}

min_range = [7000, 80000, 23000, 10000, 1300, 2700, 13000, 16000, 
                14000, 10000, 4000, 6700, 1000, 1400, 21000, 10000, 7300, 2600, 5100]
max_range = [x*2 for x in min_range]

T_scores_x = []
T_scores_y = []

strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 


def get_norm_reward(reward, benchmark, benchmark_id, macro_num = 255):
    norm_reward = 0
    norm_reward = reward / (max_range[benchmark_id]-min_range[benchmark_id]) + \
        max_range[benchmark_id]/((max_range[benchmark_id]-min_range[benchmark_id]) * macro_num) 
    return norm_reward

grid = 84


benchmark_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4',
'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', 
'ibm01', 'ibm02', 'ibm03', 'ibm04']

benchmark_list_abbre = [x[0]+x[-1] for x in benchmark_list]
print("start")

# select offline data for training
placedb_g_lib = {
    "adaptec1": PlaceDB("adaptec1"),
    # "adaptec2": PlaceDB("adaptec2"),
    # "adaptec3": PlaceDB("adaptec3"),
    # "adaptec4": PlaceDB("adaptec4"),
    # "bigblue1": PlaceDB("bigblue1"),
    # "bigblue2": PlaceDB("bigblue2"),
    # "bigblue3": PlaceDB("bigblue3"),
    # "bigblue4": PlaceDB("bigblue4"),
    # "ibm01": PlaceDB("ibm01"),
    # "ibm02": PlaceDB("ibm02"),
    # "ibm03": PlaceDB("ibm03"),
    # "ibm04": PlaceDB("ibm04"),
    # "ibm05": PlaceDB("ibm05"),
    # "ibm06": PlaceDB("ibm06"),
}

if "adaptec1" in placedb_g_lib:
    circuit_feas = np.array(placedb_g_lib["adaptec1"].circuit_fea).reshape(1, 256 * 3)
else:
    circuit_feas = np.zeros((1, 256 * 3))

for i in range(len(benchmark_to_id)):
    if benchmark_id_to_name[i] in placedb_g_lib:
        circuit_feas = np.concatenate((circuit_feas, placedb_g_lib[benchmark_id_to_name[i]].circuit_fea.reshape(1, -1)))
    else:
        circuit_feas = np.concatenate((circuit_feas, np.zeros((1, 256 * 3))))

print("circuit_feas shape", circuit_feas.shape)


level_range = [0]

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.x_0 = None
        self.m_x_0 = None
        self.training_step = 0
        self.best_acc = 0.0
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # self.writer = SummaryWriter('./tf_logs/{}'.format(strftime))
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            accs = np.zeros(0)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t, m_x, b, st, cir, l) in pbar:
                self.training_step += 1
                # place data on the correct device
                x = x.to(self.device)
                m_x = m_x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)
                b = b.to(self.device)
                st = st.to(self.device)
                cir = cir.to(self.device)
                l = l.to(self.device)
                x_0 = x[0]
                # forward the model
                with torch.set_grad_enabled(is_train):

                    logits, loss, acc = model(x, y, y, r, t, m_x, b, st, cir, l)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    # self.writer.add_scalar('loss', loss, self.training_step)
                    acc = acc.mean()
                    # self.writer.add_scalar('acc', acc, self.training_step)
                    losses.append(loss.item())
                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    # self.writer.add_scalar('lr', lr, self.training_step)

                    # report progress
                    accs = np.append(accs, acc.cpu().numpy().mean())
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}.")
                    
                # save model
                if accs.mean() > self.best_acc + 0.02 and accs.mean()>=0.3:
                    self.best_acc = accs.mean()
                    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    model.eval()
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    torch.save(raw_model.state_dict(), "save_models/{}-{:.3f}.pkl".format(strftime, accs.mean()))
                    model.train()

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay
        
        # for testing 
        if self.config.is_eval_only:
            eval_return = {}
            T_scores_y_all_1 = []
            T_scores_y_all_err_1 = []
            T_scores_x_all_1 = []
            for i, benchmark in enumerate(benchmark_list):
                eval_return[benchmark] = {}
                T_scores_x = []
                T_scores_y = []
                T_rewards_x_all_macro = []
                T_rewards_y_all_macro = []
                
                # plt.cla()
                tmp_y_1 = []
                for level in level_range:
                    level /= 100.0
                    if not self.config.test_all_macro:
                        eval_return[benchmark][str(level)], T_scores = self.get_returns(level, 
                            benchmark = benchmark)
                        for t in T_scores:
                            T_scores_x.append(level)
                            T_scores_y.append(t)
                            tmp_y_1.append(t)
                    if self.config.test_all_macro:
                        print("is all macros")
                        _, T_rewards = self.get_returns(level, benchmark = benchmark, 
                            is_all_macro = True)
                        for t in T_rewards:
                            T_rewards_x_all_macro.append(level)
                            T_rewards_y_all_macro.append(t)

                if not self.config.test_all_macro:
                    T_scores_y_all_1.append(np.mean(tmp_y_1))
                    T_scores_y_all_err_1.append(np.std(tmp_y_1))
                    T_scores_x_all_1.append(i*3)
                    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
                    # plt.savefig('./{}-{}-N.png'.format(strftime, benchmark), 
                    #     dpi=300, bbox_inches='tight', pad_inches = 0.1)
                    # self.writer.add_scalars('eval_{}'.format(benchmark), eval_return[benchmark], -1)
                    # plt.cla()
            return 

        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            if (epoch + 1) % 50 == 0:
                if self.config.model_type == 'naive':
                    assert False
                elif self.config.model_type == 'reward_conditioned':
                    eval_return = {}
                    for i, benchmark in enumerate(benchmark_list):
                        if benchmark not in placedb_g_lib:
                            continue
                        eval_return[benchmark] = {}
                        T_scores_x = []
                        T_scores_y = []
                        # plt.cla()
                        for level in [0]:
                            level /= 100.0
                            eval_return[benchmark][str(level)], T_scores = self.get_returns(level, benchmark = benchmark)
                            for t in T_scores:
                                T_scores_x.append(level)
                                T_scores_y.append(t)
                        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
                        # plt.savefig('./{}-{}-{}.png'.format(strftime, benchmark, epoch), 
                        #     dpi=300, bbox_inches='tight', pad_inches = 0.1)
                        # plt.cla()
                        # self.writer.add_scalars('eval_{}'.format(benchmark), eval_return[benchmark], epoch)
                else:
                    raise NotImplementedError()


    def get_returns(self, ret, is_single = False, benchmark = "adaptec1", 
        is_shuffle_benchmark_id = False, is_all_macro = False):
        global circuit_feas
        self.model.train(False)
        args=Args(self.config.seed)
        env = Env(args, benchmark, is_all_macro)
        if is_all_macro and not is_single:
            fwrite = open("results/log_{}.log".format(benchmark), 'w')
        global benchmark_to_id
        benchmark_id = torch.tensor(benchmark_to_id[benchmark], dtype=torch.int64).reshape(1, 1)
        env.eval()

        T_rewards, T_Qs = [], []
        T_scores = []
        done = True

        if is_single:
            repeat_num = 1
        else:
            if self.config.is_eval_only:
                repeat_num = 10
            else:
                repeat_num = 1

        circuit_feas_for_benchmark = torch.tensor(circuit_feas[benchmark_id], dtype = torch.float32)

        for i in range(repeat_num):
            state, reward_sum, done, meta_state = env.reset()
            score_sum = 0
            assert reward_sum == 0
            rewards = []
            probs = []
            state = state.type(torch.float32).to(self.device).unsqueeze(0)
            meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)

            rtgs = [ret]

            sampled_action, action_probs = sample(self.model, state.unsqueeze(0), 1, temperature=1.0, sample=True, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps = torch.arange(0, 2, dtype = torch.int64).reshape(1, 2, 1).to(self.device), 
                meta_state = meta_state, benchmarks = benchmark_id.to(self.device),
                stepwise_returns = None,
                circuit_feas = circuit_feas_for_benchmark.to(self.device),
                is_random_shuffle = is_shuffle_benchmark_id)

            j = 0
            all_states = state.type(torch.float32)
            all_meta_states = meta_state.type(torch.float32)
            actions = []
            scores = []
            while True:
                if done:
                    state, reward_sum, done, meta_state = env.reset() # , 0, False
                    score_sum = 0
                action = sampled_action.cpu().numpy()[0,-1]

                if isinstance(action, int):
                    actions += [action]
                else:
                    actions += [action.item()]
                state, reward, done, meta_state = env.step(action)
                score = get_norm_reward(reward, benchmark, benchmark_to_id[benchmark], env.placed_num_macro)
                reward_sum += reward
                scores.append(score)
                rewards.append(reward)
                probs.append(action_probs[0, action].item())
                score_sum += score
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    T_scores.append(score_sum)
                    break

                state = state.type(torch.float32).unsqueeze(0).to(self.device)
                meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)
                all_states = torch.cat([all_states, state], dim=0)
                all_meta_states = torch.cat([all_meta_states, meta_state], dim=0)
                rtgs += [rtgs[-1] - score]
                
                sampled_action, action_probs = sample(self.model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                    actions=torch.tensor(np.array(actions), dtype=torch.long).to(self.device).unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps = torch.arange(0, min(j+2, seq_len), dtype= torch.int64).reshape(1, -1, 1).to(self.device),
                    meta_state = all_meta_states.unsqueeze(0), benchmarks = benchmark_id.to(self.device),
                    stepwise_returns = None,
                    circuit_feas = circuit_feas_for_benchmark.to(self.device),
                    is_random_shuffle = is_shuffle_benchmark_id)
        env.close()
        print("T_rewards", T_rewards)
        print("T_scores", T_scores)
        if not is_single:
            eval_return = max(T_rewards)
            avg_return = sum(T_rewards)/len(T_rewards)
            print("target return: %.2f, eval return: %.2f, avg return: %.2f" % (ret, 
                eval_return, avg_return))
        self.model.train(True)
        if not is_single:
            if is_all_macro:
                return eval_return, T_rewards
            else:
                return eval_return, T_scores
        else:
            outputs = {}
            lengths = [1] * all_states.shape[0] + [0] * (seq_len-all_states.shape[0])
            if all_states.shape[0] < seq_len:
                add_length = seq_len - all_states.shape[0]
                all_states = torch.cat((all_states, torch.zeros(add_length, 3, grid, grid).to(self.device)), axis = 0)
                actions.extend([0] * add_length)
                scores.extend([0] * add_length)
                all_meta_states = torch.cat((all_meta_states, torch.zeros(add_length, 2).to(self.device)), axis= 0)
                rewards.extend([0] * add_length)
                probs.extend([0] * add_length)
            if not is_all_macro:
                outputs['states'] = all_states.cpu().reshape(seq_len, -1)
                outputs['actions'] = torch.tensor(np.array(actions), dtype=torch.long).cpu().reshape(seq_len, -1)
                new_rtgs = [0]
                assert len(scores) == seq_len
                for i in range(seq_len):
                    new_rtgs.append(new_rtgs[-1] + scores[seq_len-i-1])
                new_rtgs.reverse()
                new_rtgs.pop(-1)
                outputs['rtgs'] = torch.tensor(np.array(new_rtgs), dtype=torch.float32).cpu().reshape(seq_len, -1)
                outputs['timesteps'] = torch.arange(0, min(j+2, seq_len), dtype= torch.int64).cpu().reshape(-1, 1)
                outputs['benchmarks'] = torch.tensor([benchmark_id], dtype=torch.int32).reshape(1) 
                outputs['meta_states'] = all_meta_states.cpu().reshape(seq_len, -1)
                outputs['reward'] = T_rewards[0]
                outputs['score'] = float(T_scores[0])
                outputs['rewards'] = torch.tensor(np.array(rewards), dtype=torch.float32).cpu().reshape(seq_len, -1)
                outputs['targets'] = [0] * seq_len
                outputs['probs'] = torch.tensor(np.array(probs), dtype=torch.float32).cpu().reshape(seq_len, -1)
                outputs['lengths'] = torch.tensor(np.array(lengths), dtype = torch.bool).cpu().reshape(seq_len, -1)
                last_target = 0  
                for i in range(seq_len-1, -1, -1):
                    outputs['targets'][i] = last_target * 0.95 + outputs['rewards'][i].item() # /200.0
                    last_target = outputs['targets'][i]
                outputs['targets'] = torch.tensor(np.array(outputs["targets"]), dtype=torch.float32).reshape(seq_len, 1)
                outputs['circuit_feas'] = circuit_feas_for_benchmark.reshape(-1)
            else:
                outputs['actions'] = torch.tensor(np.array(actions[:seq_len]), 
                    dtype=torch.long).cpu().reshape(seq_len, -1)
                outputs['reward'] = T_rewards[0]
                outputs['score'] =  float(T_scores[0])
            return outputs
    
    def get_hpwl(self, actions, benchmark):
        global circuit_feas
        self.model.train(False)
        args = Args(self.config.seed)
        env = Env(args, benchmark, is_all_macro = True)
        state, reward_sum, done, meta_state = env.reset()
        step_num = 0
        while not done:
            state, reward_sum, done, meta_state = env.step(int(actions[step_num].item()))
            step_num += 1
        hpwl, cost = env.comp_res()
        return hpwl, cost
    
    def get_remain_returns(self, actions, benchmark):
        global circuit_feas
        self.model.train(False)
        args = Args(self.config.seed)
        env = Env(args, benchmark, is_all_macro = True)
        benchmark_id = torch.tensor(benchmark_to_id[benchmark], dtype=torch.int64).reshape(1, 1)
        circuit_feas_for_benchmark = torch.tensor(circuit_feas[benchmark_id], dtype = torch.float32)
        actions = actions.squeeze().tolist()
        state, reward_sum, done, meta_state = env.reset()
        rtgs = [0]
        j = 0
        state = state.type(torch.float32).unsqueeze(0).to(self.device)
        meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)
        step_num = 0
        all_states = state.type(torch.float32)
        all_meta_states = meta_state.type(torch.float32)

        outputs = {}
        for step_num in range(len(actions)):
            state, reward, done, meta_state = env.step(actions[step_num])
            if done:
                break
            rtgs += [0]
            state = state.type(torch.float32).unsqueeze(0).to(self.device)
            meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)
            all_states = torch.cat([all_states, state], dim=0)
            all_meta_states = torch.cat([all_meta_states, meta_state], dim=0)
            j += 1
            reward_sum += reward
        while not done:
            sampled_action, action_probs = sample(self.model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                actions=torch.tensor(np.array(actions), dtype=torch.long).to(self.device).unsqueeze(0),
                rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps = torch.arange(0, min(j+2, seq_len), dtype= torch.int64).reshape(1, -1, 1).to(self.device),
                meta_state = all_meta_states.unsqueeze(0), benchmarks = benchmark_id.to(self.device),
                stepwise_returns = None,
                circuit_feas = circuit_feas_for_benchmark.to(self.device),
                is_random_shuffle = False)
            action = sampled_action.cpu().numpy()[0,-1]
            if isinstance(action, int):
                actions += [action]
            else:
                actions += [action.item()]
            state, reward, done, meta_state = env.step(action)
            state = state.type(torch.float32).unsqueeze(0).to(self.device)
            meta_state = meta_state.type(torch.float32).to(self.device).unsqueeze(0)
            all_states = torch.cat([all_states, state], dim=0)
            all_meta_states = torch.cat([all_meta_states, meta_state], dim=0)
            rtgs += [0]
            j += 1
            reward_sum += reward
        hpwl, cost = env.comp_res()
        outputs["reward"] = reward_sum
        outputs["actions"] = torch.tensor(np.array(actions), dtype=torch.long).cpu().reshape(-1, 1)
        outputs["hpwl"] = hpwl
        outputs["cost"] = cost
        return outputs


# for evaluation
class Env():
    def __init__(self, args, benchmark = "adaptec1", is_all_macro = False):
        global benchmark_to_id
        self.device = args.device
        self.placedb = placedb_g_lib[benchmark]
        # 256 macro or all macros
        self.is_all_macro = is_all_macro
        self.grid = grid
        self.num_macro_placed = 0
        if is_all_macro and benchmark != "bigblue2" and "ibm" not in benchmark:
            self.placed_num_macro = self.placedb.node_cnt
        else:
            self.placed_num_macro = min(self.placedb.node_cnt, 256)
        self.init_net_min_max_ord = {}
        self.ratio = self.placedb.max_height / self.grid
        for node_name in self.placedb.fixed_node_info:
            for net_name in self.placedb.node_to_net_dict[node_name]:
                raw_x = self.placedb.fixed_node_info[node_name]['raw_x']
                raw_y = self.placedb.fixed_node_info[node_name]['raw_y']
                pin_x = round((raw_x - self.placedb.offset + self.placedb.fixed_node_info[node_name]['x']/2 + \
                    self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"])/self.ratio)
                pin_y = round((raw_y - self.placedb.offset + self.placedb.fixed_node_info[node_name]['y']/2 + \
                    self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"])/self.ratio)
                if net_name in self.init_net_min_max_ord:
                    start_x = self.init_net_min_max_ord[net_name]['min_x']
                    end_x = self.init_net_min_max_ord[net_name]['max_x']
                    start_y = self.init_net_min_max_ord[net_name]['min_y']
                    end_y = self.init_net_min_max_ord[net_name]['max_y']
                    if pin_x > self.init_net_min_max_ord[net_name]['max_x']:
                        self.init_net_min_max_ord[net_name]['max_x'] = pin_x
                    elif pin_x < self.init_net_min_max_ord[net_name]['min_x']:
                        self.init_net_min_max_ord[net_name]['min_x'] = pin_x
                    if pin_y > self.init_net_min_max_ord[net_name]['max_y']:
                        self.init_net_min_max_ord[net_name]['max_y'] = pin_y
                    elif pin_y < self.init_net_min_max_ord[net_name]['min_y']:
                        self.init_net_min_max_ord[net_name]['min_y'] = pin_y
                    start_x = self.init_net_min_max_ord[net_name]['min_x']
                    end_x = self.init_net_min_max_ord[net_name]['max_x']
                    start_y = self.init_net_min_max_ord[net_name]['min_y']
                    end_y = self.init_net_min_max_ord[net_name]['max_y']
                else:
                    self.init_net_min_max_ord[net_name] = {}
                    self.init_net_min_max_ord[net_name]['max_x'] = pin_x
                    self.init_net_min_max_ord[net_name]['min_x'] = pin_x
                    self.init_net_min_max_ord[net_name]['max_y'] = pin_y
                    self.init_net_min_max_ord[net_name]['min_y'] = pin_y

        self.net_min_max_ord = copy.deepcopy(self.init_net_min_max_ord)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        pass

    def _reset_buffer(self):
        pass

    def reset(self):
        self.net_placed_set = {}
        self.net_min_max_ord = copy.deepcopy(self.init_net_min_max_ord) #
        print("len self.net_min_max_ord", len(self.net_min_max_ord))
        self.ratio = self.placedb.max_height / self.grid
        self.node_name_list = self.placedb.node_id_to_name
        self.node_pos = {}
        self.num_macro_placed = 0
        
        self.canvas = np.zeros((self.grid, self.grid))
        next_x = math.ceil(max(1, 
            self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['x']/self.ratio))
        next_y = math.ceil(max(1, 
            self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['y']/self.ratio))
        net_img = self.get_net_img()
        mask = self.get_mask(next_x, next_y)
        self.net_img = net_img
        self.mask = mask
        states = torch.from_numpy(np.stack((self.canvas, self.net_img, self.mask), axis = 0))
        done = False
        reward = 0
        
        next_x = self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['x']/self.ratio
        next_y = self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['y']/self.ratio
        info = torch.from_numpy(np.array([next_x, next_y]))
        return states, reward, done, info

    def step(self, action):
        reward = 0
        x = round(action // self.grid)
        y = round(action % self.grid)
        node_name = self.placedb.node_id_to_name[self.num_macro_placed]
        
        size_x = math.ceil(max(1, self.placedb.node_info[node_name]['x']/self.ratio))
        size_y = math.ceil(max(1, self.placedb.node_info[node_name]['y']/self.ratio))

        for net_name in self.placedb.node_to_net_dict[node_name]:
            pin_x = round((x * self.ratio + self.placedb.node_info[node_name]['x']/2 + \
                    self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"])/self.ratio)
            pin_y = round((y * self.ratio + self.placedb.node_info[node_name]['y']/2 + \
                self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"])/self.ratio)
            if net_name in self.net_min_max_ord:
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                weight = 1.0
                if 'weight' in self.placedb.net_info[net_name]:
                    weight = self.placedb.net_info[net_name]['weight']
                if pin_x > self.net_min_max_ord[net_name]['max_x']:
                    reward += weight * (self.net_min_max_ord[net_name]['max_x'] - pin_x)
                    self.net_min_max_ord[net_name]['max_x'] = pin_x
                elif pin_x < self.net_min_max_ord[net_name]['min_x']:
                    reward += weight * (pin_x - self.net_min_max_ord[net_name]['min_x'])
                    self.net_min_max_ord[net_name]['min_x'] = pin_x
                if pin_y > self.net_min_max_ord[net_name]['max_y']:
                    reward += weight * (self.net_min_max_ord[net_name]['max_y'] - pin_y)
                    self.net_min_max_ord[net_name]['max_y'] = pin_y
                elif pin_y < self.net_min_max_ord[net_name]['min_y']:
                    reward += weight * (pin_y - self.net_min_max_ord[net_name]['min_y'])
                    self.net_min_max_ord[net_name]['min_y'] = pin_y
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
            else:
                self.net_min_max_ord[net_name] = {}
                self.net_min_max_ord[net_name]['max_x'] = pin_x
                self.net_min_max_ord[net_name]['min_x'] = pin_x
                self.net_min_max_ord[net_name]['max_y'] = pin_y
                self.net_min_max_ord[net_name]['min_y'] = pin_y
                start_x = self.net_min_max_ord[net_name]['min_x']
                end_x = self.net_min_max_ord[net_name]['max_x']
                start_y = self.net_min_max_ord[net_name]['min_y']
                end_y = self.net_min_max_ord[net_name]['max_y']
                reward += 0
        
        self.node_pos[self.node_name_list[self.num_macro_placed]] = (x, y, size_x, size_y)

        self.canvas[x : x+size_x-1, y : y+size_y-1] = 1.0
        
        self.num_macro_placed += 1
        next_x = 0
        next_y = 0
        if self.num_macro_placed < self.placed_num_macro:
            next_x = math.ceil(max(1, 
                self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['x']/self.ratio))
            next_y = math.ceil(max(1, 
                self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['y']/self.ratio))
        self.mask = self.get_mask(next_x, next_y)
        if self.num_macro_placed < self.placed_num_macro:
            self.net_img = self.get_net_img()
        if self.net_img.max() > 0:
            self.net_img /= 12800
        if self.num_macro_placed >= self.placed_num_macro:
            done = True
        else:
            done = False
        states = np.stack((self.canvas, self.net_img, self.mask), axis = 0)
        if self.num_macro_placed < self.placed_num_macro:
            next_x = self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['x']/self.ratio
            next_y = self.placedb.node_info[self.placedb.node_id_to_name[self.num_macro_placed]]['y']/self.ratio
        else:
            next_x = 0.0
            next_y = 0.0
        return torch.from_numpy(states), reward, done, \
            torch.from_numpy(np.array([next_x, next_y]))
    
    def get_mask(self, next_x, next_y):
        mask = np.zeros((self.grid, self.grid))
        for node_name in self.node_pos:
            startx = max(0, self.node_pos[node_name][0] - next_x + 1)
            starty = max(0, self.node_pos[node_name][1] - next_y + 1)
            endx = min(self.node_pos[node_name][0] + self.node_pos[node_name][2] - 1, self.grid - 1)
            endy = min(self.node_pos[node_name][1] + self.node_pos[node_name][3] - 1, self.grid - 1)
            mask[startx: endx + 1, starty : endy + 1] = 1
        mask[self.grid - next_x + 1:,:] = 1
        mask[:, self.grid - next_y + 1:] = 1
        return mask
    
    def get_net_img(self):
        net_img = np.zeros((self.grid, self.grid))
        next_node_name = self.placedb.node_id_to_name[self.num_macro_placed]
        for net_name in self.placedb.node_to_net_dict[next_node_name]:
            if net_name in self.net_min_max_ord:
                delta_pin_x = round((self.placedb.node_info[next_node_name]['x']/2 + \
                    self.placedb.net_info[net_name]["nodes"][next_node_name]["x_offset"])/self.ratio)
                delta_pin_y = round((self.placedb.node_info[next_node_name]['y']/2 + \
                    self.placedb.net_info[net_name]["nodes"][next_node_name]["y_offset"])/self.ratio)
                start_x = self.net_min_max_ord[net_name]['min_x'] - delta_pin_x
                end_x = self.net_min_max_ord[net_name]['max_x'] - delta_pin_x
                start_y = self.net_min_max_ord[net_name]['min_y'] - delta_pin_y
                end_y = self.net_min_max_ord[net_name]['max_y'] - delta_pin_y
                start_x = min(start_x, self.grid)
                start_y = min(start_y, self.grid)
                if not 'weight' in self.placedb.net_info[net_name]:
                    weight = 1.0
                else:
                    weight = self.placedb.net_info[net_name]['weight']
                for i in range(0, start_x):
                    net_img[i, :] += (start_x - i) * weight
                for i in range(end_x+1, self.grid):
                    net_img[i, :] +=  (i- end_x) * weight
                for j in range(0, start_y):
                    net_img[:, j] += (start_y - j) * weight
                for j in range(end_y+1, self.grid):
                    net_img[:, j] += (j - end_y) * weight
        return net_img

    def comp_res(self):
        hpwl = 0.0
        cost = 0.0
        for net_name in self.placedb.net_info:
            max_x = 0.0
            min_x = self.placedb.max_height * 1.5
            max_y = 0.0
            min_y = self.placedb.max_height * 1.5
            for node_name in self.placedb.net_info[net_name]["nodes"]:
                if node_name not in self.node_pos and node_name not in self.placedb.fixed_node_info:
                    continue
                if node_name in self.node_pos:
                    h = self.placedb.node_info[node_name]['x']
                    w = self.placedb.node_info[node_name]['y']
                    pin_x = self.node_pos[node_name][0] * self.ratio + h / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"]
                    pin_y = self.node_pos[node_name][1] * self.ratio + w / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"]
                else:
                    assert node_name in self.placedb.fixed_node_info
                    h = self.placedb.fixed_node_info[node_name]['x']
                    w = self.placedb.fixed_node_info[node_name]['y']
                    pin_x = self.placedb.fixed_node_info[node_name]['raw_x'] + h / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["x_offset"]
                    pin_y = self.placedb.fixed_node_info[node_name]['raw_y'] + w / 2.0 + self.placedb.net_info[net_name]["nodes"][node_name]["y_offset"]
                max_x = max(pin_x, max_x)
                min_x = min(pin_x, min_x)
                max_y = max(pin_y, max_y)
                min_y = min(pin_y, min_y)
            for port_name in self.placedb.net_info[net_name]["ports"]:
                h = self.placedb.port_info[port_name]['x']
                w = self.placedb.port_info[port_name]['y']
                pin_x = h
                pin_y = w
                max_x = max(pin_x, max_x)
                min_x = min(pin_x, min_x)
                max_y = max(pin_y, max_y)
                min_y = min(pin_y, min_y)
            if min_x <= self.placedb.max_height:
                hpwl_tmp = (max_x - min_x) + (max_y - min_y)
            else:
                hpwl_tmp = 0
            if "weight" in self.placedb.net_info[net_name]:
                hpwl_tmp *= self.placedb.net_info[net_name]["weight"]
            hpwl += hpwl_tmp
            net_node_set = set.union(set(self.placedb.net_info[net_name]["nodes"]),
                                set(self.placedb.net_info[net_name]["ports"]))
            for net_node in list(net_node_set):
                if net_node not in self.node_pos and net_node not in self.placedb.port_info \
                    and net_node not in self.placedb.fixed_node_info:
                    net_node_set.discard(net_node)
            prim_cost = prim_real(net_node_set, self.node_pos, 
                self.placedb.net_info[net_name]["nodes"], self.ratio, 
                self.placedb.node_info, self.placedb.port_info, self.placedb.fixed_node_info, self.placedb)
            if "weight" in self.placedb.net_info[net_name]:
                prim_cost *= self.placedb.net_info[net_name]["weight"]
            assert hpwl_tmp <= prim_cost +1e-5
            cost += prim_cost
        return hpwl, cost

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        pass

    def close(self):
        pass

class Args:
    def __init__(self, seed = 42):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.history_length = 4
