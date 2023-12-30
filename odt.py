# Online Decision Transformer for finetuning the pre-trained model

import os
import argparse
import pickle
import time
import numpy as np
import torch
import math
import heapq

from replay_buffer import ReplayBuffer
from mingpt.model_placement import GPT, GPTConfig
from mingpt.trainer_placement import Trainer, TrainerConfig
# from torch.utils.tensorboard import SummaryWriter
from online_trainer import SequenceTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--replay_size", type=int, default=64)
parser.add_argument("--traj_len", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--benchmark", type=str, default="adaptec1")
parser.add_argument("--max_online_iters", type=int, default=110)
parser.add_argument("--eval_interval", type=int, default=10)
parser.add_argument("--exploration_rtg", type=float, default=1.1)
parser.add_argument('--is_fifo', action='store_true')
parser.add_argument('--cuda', type=str, default='1,2,3')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


model_path = "save_models/trained_model.pkl"
grid = 84
total_traj_cnt = 0
global_best_score = -1.0
global_best_reward = 1.0e8
global_best_raw_reward = 1.0e8
global_best_hpwl = 1.0e8
global_best_cost = 1.0e8

strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
# writer = SummaryWriter('./tf_odt_logs/finetune-{}-{}'.format(strftime, args.benchmark))
if not os.path.exists("./finetune_odt_log"):
    os.makedirs("./finetune_odt_log")
fwrite = open("./finetune_odt_log/finetune-{}-{}{}.log".format(strftime, args.benchmark
    , "" if not args.is_fifo else "-fifo"), 'w')

if not os.path.exists("./finetune_odt_real_log"):
    os.makedirs("./finetune_odt_real_log")
flog = open("./finetune_odt_real_log/finetune-{}-{}{}.log".format(strftime, args.benchmark
    , "" if not args.is_fifo else "-fifo"), 'w')

class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        return traj["states"], traj["meta_states"], traj["actions"], \
            traj["rewards"], traj["rtgs"], traj["timesteps"], traj["benchmarks"], \
            traj["circuit_feas"], traj["targets"], traj["probs"], traj["lengths"]

    def __len__(self):
        return len(self.sampling_ind)


def sample_trajs(replay_buffer, sample_size):
    p_sample = np.zeros((len(replay_buffer.trajectories)))
    score_sum = 0
    for i in range(len(replay_buffer)):
        score_sum += replay_buffer.trajectories[i]['score']
        p_sample[i] = math.exp(replay_buffer.trajectories[i]['score'] * 100)
    print("avg score = {}".format(score_sum/len(replay_buffer)))
    p_sample /= p_sample.sum()
    
    inds = np.random.choice(
        np.arange(len(replay_buffer.trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    print("inds", inds)
    return inds


def create_dataloader(replay_buffer, batch_size, num_workers = 4):
    sample_size = batch_size
    sampling_ind = sample_trajs(replay_buffer, sample_size)
    subset = SubTrajectory(replay_buffer.trajectories, sampling_ind=sampling_ind)
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )


def work(model, pretrain_trainer, replay_buffer, tconf):
    global total_traj_cnt
    global global_best_score
    global global_best_raw_reward
    global global_best_reward
    global global_best_hpwl
    global global_best_cost
    th = 0.5

    start_time = time.time()
    last_update = 0
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print("device", device)
        model = torch.nn.DataParallel(model).to(device)

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = raw_model.configure_optimizers(tconf)
    
    online_trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=device,
    )
    online_iter = 0

    dataloader = create_dataloader(replay_buffer, args.batch_size)
    train_outputs = online_trainer.train_iteration(
        dataloader=dataloader,
        threshold=th,
        traj_cnt = total_traj_cnt,
        decrease_entropy = False,
    )
    logs = {}
    logs.update(train_outputs)
    while online_iter < args.max_online_iters:
        
        print("online_iter = {}".format(online_iter))
        print("score_avg = {:.4f}, worst_score = {}".format(replay_buffer.score_sum/len(replay_buffer),
            replay_buffer.worst_score))
        pretrain_trainer.model.eval()
        add_new_traj_cnt = 0
        decrease_entropy = False
        while add_new_traj_cnt < args.replay_size / 8:
            while True:
                outputs = pretrain_trainer.get_returns(args.exploration_rtg, 
                    is_single = True, benchmark = args.benchmark)

                # writer.add_scalar('score', outputs['score'], total_traj_cnt)
                # writer.add_scalar('reward', outputs['reward'], total_traj_cnt)
                flog.write("{:.2f}\t{}\t{:.4f}\n".format(-outputs['reward'], total_traj_cnt, time.time() - start_time))
                flog.flush()
                if -outputs['reward'] * 0.98 < global_best_raw_reward:
                    if th >= 0.5:
                        th = 0.5
                    if outputs['score'] >= global_best_score * 1.003:
                        decrease_entropy = True
                        th *= 0.5
                    outputs_all = pretrain_trainer.get_remain_returns(outputs['actions'], args.benchmark)
                    hpwl, cost = outputs_all["hpwl"], outputs_all["cost"]
                    print("hpwl = {:.2f}, cost = {:.2f}".format(hpwl, cost))
                    if hpwl < global_best_hpwl:
                        global_best_reward = -outputs_all["reward"] # -outputs['reward']
                        global_best_raw_reward = -outputs["reward"]
                        global_best_score = outputs['score']
                        global_best_hpwl = hpwl
                        global_best_cost = cost
                        fwrite.write("{:.2f}\t{:.2f}\t{}\t{}\t{:.4f}\n".format(global_best_hpwl, 
                            global_best_cost, global_best_reward, total_traj_cnt,
                            time.time() - start_time))
                        fwrite.flush()
                    last_update = total_traj_cnt + 1

                # writer.add_scalar('best_score', global_best_score, total_traj_cnt)
                # writer.add_scalar('best_reward', global_best_reward, total_traj_cnt)
                # writer.add_scalar('best_hpwl', global_best_hpwl, total_traj_cnt)
                # writer.add_scalar('best_cost', global_best_cost, total_traj_cnt)
                total_traj_cnt += 1
                if not args.is_fifo:
                    smallest, _ = heapq.nsmallest(1, replay_buffer.q)[0]
                    if outputs["score"] >= smallest:
                        break
                else:
                    break
            
            replay_buffer.add_new_trajs([outputs])
            add_new_traj_cnt += 1
        dataloader = create_dataloader(replay_buffer, args.batch_size)

        is_last_iter = online_iter == args.max_online_iters - 1
        if (online_iter + 1) % args.eval_interval == 0 or is_last_iter:
            evaluation = True
        else:
            evaluation = False
        
        if total_traj_cnt - last_update >= 32:
            th = max(0.5, min(1.0, th *1.25))
            last_update = total_traj_cnt
        
        if True:
            print("threshold = {}".format(th))
            train_outputs = online_trainer.train_iteration(
                dataloader=dataloader,
                threshold=th,
                traj_cnt = total_traj_cnt,
                decrease_entropy = decrease_entropy,
            )
            logs.update(train_outputs)
        
        # writer.add_scalar('loss', train_outputs["training/loss"], online_iter)
        # writer.add_scalar('loss_1', train_outputs["training/loss_1"], online_iter)
        # writer.add_scalar('acc', train_outputs["training/acc"], online_iter)
        # writer.add_scalar('entropy', train_outputs["training/entropy"], online_iter)
        # writer.add_scalar('threshold', th, online_iter)
        # if decrease_entropy:
        #     writer.add_scalar('decrease_entropy', 1, online_iter)
        # else:
        #     writer.add_scalar('decrease_entropy', 0, online_iter)

        logs["time/total"] = time.time() - start_time
        online_iter += 1


def get_model(model_path):
    mconf = GPTConfig(grid ** 2, args.traj_len * 3,
                  n_layer=6, n_head=8, n_embd=128, 
                  model_type=args.model_type, max_timestep=args.traj_len)
    model = GPT(mconf)
    if model_path is not None:
        state_dict = torch.load(model_path)
        for k,v in state_dict.items():
            if "module." in k:
                state_dict[k.split('.', 1)[1]] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    for k, v in model.named_parameters():
        if "head" or "one_kernel" in k:
            v.requires_grad = True
        else:
            v.requires_grad=False
    return model


def main():
    global global_best_reward
    global global_best_raw_reward
    global global_best_score
    global global_best_hpwl
    global global_best_cost
    global total_traj_cnt
    global model_path

    model = get_model(model_path)
    tconf = TrainerConfig(max_epochs=10, batch_size=64, learning_rate=1e-4,
                      lr_decay=True, warmup_tokens=512*20, 
                      final_tokens=2e10,
                      num_workers=4, seed=42, model_type="reward_conditioned",
                      max_timestep=args.traj_len)
    pretrain_trainer = Trainer(model, None, None, tconf)


    replay_buffer_path = None 
    if replay_buffer_path is not None:
        assert args.benchmark in replay_buffer_path

    dataset = []
    outputs_list = []
    replay_buffer = ReplayBuffer(args.replay_size, None, 
        is_unknown = True, seq_len = args.traj_len, is_fifo = args.is_fifo)
    total_traj_cnt = 0
    start_time = time.time()
    if replay_buffer_path is not None:
        outputs_list = pickle.load(open(replay_buffer_path, "rb"))
        for outputs in outputs_list[:args.replay_size]:
            replay_buffer.add_new_trajs([outputs])
            # writer.add_scalar('score', outputs['score'], total_traj_cnt)
            # writer.add_scalar('reward', outputs['reward'], total_traj_cnt)
            flog.write("{:.2f}\t{}\t{:.4f}\n".format(-outputs['reward'], total_traj_cnt, time.time() - start_time))
            flog.flush()
            if -outputs['reward'] * 0.98 < global_best_raw_reward:
                outputs_all = pretrain_trainer.get_remain_returns(outputs['actions'], args.benchmark)
                hpwl, cost = outputs_all["hpwl"], outputs_all["cost"]
                if hpwl < global_best_hpwl:
                    global_best_raw_reward = -outputs['reward']
                    global_best_reward = -outputs_all['reward']
                    global_best_score = outputs['score']
                    global_best_hpwl = hpwl
                    global_best_cost = cost
                    fwrite.write("{:.2f}\t{:.2f}\t{}\t{}\t{:.4f}\n".format(global_best_hpwl, 
                        global_best_cost, global_best_reward, total_traj_cnt,
                        time.time() - start_time))
                    fwrite.flush()

            # writer.add_scalar('best_score', global_best_score, total_traj_cnt)
            # writer.add_scalar('best_reward', global_best_reward, total_traj_cnt)
            # writer.add_scalar('best_hpwl', global_best_hpwl, total_traj_cnt)
            # writer.add_scalar('best_cost', global_best_cost, total_traj_cnt)
            total_traj_cnt += 1

    else:
        for i in range(args.replay_size):
            outputs = pretrain_trainer.get_returns(args.exploration_rtg, 
                is_single = True, benchmark = args.benchmark, is_all_macro = False)
            outputs_list.extend([outputs])
            replay_buffer.add_new_trajs([outputs])
            flog.write("{:.2f}\t{}\t{:.4f}\n".format(-outputs['reward'], total_traj_cnt, time.time() - start_time))
            flog.flush()
            if -outputs['reward'] * 0.98 < global_best_raw_reward:
                outputs_all = pretrain_trainer.get_remain_returns(outputs['actions'], args.benchmark)
                hpwl, cost = outputs_all["hpwl"], outputs_all["cost"]
                print("hpwl = {}, cost = {}".format(hpwl, cost))
                if hpwl < global_best_hpwl:
                    global_best_raw_reward = -outputs['reward']
                    global_best_reward = -outputs_all['reward']
                    global_best_score = outputs['score']
                    global_best_hpwl = hpwl
                    global_best_cost = cost
                    fwrite.write("{:.2f}\t{:.2f}\t{}\t{}\t{:.4f}\n".format(global_best_hpwl, 
                        global_best_cost, global_best_reward, total_traj_cnt,
                        time.time() - start_time))
                    fwrite.flush()
            # writer.add_scalar('best_score', global_best_score, total_traj_cnt)
            # writer.add_scalar('best_reward', global_best_reward, total_traj_cnt)
            # writer.add_scalar('best_hpwl', global_best_hpwl, total_traj_cnt)
            # writer.add_scalar('best_cost', global_best_cost, total_traj_cnt)
            total_traj_cnt += 1


    print("before train: score_avg = {:.4f}, worst_score = {}".format(replay_buffer.score_sum/len(replay_buffer),
            replay_buffer.worst_score))
    print("best wirelength = {:.2f}".format(global_best_reward))
    work(model, pretrain_trainer, replay_buffer, tconf)


if __name__ == "__main__":
    main()



    
    
        
    
    

    

