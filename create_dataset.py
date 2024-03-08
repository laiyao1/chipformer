import os
import numpy as np
import pickle
import time

from mingpt.trainer_placement import Args, Env, benchmark_to_id,\
                                        max_range, min_range

grid = 84
seq_len = 256
is_half = False

# make sure the dataset is in the root path
pkl_file_name_list = [
    # 'adaptec1_small.pkl',
    'adaptec2_small.pkl',
    # 'adaptec3_small.pkl',
    # 'adaptec4_small.pkl',
    # 'bigblue1_small.pkl',
    # 'bigblue2_small.pkl',
    # 'bigblue3_small.pkl',
    # 'bigblue4_small.pkl',
]

# set the upbound volume of the dataset
num_of_dataset = 25 * len(pkl_file_name_list)

def init_dataset(record):
    dataset = {}
    dataset['meta_observations'] = np.zeros((record, 6), float) 
    dataset['observations'] = np.zeros((record, grid, grid), bool) 
    dataset['obs_wire_mask'] = np.zeros((record, grid, grid), float)
    dataset['obs_pos_mask'] = np.zeros((record, grid, grid), bool)

    dataset['actions'] = np.zeros((record, 1), int)
    dataset['rewards'] = np.zeros((record, 1), float)
    dataset['raw_rewards'] = np.zeros((record, 1), int)
    dataset['terminals'] = np.zeros((record, 1), bool)
    dataset['benchmarks'] = np.zeros((record, 1), int)
    dataset['levels'] = np.zeros((record, 1), int)
    dataset['lengths'] = np.zeros((record, 1), int)
    return dataset


def get_ideal_rtg(obss_wire, max_reward):
    reward = 0
    for i in range(obss_wire.shape[0]):
        if obss_wire[i].max() > max_reward * 0.3:
            reward += 1
        elif obss_wire[i].max() > max_reward * 0.1:
            reward += 1
        elif obss_wire[i].max() > 0:
            reward += 1
    return reward


def create_dataset(num_buffers=0, num_steps=0, game=0, 
    data_dir_prefix=0, trajectories_per_buffer=0, is_eval_only = False):
    # -- load data from memory (make more efficient)
    global num_of_dataset
    if is_eval_only:
        num_of_dataset = 1*10
    print("start create_dataset.")
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []
    cnt= 0
    record = 0

    medians = {}
    medians = {0: -5659.5,  1: -41702.0, 2: -22239.5, 3: -11906.0, 4: -1797.0, 5: -2637.0, 
    6: -10915.0, 7: -19080.0, 8: -14187.0, 9: -13650.5, 
    10: -4541.5, 11: -11342.5, 12: -1206.0, 13: -1076.0}
    
    print("medians", medians)
    if not is_eval_only:
        record = len(pkl_file_name_list) * 500 * seq_len
    else:
        record = len(pkl_file_name_list) * 10 * seq_len
    pickle.dump(medians, open("dataset.pkl", 'wb'))
    print("record = {}".format(record))
    assert record % 2 ==0
    if is_half:
        dataset = init_dataset(record//2)
    else:
        dataset = init_dataset(record)

    cnt= 0
    start_rec = 0
    if not is_eval_only or len(pkl_file_name_list) == 1:
        for pkl_file_name in pkl_file_name_list:
            with open(os.path.join(".", pkl_file_name), "rb") as f:
                print("pkl_file_name", pkl_file_name)
                sub_cnt = 0
                benchmark = pkl_file_name.split('_')[0]
                benchmark_id = benchmark_to_id[benchmark]
                record_cnt = 0
                while True:
                    try:
                        tmp_data = pickle.load(f)
                        for key in tmp_data.keys():
                            if key.startswith("obs"):
                                tmp_data[key] = tmp_data[key].reshape(-1, seq_len, grid, grid)
                            elif key.startswith("meta"):
                                tmp_data[key] = tmp_data[key].reshape(-1, seq_len, 6)
                            else:
                                tmp_data[key] = tmp_data[key].reshape(-1, seq_len, 1)

                        for i in range(tmp_data["rewards"].shape[0]):
                            if (is_half and np.sum(tmp_data['rewards'][i]) >= medians[benchmark_id] and \
                                record_cnt < record/(len(pkl_file_name_list) * 2 * seq_len)) or (not is_half):
                                for key in dataset.keys():
                                    if key != "benchmarks" and key != "levels" and key != "rewards" and key != "raw_rewards":
                                        dataset[key][start_rec: start_rec+seq_len] = tmp_data[key][i]
                                    else:
                                        if key == "benchmarks":
                                            dataset['benchmarks'][start_rec: start_rec+seq_len] = benchmark_id
                                            dataset['rewards'][start_rec: start_rec+seq_len] = \
                                                tmp_data['rewards'][i] / (max_range[benchmark_id] - min_range[benchmark_id]) + \
                                                max_range[benchmark_id]/((max_range[benchmark_id] - min_range[benchmark_id]) * seq_len)
                                            dataset['raw_rewards'][start_rec: start_rec+seq_len] = tmp_data['rewards'][i]
                                        elif key == "levels":
                                            dataset['levels'][start_rec: start_rec+seq_len] = 0
                                        elif key == "rewards":
                                            pass
                                        else:
                                            pass
                                start_rec += seq_len
                                record_cnt += 1
                                if record_cnt >= record/seq_len:
                                    break
                        sub_cnt += 1
                        cnt += 1
                        if sub_cnt >= num_of_dataset/len(pkl_file_name_list) or cnt >= num_of_dataset or record_cnt >= record/seq_len:
                            break
                    except EOFError:
                        print("eoferror")
                        break
                print("record_cnt = {}".format(record_cnt))
                print("expected cnt = {}".format(record/(len(pkl_file_name_list) * 2 * seq_len)))

    print("load finished")
    # further to concise
    obss = dataset['observations'].reshape(-1, 1, grid, grid)
    obss_wire = dataset['obs_wire_mask'].reshape(-1, 1, grid, grid)
    obss_mask = dataset['obs_pos_mask'].reshape(-1, 1, grid, grid)
    meta_data = dataset['meta_observations']
    actions = dataset['actions']
    levels = dataset['levels']
    benchmarks = dataset['benchmarks']
    lengths = dataset['lengths']
    
    stepwise_returns = dataset['rewards'] # / 1000.0
    stepwise_raw_returns = dataset['raw_rewards']

    print("stepwise_returns", stepwise_returns)
    print("stepwise_returns shape", stepwise_returns.shape)
    print("stepwise_returns max", stepwise_returns.max())
    print("stepwise_returns min", stepwise_returns.min())

    precentiles = [0] * dataset['actions'].shape[0]
    ranges = [0] * dataset['actions'].shape[0]
    deltas = [0] * dataset['actions'].shape[0]
    minimals = [0] * dataset['actions'].shape[0]

    for i in range(dataset['actions'].shape[0]):
        max_obss_wire = obss_wire[i, 0].max() * 12800
        min_obss_wire = obss_wire[i, 0].min() * 12800
        step_return = -stepwise_returns[i, 0]
        if max_obss_wire == min_obss_wire:
            precentile = 0
        else:
            precentile = (step_return - min_obss_wire) / (max_obss_wire - min_obss_wire)
        precentiles[i] = precentile
        ranges[i] = max_obss_wire - min_obss_wire
        deltas[i] = step_return - min_obss_wire
        minimals[i] = min_obss_wire

    for i in range(len(dataset['actions'])):
        if dataset['terminals'][i]:
            done_idxs += [i+1] # special design, each episode [:done_idxs]
            returns += [0]
        returns[-1] += dataset['rewards'][i]

    actions = np.array(actions)
    print("obss_wire shape", obss_wire.shape)
    actions = actions.reshape(-1, seq_len)
    
    returns = np.array(returns)
    benchmarks = np.array(benchmarks)
    levels = np.array(levels)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)
    lengths = np.array(lengths)
    actions = actions.flatten()

    rtg_statistics = [[], [], [], []]
    rtg_raw_statistics = [[], [], [], []]
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    print("done_idxs[:10]", done_idxs[:10])
    print("stepwise_returns [0]", stepwise_returns[0])
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        raw_curr_traj_returns = stepwise_raw_returns[start_index:i]
        rtg_raw_statistics[levels[start_index, 0]].append(int(sum(raw_curr_traj_returns)))
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        rtg_statistics[levels[start_index, 0]].append(rtg[start_index, 0])
        start_index = i

    rtg = rtg.flatten()
    print("len rtg", len(rtg))
    print('max rtg is %d' % max(rtg))
    print("rtg", rtg)

    timesteps = np.zeros(len(actions), dtype=int)
    start_index = 0
    for i in done_idxs:
        if start_index >= dataset["actions"].shape[0]:
            break
        i = int(i)
        timesteps[start_index:i] = np.arange(i - start_index)
        start_index = i
    print('max timestep is %d' % max(timesteps))
    print("timesteps", timesteps)
    print("meta_data shape", meta_data.shape)

    return obss, actions, returns, done_idxs, rtg, timesteps, \
        meta_data, obss_wire, obss_mask, benchmarks, stepwise_returns, lengths


def test_returns(actions, targets = None, obss = None, obss_wire = None, 
    obss_mask = None, meta_observations = None, benchmark = 'adaptec1'):
    args=Args()
    env = Env(args, benchmark)
    print("actions", actions)
    done = True
    state, reward, done, meta_state = env.reset()
    reward_sum = 0
    print("state shape", state.shape)
    print("meta state", meta_state)
    print("state[1] sum", state[1].sum())
    state = state
    meta_state = meta_state
    correct = None
    if targets is not None:
        correct = (actions == targets)
    j = 0
    print("actions shape", actions.shape)
    for i in range(actions.shape[0]):
        if done:
            break
        action = actions[i]
        state, reward, done, meta_state = env.step(action)
        reward_sum += reward
        j += 1
        if done:
            break

    print("reward_sum = {}".format(reward_sum))


if __name__ == "__main__":
    obss, actions, returns, done_idxs, rtg, \
        timesteps, meta_data, obss_wire, obss_mask, \
        benchmarks, stepwise_returns, lengths = create_dataset(is_eval_only = False)
