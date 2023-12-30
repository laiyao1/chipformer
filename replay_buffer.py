import numpy as np
import torch
import heapq


from mingpt.trainer_placement import circuit_feas

circuit_hpwl = [[5000, 25000], [65000, 180000], [25000, 120000], [10000, 70000],
                [1400, 14000], [3400, 27000], [23000, 125000], [18000, 160000]]

# is_fifo = False

def transform_data(single_data, seq_len):
    outputs = {}
    if "obss" in single_data:
        outputs["states"] = torch.tensor(np.array(single_data["obss"]), 
            dtype=torch.float32).reshape(seq_len, -1) # (block_size, 4*84*84)
    else:
        tmp_obss = torch.tensor(np.array(single_data["observations"]), 
            dtype=torch.float32).reshape(seq_len, -1)
        tmp_obss_wire = torch.tensor(np.array(single_data["obs_wire_mask"]), 
            dtype=torch.float32).reshape(seq_len, -1)
        tmp_obss_mask = torch.tensor(np.array(single_data["obs_pos_mask"]), 
            dtype=torch.float32).reshape(seq_len, -1)
        outputs['states'] = torch.cat((tmp_obss, tmp_obss_wire, tmp_obss_mask), dim=1)

    outputs["meta_states"] = torch.tensor(np.array(single_data["meta_observations"]), dtype=torch.float32).reshape(seq_len, -1)
    # states is 3*224*224 image do not need to standard
    outputs["actions"] = torch.tensor(np.array(single_data["actions"]), dtype=torch.long).reshape(seq_len, 1) # (block_size, 1)
    outputs["rewards"] = torch.tensor(np.array(single_data["rewards"]), dtype=torch.int64).reshape(seq_len, 1)
    outputs["rtgs"] = torch.tensor(np.array(single_data["rtgs"]), dtype=torch.float32).reshape(seq_len, 1)
    outputs["timesteps"] = torch.tensor(np.array(single_data["timesteps"]), dtype=torch.int64).reshape(seq_len, 1)
    outputs["benchmarks"] = torch.tensor([int(single_data["benchmarks"][0])], dtype=torch.int32).reshape(1)
    outputs['score'] = float(outputs["rtgs"][0, 0])
    outputs["lengths"] =  torch.tensor(np.array(single_data["lengths"]), dtype=torch.int32).reshape(seq_len, 1)
    benchmark_id = int(single_data["benchmarks"][0])
    outputs['circuit_feas'] = torch.tensor(circuit_feas[benchmark_id], dtype = torch.float32).reshape(-1)

    return outputs


class ReplayBuffer(object):
    def __init__(self, capacity, dataset, is_unknown = False, seq_len = 255, is_fifo = False):
        self.capacity = capacity
        self.seq_len = seq_len
        self.trajectories = []
        self.score_sum = 0.0
        self.worst_score = 100.0
        self.q = []
        self.is_fifo = is_fifo
        if not is_unknown:
            for i in range(self.capacity-1, -1, -1):
                single_data = {}
                start_rec = i * seq_len
                idx = start_rec
                for key in dataset:
                    single_data[key] = dataset[key][start_rec:start_rec+seq_len]
                outputs = transform_data(single_data, seq_len)
                self.score_sum += outputs["score"]
                self.trajectories.append(outputs)
                if not self.is_fifo:
                    heapq.heappush(self.q, (outputs["score"], len(self.trajectories)-1))
                self.worst_score = min(self.worst_score, outputs["score"])
        else:
            pass

        if self.is_fifo:
            self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        print("len(self.trajectories) = {}, self.capacity = {}".format(len(self.trajectories),
            self.capacity))
        if len(self.trajectories) < self.capacity:

            self.trajectories.extend(new_trajs)
            if not self.is_fifo:
                heapq.heappush(self.q, (new_trajs[0]["score"], len(self.trajectories)-1))
            self.score_sum += new_trajs[0]["score"]
        else:
            # Circuilation queue
            assert len(new_trajs) == 1
            if not self.is_fifo:
                old_score, start_idx  = heapq.heappop(self.q)
                self.trajectories[start_idx] = new_trajs[0]
                heapq.heappush(self.q, (new_trajs[0]["score"], start_idx))
            else:
                old_score = self.trajectories[self.start_idx]["score"]
                self.trajectories[self.start_idx] = new_trajs[0]
                self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

            self.score_sum = self.score_sum - old_score + new_trajs[0]["score"]

        assert len(self.trajectories) <= self.capacity
