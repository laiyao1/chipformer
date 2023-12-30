# online decision transformer trainer

import numpy as np
import torch
from torch.distributions import Categorical
import time
seq_len = 256

class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        self.softmax = torch.nn.Softmax(dim=-1)

    def train_iteration(
        self,
        dataloader,
        threshold,
        traj_cnt,
        decrease_entropy
    ):

        losses, losses_1, accs, entropies = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for i in range(10):
            for _, trajs in enumerate(dataloader):
                loss, loss_1, entropy, acc = self.train_step_stochastic(trajs, threshold, 
                    traj_cnt, decrease_entropy)
                print("loss = {:.2f}, loss_1 = {:.2f}, entropy = {:.4f}, acc = {:.4f}".format(loss, 
                    loss_1, entropy, acc))
                losses.append(loss)
                losses_1.append(loss_1)
                accs.append(acc)
                entropies.append(entropy)
            

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/loss"] = losses[-1]
        logs["training/acc"] = accs[-1]
        logs["training/loss_1"] = losses_1[-1]
        logs["training/entropy"] = entropies[-1]

        return logs

    def train_step_stochastic(self, trajs, threshold, traj_cnt, decrease_entropy):
        (
            states,
            meta_states,
            actions,
            rewards,
            rtgs,
            timesteps,
            benchmarks,
            circuit_feas,
            targets,
            probs,
            lengths,
        ) = trajs


        states = states.to(self.device)
        meta_states = meta_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        rtgs = rtgs.to(self.device)
        benchmarks = benchmarks.to(self.device)
        timesteps = timesteps.to(self.device)
        circuit_feas = circuit_feas.to(self.device)
        targets = targets.to(self.device)
        probs = probs.to(self.device)
        lengths = lengths.to(self.device)

        action_target = torch.clone(actions)

        logits, loss_1, acc = self.model.forward(
            states,
            actions,
            action_target,
            rtgs,
            timesteps,
            meta_states,
            benchmarks,
            stepwise_returns = None,
            circuit_feas = circuit_feas,
            lengths= lengths
        )
        
        loss_1 = loss_1.mean()
        acc = acc.mean()
        real_len = lengths[0].sum()
        entropy = Categorical(probs = self.softmax(logits[:, :real_len, :])).entropy()
        entropy = entropy.mean()

        loss = loss_1 + 0.5 * max(0, 0.5 - entropy)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            loss_1.detach().cpu().item(),
            # nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            acc.detach().cpu().item()
        )