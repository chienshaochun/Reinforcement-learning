"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)
from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        # record per-epoch loss
        self.loss_history = []
        self.tokens = 0

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if self.config.ckpt_path:
            logger.info("saving model to %s", self.config.ckpt_path)
            torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
            losses = []
            for it, (x, y, r, t) in (tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)):
                x, y, r, t = x.to(self.device), y.to(self.device), r.to(self.device), t.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()
                    losses.append(loss.item())
                if is_train:
                    model.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    if config.lr_decay:
                        self.tokens += (y>=0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens)/max(1, config.warmup_tokens)
                        else:
                            progress = float(self.tokens - config.warmup_tokens)/max(1, config.final_tokens-config.warmup_tokens)
                            lr_mult = max(0.1, 0.5*(1.0+math.cos(math.pi*progress)))
                        for pg in optimizer.param_groups: pg['lr'] = config.learning_rate * lr_mult
                    else:
                        lr_mult = 1.0
                    if is_train:
                        tqdm.write(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}, lr_mult {lr_mult:.4f}")
            return float(np.mean(losses)) if losses else float('nan')

        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train', epoch)
            self.loss_history.append(train_loss)
            # optional evaluation code can go here
            # save checkpoint per epoch if desired
            self.save_checkpoint()
        # save loss history
        try:
            np.save('loss_history.npy', np.array(self.loss_history))
            logger.info("Saved loss history to loss_history.npy")
        except Exception as e:
            logger.warning(f"Failed saving loss history: {e}")

    def get_returns(self, ret):  # unchanged
        self.model.train(False)
        args = Args(self.config.game.lower(), self.config.seed)
        env = Env(args); env.eval()
        T_rewards = []
        done = True
        for _ in range(10):
            state = env.reset().type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            sampled_action = sample(self.model.module, state, 1,
                                    temperature=1.0, sample=True,
                                    actions=None,
                                    rtgs=torch.tensor(rtgs, dtype=torch.long)
                                        .to(self.device).unsqueeze(0).unsqueeze(-1),
                                    timesteps=torch.zeros((1,1,1), dtype=torch.int64)
                                        .to(self.device))
            j, reward_sum = 0, 0
            all_states, actions = state, []
            while True:
                if done: state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions.append(sampled_action)
                state, reward, done = env.step(action); reward_sum += reward; j+=1
                if done: T_rewards.append(reward_sum); break
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                all_states = torch.cat([all_states, state], dim=0)
                rtgs.append(rtgs[-1]-reward)
                sampled_action = sample(self.model.module, all_states.unsqueeze(0),1,
                                      temperature=1.0, sample=True,
                                      actions=torch.tensor(actions, dtype=torch.long)
                                          .to(self.device).unsqueeze(1).unsqueeze(0),
                                      rtgs=torch.tensor(rtgs, dtype=torch.long)
                                          .to(self.device).unsqueeze(0).unsqueeze(-1),
                                      timesteps=(min(j, self.config.max_timestep)
                                                 *torch.ones((1,1,1),dtype=torch.int64)
                                              .to(self.device)))
        env.close(); eval_return = sum(T_rewards)/10.
        print(f"target return: {ret}, eval return: {eval_return}")
        self.model.train(True)
        return eval_return

# Env and Args classes unchanged below...
class Env: pass
class Args: pass
