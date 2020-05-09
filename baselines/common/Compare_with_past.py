import numpy as np
import tensorflow as tf
import random
from gym import spaces
from baselines.a2c.utils import discount_with_dones,discount_with_dones_equal
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
#from baselines.a2c.policies import nature_cnn
from baselines.common.input import observation_input
class ReplayBuffer(object):
    def __init__(self, size):
        
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, R):#add replay buffer
        data = (obs_t, action, R)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, returns= [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            returns.append(R)
        return np.array(obses_t), np.array(actions), np.array(returns)

    def sample(self, batch_size):
        
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma
    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        
        self.var = np.ones(shape, 'float64')
        self.count = epsilon


    def update(self, x):

        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class compare_with_past(object):
    
    def __init__(self, fn_reward=np.sign, fn_obs=None,
            n_env=32, batch_size=128, n_update=4, 
            max_steps=int(1e5), gamma=0.99, stack=1):#fn_reward ==np.sign

        self.obsbuffer=RunningMeanStd()
        self.rewbuffer=RunningMeanStd()
        self.rff_int=RewardForwardFilter(gamma)
        #self.buf_rews_train()

        self.fn_reward = fn_reward
        self.fn_obs = fn_obs
        self.net={}
        self.buffer = ReplayBuffer(max_steps)
        self.buffer_negative = ReplayBuffer(max_steps)
        self.n_env = n_env
        self.batch_size = batch_size
        self.n_update = n_update

        self.max_steps = max_steps
        self.gamma = gamma
        self.stack = stack
        self.train_count = 0
        self.update_count = 0
        self.total_steps = []
        self.total_steps_negative = []
        self.total_rewards = []
        self.running_episodes = [[] for _ in range(n_env)]
        #self.buf_rews_int(sample_batch(self, batch_size)[0],sample_batch(self, batch_size)[1]):
        #self.buf_rews_train()

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []
        obsave=[]
        rsave=[]

        if self.stack > 1:
            ob_shape = list(trajectory[0][0].shape)
            nc = ob_shape[-1]
            ob_shape[-1] = nc*self.stack
            stacked_ob = np.zeros(ob_shape, dtype=trajectory[0][0].dtype)
        for (ob, action, reward) in trajectory:
            if ob is not None:
                x = self.fn_obs(ob) if self.fn_obs is not None else ob
                if self.stack > 1:
                    stacked_ob = np.roll(stacked_ob, shift=-nc, axis=2)
                    stacked_ob[:, :, -nc:] = x
                    obs.append(stacked_ob)
                else:
                    obs.append(x)
                actions.append(action)
                rewards.append(self.fn_reward(reward))
                dones.append(False)
        dones[len(dones)-1]=True
        returns = discount_with_dones_equal(rewards, dones)
        #obsofrew=np.zeros((len(dones),1))
        #for ik in range(int(len(dones)/2048)+1):
        #    if ik<int(len(dones)/2048):
        #        obsofrew[ik*2048:ik*2048+2048,0]=np.array(self.sess.run(self.int_rew,feed_dict={self.obss:np.array(obs[ik*2048:ik*2048+2048])}))
        #    else:
        #        if ik*2048<len(dones):
          
        #            obsofrew[ik*2048:len(dones),0]=np.array(self.sess.run(self.int_rew,feed_dict={self.obss:np.array(obs[ik*2048:len(dones)])}))
        
        #obssortrew=np.sort(obsofrew,0)
        #obla=obsofrew.shape[0]
        #obmax025=obssortrew[int(0.75*obla)]
        #i=0
        #rmax=0
        for (ob, action, R) in list(zip(obs, actions, returns)):
            #if obmax025<obsofrew[i] and R -rmax>0.1:
            #    self.buffer.add(ob,action, R)
            #rmax=R
            #i=i+1
            self.buffer.add(ob,action, R)

    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r) in trajectory:
            if r >0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.max_steps and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)


    def num_steps(self):
        return len(self.buffer)

    def num_episodes(self):
        return len(self.total_rewards)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def step(self, obs, actions, rewards, dones):
        for n in range(self.n_env):
            if self.n_update > 0:
                self.running_episodes[n].append([obs[n], actions[n], rewards[n]])
            else:
                self.running_episodes[n].append([None, actions[n], rewards[n]])

        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    def sample_batch(self, batch_size):
        if len(self.buffer) > 0:
            obs,action,reward=self.buffer.sample(batch_size)
            return obs,reward.reshape((batch_size,1))
        else:
            return None,None


