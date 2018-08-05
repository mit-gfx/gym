import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import *
import IPython


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 1.0)') #defaults to not discounted; gamma = 1.0
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)') #Yo it's a seed
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('Spaceship-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 2)
        torch.nn.init.normal(self.affine1.weight)
        torch.nn.init.normal(self.affine1.bias)
        self.affine2 = nn.Linear(2, 2)
        torch.nn.init.normal(self.affine2.weight)
        torch.nn.init.normal(self.affine2.bias)

        self.saved_log_probs = []
        self.rewards = []
        self.step_rewards = []

    def forward(self, x):        
        x = F.tanh(self.affine1(x))
        action_scores = F.tanh(self.affine2(x))
        return action_scores
        
    def commit_rewards(self):
        self.rewards.append(np.mean(self.step_rewards))
        self.step_rewards = []
        



policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1.0e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Normal(probs[0][0], torch.abs(probs[0][1]))
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    #TODO: this only works because the action is 1-D
    return action.item()


def finish_episode(i):
    policy_loss = []
    rewards = []
    for r in policy.rewards:
        rewards.append(r)
    rewards = torch.tensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward.float())
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward(retain_graph=True, create_graph=False)
    optimizer.step()
    #let's just print out the norm of part of the paramters here, for simplicity:
    if i % 1000 == 0:
        print('loss is ', policy_loss)
        print('mean reward is ', torch.mean(rewards))
        print('some of the norm is')
        print(np.linalg.norm(list(policy.parameters())[1].grad.detach().numpy()))
    
    del policy.rewards[:]
    del policy.step_rewards[:]
    del policy.saved_log_probs[:]


def main():
    i = 0
    for i_episode in count(1):
        state = env.reset()        
        for t in range(10000):  # Don't infinite loop while learning
            i += 1
            #TODO: make sure episodes line up with rollouts or in some way flag that we're in the final step
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.step_rewards.append(reward)
            if args.render:
                env.render()
            if done:
                env.reset(random=True)
                policy.commit_rewards()
            

        #print(np.mean(policy.rewards))
        finish_episode(i)
        '''
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        '''


if __name__ == '__main__':
    main()
