import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from newenv import AnomalyMetricEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128).to(device)
        self.fc2 = nn.Linear(128, 256).to(device)
        self.fc3 = nn.Linear(256, 128).to(device)
        self.fc4 = nn.Linear(128, action_dim).to(device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128).to(device)
        self.fc2 = nn.Linear(128, 256).to(device)
        self.fc3 = nn.Linear(256, 128).to(device)
        self.fc4 = nn.Linear(128, 1).to(device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ActorCriticAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001, weight_decay=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001, weight_decay=0.01)
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        probs = self.actor(state)
        action = np.random.choice(self.action_dim, p=probs.cpu().detach().numpy())
        return action

    def train(self, num_episodes=5000, gamma=0.98):
        initial_states = []  # 存储每集开始时的状态
        strategy_sequences = []  # 存储每集中采取的所有策略序列
        final_states = []  # 存储每集结束时的状态

        for episode in range(num_episodes):
            state = self.env.reset()  # 重置环境并获取初始状态
            episode_reward = 0
            episode_strategies = []

            initial_states.append(state)  # 将初始状态存储到列表中

            while True:
                # 选择动作并执行，得到下一个状态和奖励
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward  # 更新总奖励

                # 将状态、下一个状态和奖励转换为张量
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                # 计算价值函数
                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)

                # 计算目标值（TD Target），使用Bellman方程
                target = reward_tensor + (1 - done) * gamma * next_value
                target = target.detach()  # 避免计算图的梯度传播
                current_value = self.critic(state_tensor)

                # 平滑更新目标值，使学习更稳定
                smoothed_target = current_value + (target - current_value) * 0.1

                # 计算critic的损失，即预测值与目标值之间的差的平方
                advantage = smoothed_target - value
                critic_loss = advantage.pow(2).mean()

                # 更新critic网络
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)  # 梯度裁剪，避免梯度爆炸
                self.critic_optimizer.step()

                # 计算actor的损失，即策略的对数概率乘以优势
                log_prob = torch.log(self.actor(state_tensor)[action])
                actor_loss = -log_prob * advantage.detach()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 如果集结束，记录相关信息并跳出循环
                if done:
                    episode_strategies.append(self.env.selected_strategies[-1])
                    strategy_sequences.append(episode_strategies)
                    final_states.append(next_state)
                    print(f"Episode {episode + 1}, "
                          f"初始偏移量: {initial_states[episode]}, "
                          f"最终值: {final_states[episode]}, "
                          f"采取的策略: {strategy_sequences[episode]}")
                    break
                state = next_state  # 更新当前状态为下一个状态
                episode_strategies.append(self.env.selected_strategies[-1])  # 添加采取的策略到列表

            # 调整学习率
            self.actor_scheduler.step()
            self.critic_scheduler.step()


if __name__ == "__main__":
    env = AnomalyMetricEnv(normal_value=0, max_deviation=50, stability=0.2)
    agent = ActorCriticAgent(env)
    agent.train()
