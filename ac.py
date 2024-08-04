import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from newenv import AnomalyMetricEnv
from ac_net import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: 论文   元学习         博弈论、强化学习  结果分析


class ActorCriticAgent:
    """
    Actor-Critic强化学习算法的智能体类。

    参数:
    - env: 环境对象，用于与智能体交互。
    - meta_lr: 元学习率，用于更新元模型参数。
    - inner_lr: 内循环学习率，用于在每个episode中更新克隆模型参数。
    - num_inner_steps: 每个episode中内循环更新的次数。
    - scheduler_step_size: 学习率调度的步长。
    - scheduler_gamma: 学习率调度的乘数。
    """

    def __init__(self, env, meta_lr=0.001, inner_lr=0.01, num_inner_steps=1, scheduler_step_size=200,
                 scheduler_gamma=0.5):
        # 初始化环境和模型参数
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim).to(device)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        # 初始化优化器和学习率调度
        self.actor_meta_optimizer = optim.Adam(self.actor.parameters(), lr=self.meta_lr)
        self.critic_meta_optimizer = optim.Adam(self.critic.parameters(), lr=self.meta_lr)
        self.actor_scheduler = StepLR(self.actor_meta_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.critic_scheduler = StepLR(self.critic_meta_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def train(self, num_episodes=1000, gamma=0.98, num_training_attempts=5):
        """
        训练模型。

        参数:
        - num_episodes (int): 训练的回合数，默认为1000。
        - gamma (float): 折扣因子，默认为0.98。
        - num_training_attempts (int): 每个回合的训练尝试次数，默认为5。

        返回:
        无。
        """
        # 初始化存储每个回合的初始状态、策略序列和最终状态的列表
        initial_states = []
        strategy_sequences = []
        final_states = []

        # 遍历每个回合
        for episode in range(num_episodes):
            # 初始化当前最佳的回合奖励、初始状态、最终状态和策略序列
            best_episode_reward = -float('inf')
            best_initial_state = None
            best_final_state = None
            best_strategy_sequence = None

            # 针对每个回合进行多次训练尝试
            for attempt in range(num_training_attempts):
                # 重置环境并获取初始状态
                state = self.env.reset()
                episode_reward = 0
                episode_strategies = []
                # 将当前回合的初始状态添加到列表中
                initial_states.append(state)
                # 克隆演员和评论家模型
                actor_clone = self.clone_model(self.actor, self.state_dim, self.action_dim)
                critic_clone = self.clone_model(self.critic, self.state_dim)
                # 克隆模型的优化器
                actor_clone_optimizer = optim.SGD(actor_clone.parameters(), lr=self.inner_lr)
                critic_clone_optimizer = optim.SGD(critic_clone.parameters(), lr=self.inner_lr)

                # 进行内循环步数的学习
                for _ in range(self.num_inner_steps):
                    # 与环境交互直到回合结束
                    while True:
                        # 选择动作并执行，获取下一个状态和奖励
                        action = self.select_action(state, actor_clone)
                        next_state, reward, done, _ = self.env.step(action)
                        episode_reward += reward

                        # 将状态、下一个状态和奖励转换为张量
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
                        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                        # 计算当前状态和下一个状态的价值
                        value = critic_clone(state_tensor)
                        next_value = critic_clone(next_state_tensor)

                        # 计算目标值
                        target = reward_tensor + (1 - done) * gamma * next_value
                        target = target.detach()

                        # 平滑目标值
                        smoothed_target = value + (target - value) * 0.1

                        # 计算评论家损失并反向传播
                        critic_loss = (smoothed_target - value).pow(2).mean()
                        critic_clone_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic_clone.parameters(), max_norm=0.5)
                        critic_clone_optimizer.step()

                        # 计算日志概率和演员损失并反向传播
                        log_prob = torch.log(actor_clone(state_tensor)[action])
                        advantage = smoothed_target - value
                        actor_loss = -log_prob * advantage.detach()
                        actor_clone_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_clone_optimizer.step()

                        # 如果回合结束，则记录当前策略并更新最佳状态和策略
                        if done:
                            episode_strategies.append(self.env.selected_strategies[-1])
                            if episode_reward > best_episode_reward:
                                best_episode_reward = episode_reward
                                best_initial_state = initial_states[-1]
                                best_final_state = next_state
                                best_strategy_sequence = episode_strategies[:]
                            break

                        # 更新状态和策略
                        state = next_state
                        episode_strategies.append(self.env.selected_strategies[-1])

            # 将当前回合的最佳策略序列和最终状态添加到列表中
            strategy_sequences.append(best_strategy_sequence)
            final_states.append(best_final_state)
            # 打印当前回合的信息
            print(f"Episode {episode + 1}, "
                  f"初始偏移量: {best_initial_state}, "
                  f"最终值: {best_final_state}, "
                  f"采取的策略: {best_strategy_sequence}")

            # 更新演员和评论家元优化器
            actor_clone_optimizer.zero_grad()
            for param, clone_param in zip(self.actor.parameters(), actor_clone.parameters()):
                param.grad = (clone_param - param).detach()
            self.actor_meta_optimizer.step()

            critic_clone_optimizer.zero_grad()
            for param, clone_param in zip(self.critic.parameters(), critic_clone.parameters()):
                param.grad = (clone_param - param).detach()
            self.critic_meta_optimizer.step()

            # 更新学习率调度器
            self.actor_scheduler.step()
            self.critic_scheduler.step()

    def clone_model(self, model, state_dim=None, action_dim=None):
        # 根据模型类型创建并初始化克隆模型
        if isinstance(model, Actor):
            cloned_model = Actor(state_dim, action_dim).to(device)
        elif isinstance(model, Critic):
            cloned_model = Critic(state_dim).to(device)
        # 加载原模型的参数
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model

    def select_action(self, state, model):
        # 将状态转换为张量
        state = torch.tensor(state, dtype=torch.float32, device=device)
        # 计算状态对应的动作概率
        probs = model(state)
        # 根据概率选择动作
        action = np.random.choice(self.action_dim, p=probs.cpu().detach().numpy())
        return action

    def save_models(self, path):
        # 保存actor和critic模型的参数
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")

    def load_models(self, path):
        # 加载actor和critic模型的参数
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth"))


if __name__ == "__main__":
    # 初始化环境和智能体，并开始训练
    env = AnomalyMetricEnv(normal_value=0, max_deviation=20, stability=0.5)
    agent = ActorCriticAgent(env)
    agent.train()
