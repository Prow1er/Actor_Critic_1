import gym
from gym import spaces
import numpy as np
from strategy_library import StrategyLibrary


class AnomalyMetricEnv(gym.Env):
    def __init__(self, normal_value=0, max_deviation=50, stability = 0.2):
        super(AnomalyMetricEnv, self).__init__()

        self.normal_value = normal_value
        self.max_deviation = max_deviation
        self.state = None
        self.selected_strategies = []
        self.stability = stability

        # 初始化策略库
        self.strategy_library = StrategyLibrary()

        # 定义状态空间和动作空间
        self.observation_space = spaces.Box(low=-max_deviation, high=max_deviation, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self._get_all_strategies()))

    def _get_all_strategies(self):
        # 获取所有策略的列表
        all_strategies = []
        for key in self.strategy_library.strategies.keys():
            all_strategies.extend(self.strategy_library.strategies[key])
        return all_strategies

    def reset(self):
        # 初始化状态，随机生成一个偏离值
        self.state = np.random.uniform(-self.max_deviation, self.max_deviation)
        self.selected_strategies = []  # 重置所选策略列表
        return np.array([self.state], dtype=np.float32)

    def step(self, action):
        # 执行策略（动作），并更新状态
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        # 获取当前状态对应的策略列表
        current_strategies = self.strategy_library.get_strategies(self.state)

        if not current_strategies:
            raise ValueError(f"No strategies available for state {self.state}")

        # 应用选定的策略
        selected_strategy = current_strategies[action % len(current_strategies)]
        self.state = selected_strategy.apply(self.state)
        self.selected_strategies.append(selected_strategy.name)  # 记录所选策略

        # 计算奖励，目标是使状态接近正常值
        reward = -(100*(abs(self.state - self.normal_value)))

        # 检查是否达到终止条件
        done = abs(self.state - self.normal_value) < self.stability

        # 返回新的状态、奖励、是否终止和额外信息
        return np.array([self.state], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        # 可选的渲染方法
        print(f"Current state: {self.state}, Strategies: {self.selected_strategies}")

    def close(self):
        pass


# 测试自定义环境
if __name__ == "__main__":
    env = AnomalyMetricEnv()
    state = env.reset()
    print(f"Initial state: {state}")

    for _ in range(300):
        action = env.action_space.sample()  # 随机选择一个动作
        state, reward, done, _ = env.step(action)
        print(f"Reward: {reward} ")
        env.render()
        if done:
            print("Episode finished")
            break
