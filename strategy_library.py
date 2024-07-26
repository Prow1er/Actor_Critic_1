from judge import Judge


# 定义策略类
class Strategy:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect

    def apply(self, value):
        return value + self.effect


class StrategyLibrary:
    def __init__(self):
        # 定义不同偏移区间的策略
        self.strategies = {
            "[-50%, -20%)": [
                Strategy("A2", 20),
                Strategy("A3", 15),
            ],
            "[-20%, -10%)": [
                Strategy("B1", 10),
                Strategy("B2", 5),
            ],
            "[-10%, -5%)": [
                Strategy("C1", 3),
                Strategy("C2", 2),
            ],
            "[-5%, 0%)": [
                Strategy("D1", 1),
                Strategy("D2", 0.5),
                Strategy("D3", 0.1),
                Strategy("D4", 0.05),
            ],
            "[0%, 5%)": [
                Strategy("E1", -0.05),
                Strategy("E2", -0.1),
                Strategy("E3", -0.5),
                Strategy("E4", -1),
            ],
            "[5%, 10%)": [
                Strategy("F1", -2),
                Strategy("F2", -3),
            ],
            "[10%, 20%)": [
                Strategy("G1", -5),
                Strategy("G2", -10),
            ],
            "[20%, 50%)": [
                Strategy("H1", -15),
                Strategy("H2", -20),
            ],
        }

    def get_strategies(self, value):
        if -50 <= value < -20:
            return self.strategies["[-50%, -20%)"]
        elif -20 <= value < -10:
            return self.strategies["[-20%, -10%)"]
        elif -10 <= value < -5:
            return self.strategies["[-10%, -5%)"]
        elif -5 <= value < 0:
            return self.strategies["[-5%, 0%)"]
        elif 0 <= value < 5:
            return self.strategies["[0%, 5%)"]
        elif 5 <= value < 10:
            return self.strategies["[5%, 10%)"]
        elif 10 <= value < 20:
            return self.strategies["[10%, 20%)"]
        elif 20 <= value < 50:
            return self.strategies["[20%, 50%)"]
        else:
            return []


# 测试策略库
if __name__ == "__main__":
    library = StrategyLibrary()
    test_values = [-45, -15, -7, -2, 3, 7, 15, 25]

    for value in test_values:
        strategies = library.get_strategies(value)
        print(f"Value: {value}")
        for strategy in strategies:
            new_value = strategy.apply(value)
            print(f"  Strategy: {strategy.name}, Effect: {strategy.effect}, New Value: {new_value}")
