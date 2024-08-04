from fetch import fetch_data_from_database


class Strategy:
    def __init__(self, name, effect):
        self.name = name
        self.effect = effect

    def apply(self, value):
        return value + self.effect


weights_a = fetch_data_from_database(' -20~-10')
weights_b = fetch_data_from_database(' -10~-5')
weights_c = fetch_data_from_database(' -5~0')
weights_d = fetch_data_from_database('0~5')
weights_e = fetch_data_from_database('5~10')
weights_f = fetch_data_from_database('10~20')

strategies_a, strategies_b, strategies_c, strategies_d, strategies_e, strategies_f = [], [], [], [], [], []

for i in range(len(weights_a)):
    strategies_a.append(Strategy(f"A{i + 1}", weights_a[i]))

for i in range(len(weights_b)):
    strategies_b.append(Strategy(f"B{i + 1}", weights_b[i]))

for i in range(len(weights_c)):
    strategies_c.append(Strategy(f"C{i + 1}", weights_c[i]))

for i in range(len(weights_d)):
    strategies_d.append(Strategy(f"D{i + 1}", - weights_d[i]))

for i in range(len(weights_e)):
    strategies_e.append(Strategy(f"E{i + 1}", - weights_e[i]))

for i in range(len(weights_f)):
    strategies_f.append(Strategy(f"F{i + 1}", - weights_f[i]))


class StrategyLibrary:

    def __init__(self):
        # 定义不同偏移区间的策略
        self.strategies = {
            "[-20%, -10%)": strategies_a,
            "[-10%, -5%)": strategies_b,
            "[-5%, 0%)": strategies_c,
            "[0%, 5%)": strategies_d,
            "[5%, 10%)": strategies_e,
            "[10%, 20%)": strategies_f
        }

    def get_strategies(self, value):
        # if -50 <= value < -20:
        #     return self.strategies["[-50%, -20%)"]
        if -20 <= value < -10:
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
        # elif 20 <= value < 50:
        #     return self.strategies["[20%, 50%)"]
        else:
            return []


# 测试策略库
if __name__ == "__main__":
    library = StrategyLibrary()
    for i in range(0, len(library.strategies["[-20%, -10%)"])):
        print(library.strategies["[-20%, -10%)"][i].effect)
    print()
    for i in range(0, len(library.strategies["[-10%, -5%)"])):
        print(library.strategies["[-10%, -5%)"][i].effect)
    print()
    for i in range(0, len(library.strategies["[-5%, 0%)"])):
        print(library.strategies["[-5%, 0%)"][i].effect)
    print()
    for i in range(0, len(library.strategies["[0%, 5%)"])):
        print(library.strategies["[0%, 5%)"][i].effect)
    print()
    for i in range(0, len(library.strategies["[5%, 10%)"])):
        print(library.strategies["[5%, 10%)"][i].effect)
    print()
    for i in range(0, len(library.strategies["[10%, 20%)"])):
        print(library.strategies["[10%, 20%)"][i].effect)

    """
    A1 3.451007436455655
    A2 3.312791614171911
    A3 3.236200949372433
    
    B1 1.6374408645720582
    B2 1.6925993437760911
    B3 1.669959791651851
    
    C1 1.732527479252675
    C2 1.6401014345897753
    C3 1.6273710861575497
    
    D1 -1.9305526789759806
    D2 -1.648771663546766
    D3 -1.4206756574772537
    
    E1 -1.8352073530372375
    E2 -1.601480007831021
    E3 -1.5633126391317416
    
    F1 -3.311735451286095
    F2 -3.3575547599097804
    F3 -3.3307097888041244
    """
    # test_values = [-15, -7, -2, 3, 7, 15, 25]
    #
    # for value in test_values:
    #     strategies = library.get_strategies(value)
    #     print(f"Value: {value}")
    #     for strategy in strategies:
    #         new_value = strategy.apply(value)
    #         print(f"  Strategy: {strategy.name}, Effect: {strategy.effect}, New Value: {new_value}")
