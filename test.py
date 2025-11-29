MAX_HAND_SIZE = 10 # 手牌数量上限
MAX_DECK_SIZE = 100 # 卡组最大数量, 理论上没有上限，但实际游戏中一般不会超过100张吧？或者超过100张就很少见了，或者干脆超过100张就进行惩罚
MAX_MONSTER_COUNT = 5 # 最多同时存在的怪物数量：蛇女+四个匕首 或者 五只小史莱姆。有待求证
MAX_POTION_COUNT = 5 # 最多携带5个药水

class ActionMapper:
    def __init__(self):
        """初始化动作映射器"""
        # 最大动作维度应该等于{"choose": 0, "return": 1, "play": 2, "end": 3, "proceed": 4, "skip": 5, "potion": 6, "leave": 7, "confirm": 8, "cancel": 9}
        # 以上这些动作加上后续可能的target加起来。
        # choose: 考虑到升级卡牌，所以最大值应该是卡组大小 MAX_DECK_SIZE
        # play: 最大值应该是手牌数量 MAX_HAND_SIZE 乘以 最大怪物数量 MAX_MONSTER_COUNT（考虑有目标和无目标两种情况）
        # potion: 最大值应该是药水数量 乘以 最大怪物数量 MAX_MONSTER_COUNT（考虑有目标和无目标两种情况）
        # 剩余其他动作都是单一动作
        self.max_choose_dim = MAX_DECK_SIZE
        print(f"最大选择动作维度: {self.max_choose_dim}")
        self.max_play_dim = MAX_HAND_SIZE * MAX_MONSTER_COUNT
        print(f"最大出牌动作维度: {self.max_play_dim}")
        self.max_potion_dim = MAX_POTION_COUNT + MAX_POTION_COUNT * MAX_MONSTER_COUNT
        print(f"最大药水动作维度: {self.max_potion_dim}")
        self.max_single_dim = 7  # 单一动作的数量
        print(f"最大单一动作维度: {self.max_single_dim}")

        self.max_action_dim = self.max_choose_dim + self.max_play_dim + self.max_potion_dim + self.max_single_dim
        print(f"总动作维度: {self.max_action_dim}")

        """根据动作类型，区分索引范围，方便后续映射"""
        self.choose_start = 0
        self.choose_end = self.choose_start + self.max_choose_dim - 1
        print(f"选择动作索引范围: {self.choose_start} - {self.choose_end}")
        self.play_start = self.choose_end + 1
        self.play_end = self.play_start + self.max_play_dim - 1
        print(f"出牌动作索引范围: {self.play_start} - {self.play_end}")
        self.potion_start = self.play_end + 1
        self.potion_end = self.potion_start + self.max_potion_dim - 1
        print(f"药水动作索引范围: {self.potion_start} - {self.potion_end}")
        self.single_start = self.potion_end + 1
        self.single_actions = {
            "return": self.single_start + 0, 
            "end": self.single_start + 1,    
            "proceed": self.single_start + 2,
            "skip": self.single_start + 3,
            "leave": self.single_start + 4,
            "confirm": self.single_start + 5,
            "cancel": self.single_start + 6,
        }
        self.single_end = self.single_start + self.max_single_dim - 1
        print(f"单一动作索引范围: {self.single_start} - {self.single_end}")
    # def action_to_index(self, command: str):
