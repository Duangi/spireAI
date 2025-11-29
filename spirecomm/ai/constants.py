# 和action space相关的一些常量 

MAX_HAND_SIZE = 10 # 手牌数量上限
MAX_MONSTER_COUNT = 5 # 最多同时存在的怪物数量：蛇女+四个匕首 或者 五只小史莱姆。有待求证
MAX_DECK_SIZE = 100 # 卡组最大数量, 理论上没有上限，但实际游戏中一般不会超过100张吧？或者超过100张就很少见了，或者干脆超过100张就进行惩罚
MAX_POTION_COUNT = 5 # 最多携带3个药水
MAX_MAP_NODE_COUNT = 75 # 每一层地图上最多存在的节点数量，应该是加了不少冗余，例子中只有50多个。
MAX_ORB_COUNT = 10 # 最多10个充能球
MAX_POWER_COUNT = 20 # 最多9种通用buff，8种debuff，以及一些特殊buff，再怎么应该也不会超过20层
MAX_CHOICES = 20 # 通用选择项的最大数量
MAX_SHOP_ITEMS = 8 # 商店物品（7卡+1删牌）
MAX_REST_OPTIONS = 6 # 休息处选项