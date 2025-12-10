from collections import deque
from dataclasses import dataclass, field
import json
import torch.nn as nn
from typing import List
from spirecomm.utils.data_processing import minmax_normalize,normal_normalize
from enum import Enum
import torch

class RoomType(Enum):
    # 地图里没有boss节点
    UNKNOWN = 1 # 未知
    MERCHANT = 2 # 商店
    TREASURE = 3 # 宝箱 
    REST = 4 # 火堆
    ENEMY = 5 # 小怪
    ELITE = 6 # 精英
# 一个只有x和y的结构：
@dataclass
class Point:
    x:int
    y:int


@dataclass
class Node:
    x:int = 0
    y:int = 0
    symbol:str = "M"
    # 显式声明类型，且初始化为空列表，不要在 from_json 里塞 Point 进去
    children: List['Node'] = field(default_factory=list)
    @property
    def type_id(self) -> int:
        mapping = {"M": 1, "?": 2, "R": 3, "E": 4, "T": 5, "$": 6, "BOSS": 7}

        BOSS_FLOORS = {17, 34, 51, 55} # 第一幕, 第二幕, 第三幕, 终幕
        if self.y in BOSS_FLOORS:
            return 7 # 返回 BOSS 专用的 ID
        
        return mapping.get(self.symbol, 0)

    @classmethod
    def get_vec_length(self):
        return 14
        
    def get_pos_features(self):
        # 你的坐标归一化逻辑
        return torch.tensor([
            minmax_normalize(self.x, 0, 6),
            minmax_normalize(self.y, 0, 17) # 每层最多17层
        ], dtype=torch.float32)

    @classmethod
    def from_json(cls, json_object):
        return cls(
            x = json_object.get("x"), 
            y = json_object.get("y"), 
            symbol = json_object.get("symbol"),
            children = [] # 不在这里处理children
        )

    def __repr__(self):
        return "({},{})".format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

@dataclass
class Map:
    nodes:dict = field(default_factory=dict)
    nodes_flattened: List[Node] = field(default_factory=list)

    def add_node(self, node: Node):
        if node.y in self.nodes:
            self.nodes[node.y][node.x] = node
        else:
            self.nodes[node.y] = {node.x: node}

    def get_node(self, x, y):
        if y in self.nodes and x in self.nodes[y]:
            return self.nodes[y][x]
        else:
            return None

    def get_vector(self):
        if not self.nodes_flattened:
            return torch.zeros(14)
        vec_list = [node.get_vector() for node in self.nodes_flattened]
        return torch.flatten(torch.cat(vec_list,dim=0))

    @classmethod
    def from_json(cls, node_list):
        dungeon_map = Map()
        # dict表示的树状结构
        for json_node in node_list:
            node = Node.from_json(json_node)
            dungeon_map.add_node(node)
            dungeon_map.nodes_flattened.append(node) # 顺便填入 flatten 列表


        for json_node in node_list:
            parent_node = dungeon_map.get_node(json_node.get("x"), json_node.get("y"))
            children_data = json_node.get("children", [])
            
            for child_data in children_data:
                child_node = dungeon_map.get_node(child_data.get("x"), child_data.get("y"))
                if child_node is not None:
                    parent_node.children.append(child_node)

        return dungeon_map

    def get_reachable_mask(self, current_x, current_y) -> List[bool]:
        reachable = set()
        if current_x == 0 and current_y == -1:
            # 起始节点，全部可达
            return [True] * len(self.nodes_flattened)

        start_node = self.get_node(current_x, current_y)
        if start_node is None:
            return [False] * len(self.nodes_flattened)
        
        queue = deque([start_node])
        reachable.add((start_node.x, start_node.y))

        while queue:
            curr_node = queue.popleft()

            for child in curr_node.children:
                if (child.x, child.y) not in reachable:
                    reachable.add((child.x, child.y))
                    queue.append(child)

        mask = []
        for node in self.nodes_flattened:
            if (node.x, node.y) in reachable:
                mask.append(True)
            else:
                mask.append(False)
        return mask

if __name__ == "__main__":
    exam = """
    {
    "map": [
        {
            "symbol": "M",
            "children": [
                {
                    "x": 2,
                    "y": 1
                }
            ],
            "x": 1,
            "y": 0,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 3,
                    "y": 1
                }
            ],
            "x": 2,
            "y": 0,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 5,
                    "y": 1
                }
            ],
            "x": 6,
            "y": 0,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 3,
                    "y": 2
                }
            ],
            "x": 2,
            "y": 1,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 3,
                    "y": 2
                },
                {
                    "x": 4,
                    "y": 2
                }
            ],
            "x": 3,
            "y": 1,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 4,
                    "y": 2
                }
            ],
            "x": 5,
            "y": 1,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 2,
                    "y": 3
                }
            ],
            "x": 3,
            "y": 2,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 3,
                    "y": 3
                },
                {
                    "x": 5,
                    "y": 3
                }
            ],
            "x": 4,
            "y": 2,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 1,
                    "y": 4
                },
                {
                    "x": 2,
                    "y": 4
                }
            ],
            "x": 2,
            "y": 3,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 2,
                    "y": 4
                },
                {
                    "x": 3,
                    "y": 4
                }
            ],
            "x": 3,
            "y": 3,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 4,
                    "y": 4
                }
            ],
            "x": 5,
            "y": 3,
            "parents": []
        },
        {
            "symbol": "$",
            "children": [
                {
                    "x": 0,
                    "y": 5
                }
            ],
            "x": 1,
            "y": 4,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 1,
                    "y": 5
                },
                {
                    "x": 3,
                    "y": 5
                }
            ],
            "x": 2,
            "y": 4,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 4,
                    "y": 5
                }
            ],
            "x": 3,
            "y": 4,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 4,
                    "y": 5
                }
            ],
            "x": 4,
            "y": 4,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 0,
                    "y": 6
                }
            ],
            "x": 0,
            "y": 5,
            "parents": []
        },
        {
            "symbol": "E",
            "children": [
                {
                    "x": 0,
                    "y": 6
                }
            ],
            "x": 1,
            "y": 5,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 2,
                    "y": 6
                },
                {
                    "x": 3,
                    "y": 6
                }
            ],
            "x": 3,
            "y": 5,
            "parents": []
        },
        {
            "symbol": "E",
            "children": [
                {
                    "x": 3,
                    "y": 6
                }
            ],
            "x": 4,
            "y": 5,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 0,
                    "y": 7
                },
                {
                    "x": 1,
                    "y": 7
                }
            ],
            "x": 0,
            "y": 6,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 1,
                    "y": 7
                }
            ],
            "x": 2,
            "y": 6,
            "parents": []
        },
        {
            "symbol": "$",
            "children": [
                {
                    "x": 2,
                    "y": 7
                },
                {
                    "x": 3,
                    "y": 7
                }
            ],
            "x": 3,
            "y": 6,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 0,
                    "y": 8
                }
            ],
            "x": 0,
            "y": 7,
            "parents": []
        },
        {
            "symbol": "E",
            "children": [
                {
                    "x": 1,
                    "y": 8
                }
            ],
            "x": 1,
            "y": 7,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 1,
                    "y": 8
                },
                {
                    "x": 3,
                    "y": 8
                }
            ],
            "x": 2,
            "y": 7,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 4,
                    "y": 8
                }
            ],
            "x": 3,
            "y": 7,
            "parents": []
        },
        {
            "symbol": "T",
            "children": [
                {
                    "x": 1,
                    "y": 9
                }
            ],
            "x": 0,
            "y": 8,
            "parents": []
        },
        {
            "symbol": "T",
            "children": [
                {
                    "x": 1,
                    "y": 9
                },
                {
                    "x": 2,
                    "y": 9
                }
            ],
            "x": 1,
            "y": 8,
            "parents": []
        },
        {
            "symbol": "T",
            "children": [
                {
                    "x": 3,
                    "y": 9
                }
            ],
            "x": 3,
            "y": 8,
            "parents": []
        },
        {
            "symbol": "T",
            "children": [
                {
                    "x": 3,
                    "y": 9
                }
            ],
            "x": 4,
            "y": 8,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 0,
                    "y": 10
                }
            ],
            "x": 1,
            "y": 9,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 1,
                    "y": 10
                },
                {
                    "x": 3,
                    "y": 10
                }
            ],
            "x": 2,
            "y": 9,
            "parents": []
        },
        {
            "symbol": "$",
            "children": [
                {
                    "x": 3,
                    "y": 10
                },
                {
                    "x": 4,
                    "y": 10
                }
            ],
            "x": 3,
            "y": 9,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 1,
                    "y": 11
                }
            ],
            "x": 0,
            "y": 10,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 1,
                    "y": 11
                }
            ],
            "x": 1,
            "y": 10,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 2,
                    "y": 11
                },
                {
                    "x": 4,
                    "y": 11
                }
            ],
            "x": 3,
            "y": 10,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 4,
                    "y": 11
                }
            ],
            "x": 4,
            "y": 10,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 0,
                    "y": 12
                },
                {
                    "x": 1,
                    "y": 12
                },
                {
                    "x": 2,
                    "y": 12
                }
            ],
            "x": 1,
            "y": 11,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 3,
                    "y": 12
                }
            ],
            "x": 2,
            "y": 11,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 4,
                    "y": 12
                },
                {
                    "x": 5,
                    "y": 12
                }
            ],
            "x": 4,
            "y": 11,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 0,
                    "y": 13
                }
            ],
            "x": 0,
            "y": 12,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 0,
                    "y": 13
                }
            ],
            "x": 1,
            "y": 12,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 1,
                    "y": 13
                }
            ],
            "x": 2,
            "y": 12,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 4,
                    "y": 13
                }
            ],
            "x": 3,
            "y": 12,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 4,
                    "y": 13
                }
            ],
            "x": 4,
            "y": 12,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 4,
                    "y": 13
                }
            ],
            "x": 5,
            "y": 12,
            "parents": []
        },
        {
            "symbol": "E",
            "children": [
                {
                    "x": 0,
                    "y": 14
                },
                {
                    "x": 1,
                    "y": 14
                }
            ],
            "x": 0,
            "y": 13,
            "parents": []
        },
        {
            "symbol": "M",
            "children": [
                {
                    "x": 2,
                    "y": 14
                }
            ],
            "x": 1,
            "y": 13,
            "parents": []
        },
        {
            "symbol": "?",
            "children": [
                {
                    "x": 3,
                    "y": 14
                },
                {
                    "x": 5,
                    "y": 14
                }
            ],
            "x": 4,
            "y": 13,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 3,
                    "y": 16
                }
            ],
            "x": 0,
            "y": 14,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 3,
                    "y": 16
                }
            ],
            "x": 1,
            "y": 14,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 3,
                    "y": 16
                }
            ],
            "x": 2,
            "y": 14,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 3,
                    "y": 16
                }
            ],
            "x": 3,
            "y": 14,
            "parents": []
        },
        {
            "symbol": "R",
            "children": [
                {
                    "x": 3,
                    "y": 16
                }
            ],
            "x": 5,
            "y": 14,
            "parents": []
        }
    ]
}
"""
    # 测试exam转成Map
    exam2= """{
    "map":[{"symbol":"M","children":[{"x":2,"y":1}],"x":2,"y":0,"parents":[]},{"symbol":"M","children":[{"x":5,"y":1}],"x":5,"y":0,"parents":[]},{"symbol":"M","children":[{"x":6,"y":1}],"x":6,"y":0,"parents":[]},{"symbol":"M","children":[{"x":1,"y":2}],"x":2,"y":1,"parents":[]},{"symbol":"?","children":[{"x":4,"y":2},{"x":5,"y":2},{"x":6,"y":2}],"x":5,"y":1,"parents":[]},{"symbol":"M","children":[{"x":6,"y":2}],"x":6,"y":1,"parents":[]},{"symbol":"M","children":[{"x":1,"y":3}],"x":1,"y":2,"parents":[]},{"symbol":"?","children":[{"x":3,"y":3}],"x":4,"y":2,"parents":[]},{"symbol":"M","children":[{"x":4,"y":3}],"x":5,"y":2,"parents":[]},{"symbol":"$","children":[{"x":6,"y":3}],"x":6,"y":2,"parents":[]},{"symbol":"?","children":[{"x":2,"y":4}],"x":1,"y":3,"parents":[]},{"symbol":"?","children":[{"x":2,"y":4}],"x":3,"y":3,"parents":[]},{"symbol":"M","children":[{"x":3,"y":4},{"x":4,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"M","children":[{"x":5,"y":4},{"x":6,"y":4}],"x":6,"y":3,"parents":[]},{"symbol":"M","children":[{"x":1,"y":5}],"x":2,"y":4,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"M","children":[{"x":4,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"?","children":[{"x":4,"y":5}],"x":5,"y":4,"parents":[]},{"symbol":"M","children":[{"x":6,"y":5}],"x":6,"y":4,"parents":[]},{"symbol":"E","children":[{"x":0,"y":6}],"x":1,"y":5,"parents":[]},{"symbol":"R","children":[{"x":3,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"E","children":[{"x":3,"y":6}],"x":4,"y":5,"parents":[]},{"symbol":"R","children":[{"x":5,"y":6}],"x":6,"y":5,"parents":[]},{"symbol":"R","children":[{"x":1,"y":7}],"x":0,"y":6,"parents":[]},{"symbol":"?","children":[{"x":2,"y":7},{"x":3,"y":7},{"x":4,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"M","children":[{"x":4,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"M","children":[{"x":0,"y":8},{"x":1,"y":8}],"x":1,"y":7,"parents":[]},{"symbol":"R","children":[{"x":2,"y":8}],"x":2,"y":7,"parents":[]},{"symbol":"?","children":[{"x":4,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"E","children":[{"x":4,"y":8}],"x":4,"y":7,"parents":[]},{"symbol":"T","children":[{"x":0,"y":9}],"x":0,"y":8,"parents":[]},{"symbol":"T","children":[{"x":1,"y":9}],"x":1,"y":8,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9},{"x":4,"y":9},{"x":5,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"?","children":[{"x":1,"y":10}],"x":0,"y":9,"parents":[]},{"symbol":"?","children":[{"x":2,"y":10}],"x":1,"y":9,"parents":[]},{"symbol":"R","children":[{"x":2,"y":10},{"x":3,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"M","children":[{"x":4,"y":10}],"x":4,"y":9,"parents":[]},{"symbol":"E","children":[{"x":5,"y":10}],"x":5,"y":9,"parents":[]},{"symbol":"M","children":[{"x":2,"y":11}],"x":1,"y":10,"parents":[]},{"symbol":"E","children":[{"x":2,"y":11}],"x":2,"y":10,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"R","children":[{"x":3,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"?","children":[{"x":5,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"M","children":[{"x":1,"y":12},{"x":2,"y":12}],"x":2,"y":11,"parents":[]},{"symbol":"M","children":[{"x":2,"y":12},{"x":3,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"M","children":[{"x":5,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"M","children":[{"x":1,"y":13}],"x":1,"y":12,"parents":[]},{"symbol":"R","children":[{"x":2,"y":13},{"x":3,"y":13}],"x":2,"y":12,"parents":[]},{"symbol":"E","children":[{"x":3,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"?","children":[{"x":5,"y":13}],"x":5,"y":12,"parents":[]},{"symbol":"M","children":[{"x":1,"y":14}],"x":1,"y":13,"parents":[]},{"symbol":"E","children":[{"x":2,"y":14},{"x":3,"y":14}],"x":2,"y":13,"parents":[]},{"symbol":"$","children":[{"x":3,"y":14},{"x":4,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"$","children":[{"x":4,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":1,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":2,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":3,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]}]
    }
    """
    """
    map2 的结构如下：
            /  / | \        
14        R  R  R  R       
          |  |/ |/   \     
13        M  E  $     $    
          |  |/ |     |    
12        M  R  E     ?    
            \| \|     |    
11           M  M     M    
           / |  | \   |    
10        M  E  M  R  ?    
        /  /   \|  |  |    
9      ?  ?     R  M  E    
       |  |   /   \|/      
8      T  T  T     T       
         \|  |   / |       
7         M  R  ?  E       
        /      \|/   \     
6      R        ?     M    
         \      | \     \  
5         E     R  E     R 
            \   |  | \   | 
4            M  ?  M  ?  M 
           /   \  \|    \| 
3         ?     ?  M     M 
          |       \  \   | 
2         M        ?  M  $ 
            \        \|/ | 
1            M        ?  M 
             |        |  | 
0            M        M  M 
       1  2  3  4   5  6
    """
    json_obj = json.loads(exam)
    map = Map.from_json(json_obj.get("map"))
    map2 = Map.from_json(json.loads(exam2).get("map"))
    # 测试Map的节点数量是否正确
    print(map2.nodes)
    print(map2.get_reachable_mask(0,-1))
    print(map2.nodes_flattened)