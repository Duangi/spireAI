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
    children: List[Point] = field(default_factory=list)
    @classmethod
    def get_vec_length(self):
        return 14
        
    def get_vector(self):
        # 调试：打印关键变量类型和值（运行后看输出）
        room_type = self.symbol_to_type(self.symbol)
        room_type_value = room_type.value
        # 默认3个children，如果只有一个剩余的就空着
        children_vec = torch.zeros([3,2])
        if len(self.children) > 0:
            for i,c in enumerate(self.children):
                children_vec[i] = torch.tensor([
                    minmax_normalize(c.x,0,6),
                    minmax_normalize(c.y,0,20),
                ])
        
        vec = torch.cat([
            torch.tensor([minmax_normalize(self.x,0,6)],dtype=torch.float32), # 1维
            torch.tensor([minmax_normalize(self.y,0,20)], dtype=torch.float32), # 1维
            nn.functional.one_hot(torch.tensor(int(room_type_value - 1)), num_classes=len(RoomType)), # 6维
            children_vec.flatten(),# 3*2=6维
        ])
        return vec # 14维
    def symbol_to_type(self,str):
        if str == "?":
            return RoomType.UNKNOWN
        elif str == "$":
            return RoomType.MERCHANT
        elif str == "T":
            return RoomType.TREASURE
        elif str == "R":
            return RoomType.REST
        elif str == "M":
            return RoomType.ENEMY
        elif str == "E":
            return RoomType.ELITE
        else:
            raise ValueError(f"Unknown symbol: {str}")
    @classmethod
    def from_json(cls, json_object):
        children_dicts = json_object.get("children", [])
        children_points = [Point(c.get("x"), c.get("y")) for c in children_dicts]
        return cls(json_object.get("x"), json_object.get("y"), json_object.get("symbol"), children_points)

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

        for json_node in node_list:
            children = json_node.get("children")
            parent_node = dungeon_map.get_node(json_node.get("x"), json_node.get("y"))
            for json_child in children:
                child_node = dungeon_map.get_node(json_child.get("x"), json_child.get("y"))
                if child_node is not None:
                    parent_node.children.append(child_node)
        
        # list表示的展平结构
        for json_node in node_list:
            node = Node.from_json(json_node)
            dungeon_map.nodes_flattened.append(node)
        return dungeon_map

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
    json_obj = json.loads(exam)
    map = Map.from_json(json_obj.get("map"))
    # 测试Map的节点数量是否正确
    assert len(map.nodes) == 15
    print(map.nodes)
    print(map.get_vector())