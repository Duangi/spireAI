"""
小工具：从 JSON 字符串或文件构造 Game 对象并让 Agent 给出动作（打印出来）。
用法：
  python scripts/action_from_state.py '<json_string>'
  python scripts/action_from_state.py path/to/state.json
  echo '<json_string>' | python scripts/action_from_state.py

JSON 格式：推荐传入完整的 communication_state，示例：
{
  "game_state": { ... },
  "available_commands": ["play","end"]
}
或直接传入 game_state 对象并单独提供 available_commands（脚本会自动适配）

输出：打印出 agent 选择的动作字符串（如 'play 0 1' 或 'end' 等）
"""
import sys
import os
import json

# 保证工程根目录在 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from spirecomm.spire.game import Game
from spirecomm.ai.dqn import DQNAgent


def load_input(arg: str):
    """如果 arg 是路径则读取文件，否则尝试解析为 JSON 字符串。若 arg 为空则从 stdin 读取。"""
    if not arg:
        s = sys.stdin.read()
    else:
        # 如果是路径
        if os.path.exists(arg):
            with open(arg, 'r', encoding='utf-8') as f:
                s = f.read()
        else:
            s = arg
    s = s.strip()
    if not s:
        raise ValueError('没有输入数据')
    try:
        data = json.loads(s)
    except Exception as e:
        raise ValueError(f'JSON 解析失败: {e}')
    return data


# 如果你想直接在脚本文件里粘很长的 JSON 字符串用于测试，请把它粘到下面的变量里（保留三引号）。
# 示例：
# TEST_INPUT_JSON = '''{ "game_state": {...}, "available_commands": ["play","end"] }'''
# 设置为 None 时脚本仍会从命令行参数或 stdin 读取输入
TEST_INPUT_JSON = """{"available_commands":["play","end","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"screen_type":"NONE","screen_state":{},"seed":5000075112971018798,"combat_state":{"draw_pile":[{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"09d7aeec-300f-456c-a315-948f1e3a2798","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"8f0ca0f5-ad9a-4ea3-b939-9e41870da2f4","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"4fcfbac3-346f-44c1-a44f-edff143823a5","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"防御","id":"Defend_B","type":"SKILL","ethereal":false,"uuid":"215ea22e-3145-473d-b0c4-ab9064dc424d","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"防御","id":"Defend_B","type":"SKILL","ethereal":false,"uuid":"3d2a62b5-efce-45a4-8afa-8c39ac0067c0","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"弹回","id":"Rebound","type":"ATTACK","ethereal":false,"uuid":"b2f6be05-f991-496d-98fb-2b18c2d9496f","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"防御","id":"Defend_B","type":"SKILL","ethereal":false,"uuid":"41557a91-cc63-449b-ba8b-4ca2cecc7e43","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"549d480d-46d5-41b0-af74-c91a8a873f40","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"39a612a5-a9af-4bf0-882e-eede456d2460","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"is_playable":true,"cost":0,"name":"电击+","id":"Zap","type":"SKILL","ethereal":false,"uuid":"14e04fb1-928a-4bb8-99ac-f058d89ca84f","upgrades":1,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"双重释放","id":"Dualcast","type":"SKILL","ethereal":false,"uuid":"66242864-88ed-4fb3-a2c8-a609e3df6a34","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":2,"name":"愁云惨淡","id":"Doom and Gloom","type":"ATTACK","ethereal":false,"uuid":"68ab45bc-7f05-4b49-9f42-b766e26272ff","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"is_playable":true,"cost":0,"name":"光束射线","id":"Beam Cell","type":"ATTACK","ethereal":false,"uuid":"a84b412b-1806-4bd9-b08c-12f000e341f1","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"81802a25-ad69-4fda-959d-09a8c4209b50","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"e97c42f6-c3d3-4f8f-a50b-df0d02ccd8f0","upgrades":0,"rarity":"COMMON","has_target":false}],"discard_pile":[],"exhaust_pile":[{"exhausts":true,"is_playable":true,"cost":1,"name":"全息影像","id":"Hologram","type":"SKILL","ethereal":false,"uuid":"10d949da-8eb3-47c8-8202-afc019f060d4","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":true,"is_playable":true,"cost":1,"name":"黏液","id":"Slimed","type":"STATUS","ethereal":false,"uuid":"e76ca35b-0be7-4791-aeef-085969d1db13","upgrades":0,"rarity":"COMMON","has_target":false}],"cards_discarded_this_turn":0,"times_damaged":2,"monsters":[{"is_gone":false,"move_hits":1,"move_base_damage":-1,"half_dead":false,"move_adjusted_damage":-1,"max_hp":26,"intent":"DEBUFF","move_id":4,"name":"尖刺史莱姆（中）","current_hp":26,"block":0,"id":"SpikeSlime_M","powers":[]},{"is_gone":true,"move_hits":1,"move_base_damage":-1,"last_move_id":3,"half_dead":false,"move_adjusted_damage":-1,"max_hp":69,"intent":"DEBUFF","second_last_move_id":3,"move_id":4,"name":"尖刺史莱姆（大）","current_hp":0,"block":0,"id":"SpikeSlime_L","powers":[]},{"is_gone":false,"move_hits":1,"move_base_damage":-1,"half_dead":false,"move_adjusted_damage":-1,"max_hp":26,"intent":"DEBUFF","move_id":4,"name":"尖刺史莱姆（中）","current_hp":26,"block":0,"id":"SpikeSlime_M","powers":[]},{"is_gone":false,"move_hits":1,"move_base_damage":7,"last_move_id":4,"half_dead":false,"move_adjusted_damage":7,"max_hp":27,"intent":"ATTACK_DEBUFF","move_id":1,"name":"酸液史莱姆（中）","current_hp":9,"block":0,"id":"AcidSlime_M","powers":[]},{"is_gone":true,"move_hits":1,"move_base_damage":-1,"last_move_id":3,"half_dead":false,"move_adjusted_damage":-1,"max_hp":140,"intent":"UNKNOWN","second_last_move_id":3,"move_id":3,"name":"史莱姆老大","current_hp":0,"block":0,"id":"SlimeBoss","powers":[]},{"is_gone":true,"move_hits":1,"move_base_damage":-1,"last_move_id":3,"half_dead":false,"move_adjusted_damage":-1,"max_hp":69,"intent":"UNKNOWN","second_last_move_id":3,"move_id":3,"name":"酸液史莱姆（大）","current_hp":0,"block":0,"id":"AcidSlime_L","powers":[]},{"is_gone":false,"move_hits":1,"move_base_damage":10,"last_move_id":4,"half_dead":false,"move_adjusted_damage":10,"max_hp":27,"intent":"ATTACK","move_id":2,"name":"酸液史莱姆（中）","current_hp":14,"block":0,"id":"AcidSlime_M","powers":[]}],"turn":7,"limbo":[],"hand":[{"exhausts":false,"is_playable":true,"cost":2,"name":"冰川","id":"Glacier","type":"SKILL","ethereal":false,"uuid":"9195f4fd-d6dc-4eff-98e5-928ebe6cd585","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"e93e086a-8d98-4084-966d-79d6a6bbbb00","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"c43bc52c-6da4-40ff-a138-e630be2ded9b","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"球状闪电","id":"Ball Lightning","type":"ATTACK","ethereal":false,"uuid":"893d274a-462d-4e04-83c9-072e652f9fb6","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"026081b2-42eb-4afe-847d-e96766b350fe","upgrades":0,"rarity":"BASIC","has_target":true}],"player":{"orbs":[{"passive_amount":3,"name":"闪电","id":"Lightning","evoke_amount":8},{"passive_amount":3,"name":"闪电","id":"Lightning","evoke_amount":8},{"passive_amount":6,"name":"黑暗","id":"Dark","evoke_amount":12}],"current_hp":18,"block":0,"max_hp":75,"powers":[{"amount":2,"just_applied":false,"name":"虚弱","id":"Weakened"}],"energy":3}},"deck":[{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"e93e086a-8d98-4084-966d-79d6a6bbbb00","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"c43bc52c-6da4-40ff-a138-e630be2ded9b","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"81802a25-ad69-4fda-959d-09a8c4209b50","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"打击","id":"Strike_B","type":"ATTACK","ethereal":false,"uuid":"026081b2-42eb-4afe-847d-e96766b350fe","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"is_playable":true,"cost":1,"name":"防御","id":"Defend_B","type":"SKILL","ethereal":false,"uuid":"41557a91-cc63-449b-ba8b-4ca2cecc7e43","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"防御","id":"Defend_B","type":"SKILL","ethereal":false,"uuid":"215ea22e-3145-473d-b0c4-ab9064dc424d","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"防御","id":"Defend_B","type":"SKILL","ethereal":false,"uuid":"3d2a62b5-efce-45a4-8afa-8c39ac0067c0","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":0,"name":"电击+","id":"Zap","type":"SKILL","ethereal":false,"uuid":"14e04fb1-928a-4bb8-99ac-f058d89ca84f","upgrades":1,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"双重释放","id":"Dualcast","type":"SKILL","ethereal":false,"uuid":"66242864-88ed-4fb3-a2c8-a609e3df6a34","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"is_playable":true,"cost":2,"name":"愁云惨淡","id":"Doom and Gloom","type":"ATTACK","ethereal":false,"uuid":"68ab45bc-7f05-4b49-9f42-b766e26272ff","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":true,"is_playable":true,"cost":1,"name":"全息影像","id":"Hologram","type":"SKILL","ethereal":false,"uuid":"10d949da-8eb3-47c8-8202-afc019f060d4","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"弹回","id":"Rebound","type":"ATTACK","ethereal":false,"uuid":"b2f6be05-f991-496d-98fb-2b18c2d9496f","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"is_playable":true,"cost":2,"name":"冰川","id":"Glacier","type":"SKILL","ethereal":false,"uuid":"9195f4fd-d6dc-4eff-98e5-928ebe6cd585","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"is_playable":true,"cost":1,"name":"球状闪电","id":"Ball Lightning","type":"ATTACK","ethereal":false,"uuid":"893d274a-462d-4e04-83c9-072e652f9fb6","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"is_playable":true,"cost":0,"name":"光束射线","id":"Beam Cell","type":"ATTACK","ethereal":false,"uuid":"a84b412b-1806-4bd9-b08c-12f000e341f1","upgrades":0,"rarity":"COMMON","has_target":true}],"relics":[{"name":"破损核心","id":"Cracked Core","counter":-1},{"name":"涅奥的悲恸","id":"NeowsBlessing","counter":-2}],"max_hp":75,"act_boss":"Slime Boss","gold":118,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"NONE","room_phase":"COMBAT","is_screen_up":false,"potions":[{"requires_target":false,"can_use":false,"can_discard":false,"name":"药水栏","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"药水栏","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"药水栏","id":"Potion Slot"}],"current_hp":18,"floor":16,"ascension_level":0,"class":"DEFECT","map":[{"symbol":"M","children":[{"x":1,"y":1}],"x":0,"y":0,"parents":[]},{"symbol":"M","children":[{"x":2,"y":1}],"x":1,"y":0,"parents":[]},{"symbol":"M","children":[{"x":4,"y":1}],"x":4,"y":0,"parents":[]},{"symbol":"M","children":[{"x":6,"y":1}],"x":6,"y":0,"parents":[]},{"symbol":"?","children":[{"x":1,"y":2}],"x":1,"y":1,"parents":[]},{"symbol":"?","children":[{"x":1,"y":2}],"x":2,"y":1,"parents":[]},{"symbol":"M","children":[{"x":3,"y":2}],"x":4,"y":1,"parents":[]},{"symbol":"$","children":[{"x":5,"y":2}],"x":6,"y":1,"parents":[]},{"symbol":"?","children":[{"x":1,"y":3},{"x":2,"y":3}],"x":1,"y":2,"parents":[]},{"symbol":"M","children":[{"x":4,"y":3}],"x":3,"y":2,"parents":[]},{"symbol":"?","children":[{"x":6,"y":3}],"x":5,"y":2,"parents":[]},{"symbol":"M","children":[{"x":0,"y":4},{"x":1,"y":4}],"x":1,"y":3,"parents":[]},{"symbol":"?","children":[{"x":1,"y":4},{"x":2,"y":4}],"x":2,"y":3,"parents":[]},{"symbol":"M","children":[{"x":5,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"M","children":[{"x":6,"y":4}],"x":6,"y":3,"parents":[]},{"symbol":"?","children":[{"x":0,"y":5}],"x":0,"y":4,"parents":[]},{"symbol":"M","children":[{"x":0,"y":5},{"x":1,"y":5}],"x":1,"y":4,"parents":[]},{"symbol":"?","children":[{"x":3,"y":5}],"x":2,"y":4,"parents":[]},{"symbol":"M","children":[{"x":6,"y":5}],"x":5,"y":4,"parents":[]},{"symbol":"?","children":[{"x":6,"y":5}],"x":6,"y":4,"parents":[]},{"symbol":"R","children":[{"x":0,"y":6}],"x":0,"y":5,"parents":[]},{"symbol":"E","children":[{"x":1,"y":6}],"x":1,"y":5,"parents":[]},{"symbol":"R","children":[{"x":2,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"R","children":[{"x":6,"y":6}],"x":6,"y":5,"parents":[]},{"symbol":"M","children":[{"x":1,"y":7}],"x":0,"y":6,"parents":[]},{"symbol":"R","children":[{"x":1,"y":7}],"x":1,"y":6,"parents":[]},{"symbol":"?","children":[{"x":3,"y":7}],"x":2,"y":6,"parents":[]},{"symbol":"?","children":[{"x":5,"y":7},{"x":6,"y":7}],"x":6,"y":6,"parents":[]},{"symbol":"M","children":[{"x":1,"y":8}],"x":1,"y":7,"parents":[]},{"symbol":"M","children":[{"x":2,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"M","children":[{"x":4,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"R","children":[{"x":5,"y":8}],"x":6,"y":7,"parents":[]},{"symbol":"T","children":[{"x":0,"y":9},{"x":2,"y":9}],"x":1,"y":8,"parents":[]},{"symbol":"T","children":[{"x":2,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":4,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"T","children":[{"x":4,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"M","children":[{"x":1,"y":10}],"x":0,"y":9,"parents":[]},{"symbol":"E","children":[{"x":2,"y":10},{"x":3,"y":10}],"x":2,"y":9,"parents":[]},{"symbol":"M","children":[{"x":3,"y":10},{"x":5,"y":10}],"x":4,"y":9,"parents":[]},{"symbol":"M","children":[{"x":1,"y":11},{"x":2,"y":11}],"x":1,"y":10,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11}],"x":2,"y":10,"parents":[]},{"symbol":"?","children":[{"x":3,"y":11},{"x":4,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"E","children":[{"x":5,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"M","children":[{"x":1,"y":12}],"x":1,"y":11,"parents":[]},{"symbol":"E","children":[{"x":3,"y":12}],"x":2,"y":11,"parents":[]},{"symbol":"M","children":[{"x":4,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"$","children":[{"x":5,"y":12}],"x":4,"y":11,"parents":[]},{"symbol":"M","children":[{"x":6,"y":12}],"x":5,"y":11,"parents":[]},{"symbol":"?","children":[{"x":0,"y":13}],"x":1,"y":12,"parents":[]},{"symbol":"?","children":[{"x":3,"y":13}],"x":3,"y":12,"parents":[]},{"symbol":"R","children":[{"x":3,"y":13},{"x":4,"y":13}],"x":4,"y":12,"parents":[]},{"symbol":"M","children":[{"x":4,"y":13}],"x":5,"y":12,"parents":[]},{"symbol":"$","children":[{"x":5,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"E","children":[{"x":0,"y":14}],"x":0,"y":13,"parents":[]},{"symbol":"M","children":[{"x":3,"y":14},{"x":4,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"M","children":[{"x":4,"y":14},{"x":5,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"M","children":[{"x":6,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":0,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":3,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":6,"y":14,"parents":[]}],"room_type":"MonsterRoomBoss"}}"""


def main():
    # 优先使用文件内的 TEST_INPUT_JSON（方便手动粘很长字符串）
    if TEST_INPUT_JSON is not None:
        try:
            parsed = json.loads(TEST_INPUT_JSON)
        except Exception as e:
            print(f'从 TEST_INPUT_JSON 解析 JSON 失败: {e}', file=sys.stderr)
            sys.exit(2)
    else:
        arg = sys.argv[1] if len(sys.argv) > 1 else None
        try:
            parsed = load_input(arg)
        except Exception as e:
            print(f'加载输入失败: {e}', file=sys.stderr)
            sys.exit(2)

    # 尝试用 Game.from_json 构造 Game 对象
    try:
        # 如果 parsed 看起来像完整的 communication_state（包含 game_state 或 available_commands），直接传入
        if isinstance(parsed, dict) and ("game_state" in parsed or "available_commands" in parsed):
            game = Game.from_json(parsed)
        else:
            # 假设传入的是 game_state 本体（没有 available_commands），尝试包一层
            game = Game.from_json({"game_state": parsed, "available_commands": parsed.get("available_commands", [])})
    except Exception as e:
        print(f'构造 Game 对象失败: {e}', file=sys.stderr)
        sys.exit(3)

    # 创建 agent（游玩模式，不会训练）
    try:
        agent = DQNAgent(play_mode=True)
    except Exception as e:
        print(f'创建 DQNAgent 失败: {e}', file=sys.stderr)
        sys.exit(4)

    try:
        action_str = agent.get_next_action_in_game(game)
        print(action_str)
    except Exception as e:
        print(f'获取动作失败: {e}', file=sys.stderr)
        sys.exit(5)


if __name__ == '__main__':
    main()
