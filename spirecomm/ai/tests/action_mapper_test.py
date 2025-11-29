from spirecomm.ai.dqn_core.action import ActionMapper, ChooseAction, PlayAction, PotionDiscardAction, PotionUseAction, SingleAction, ActionType
import pytest

def test_ranges_are_correct():
    am = ActionMapper()

    # expected ranges based on constants in the module
    assert am.choose_start == 0
    assert am.choose_end == 99

    assert am.play_without_target_start == 100
    assert am.play_without_target_end == 109
    assert am.play_with_target_start == 110
    assert am.play_with_target_end == 159

    assert am.potion_discard_start == 160
    assert am.potion_discard_end == 164
    assert am.potion_use_without_target_start == 165
    assert am.potion_use_without_target_end == 169
    assert am.potion_use_with_target_start == 170
    assert am.potion_use_with_target_end == 194

    assert am.single_start == 195
    assert am.single_end == 201

    assert am.max_action_dim == 202


def test_action_index_roundtrip_all_indices():
    am = ActionMapper()
    for idx in range(am.max_action_dim):
        action = am.index_to_action(idx)
        idx2 = am.action_to_index(action)
        assert idx == idx2, f"roundtrip mismatch for index {idx}: got {idx2}"


def test_specific_action_creations_and_mappings():
    am = ActionMapper()

    # Choose action
    a = ChooseAction(type=ActionType.CHOOSE, choice_idx=0)
    assert am.action_to_index(a) == am.choose_start + 0
    assert am.index_to_action(am.choose_start) == a

    a_last = ChooseAction(type=ActionType.CHOOSE, choice_idx=99)
    assert am.action_to_index(a_last) == am.choose_end
    assert am.index_to_action(am.choose_end) == a_last

    # Play without target (hand_idx 0 and last)
    p = PlayAction(type=ActionType.PLAY, hand_idx=0, target_idx=None)
    assert am.action_to_index(p) == am.play_without_target_start
    assert am.index_to_action(am.play_without_target_start) == p

    p_last = PlayAction(type=ActionType.PLAY, hand_idx=9, target_idx=None)
    assert am.action_to_index(p_last) == am.play_without_target_end
    assert am.index_to_action(am.play_without_target_end) == p_last

    # Play with target (first and last)
    pwt = PlayAction(type=ActionType.PLAY, hand_idx=0, target_idx=0)
    assert am.action_to_index(pwt) == am.play_with_target_start
    assert am.index_to_action(am.play_with_target_start) == pwt

    pwt_last = PlayAction(type=ActionType.PLAY, hand_idx=9, target_idx=4)
    assert am.action_to_index(pwt_last) == am.play_with_target_end
    assert am.index_to_action(am.play_with_target_end) == pwt_last

    # Potion discard
    pd = PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=0)
    assert am.action_to_index(pd) == am.potion_discard_start
    assert am.index_to_action(am.potion_discard_start) == pd

    pd_last = PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=4)
    assert am.action_to_index(pd_last) == am.potion_discard_end
    assert am.index_to_action(am.potion_discard_end) == pd_last

    # Potion use without target
    pu = PotionUseAction(type=ActionType.POTION_USE, potion_idx=0, target_idx=None)
    assert am.action_to_index(pu) == am.potion_use_without_target_start
    assert am.index_to_action(am.potion_use_without_target_start) == pu

    pu_last = PotionUseAction(type=ActionType.POTION_USE, potion_idx=4, target_idx=None)
    assert am.action_to_index(pu_last) == am.potion_use_without_target_end
    assert am.index_to_action(am.potion_use_without_target_end) == pu_last

    # Potion use with target
    puwt = PotionUseAction(type=ActionType.POTION_USE, potion_idx=0, target_idx=0)
    assert am.action_to_index(puwt) == am.potion_use_with_target_start
    assert am.index_to_action(am.potion_use_with_target_start) == puwt

    puwt_last = PotionUseAction(type=ActionType.POTION_USE, potion_idx=4, target_idx=4)
    assert am.action_to_index(puwt_last) == am.potion_use_with_target_end
    assert am.index_to_action(am.potion_use_with_target_end) == puwt_last

    # Single actions
    for atype, idx in am.single_actions.items():
        s = SingleAction(type=atype)
        assert am.action_to_index(s) == idx
        assert am.index_to_action(idx) == s


def test_invalid_indices_raise():
    am = ActionMapper()
    with pytest.raises(ValueError):
        am.index_to_action(am.max_action_dim)  # out of range
    with pytest.raises(ValueError):
        am.index_to_action(-1)
