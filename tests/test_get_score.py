import pytest
from weclone.data.clean.get_score import adjust_score_tiered

# 定义通用的参数
THRESHOLDS = [0.6, 0.3]       # 置信度阈值：>=0.6 高, >=0.3 中, <0.3 低
DOWNGRADE_LEVELS = [0, 1, 2] # 对应降级幅度：高->0级, 中->1级, 低->2级

THRESHOLDS_FINE = [0.7, 0.5, 0.3]
DOWNGRADE_LEVELS_FINE = [0, 1, 2, 3] # 对应 >=0.7, >=0.5, >=0.3, <0.3

test_cases = [
    # 案例 1: 高置信度
    (5, [0.05, 0.05, 0.1, 0.1, 0.7], THRESHOLDS, DOWNGRADE_LEVELS, 5, "高置信度"),
    # 案例 2: 中等置信度
    (4, [0.1, 0.15, 0.2, 0.45, 0.1], THRESHOLDS, DOWNGRADE_LEVELS, 3, "中等置信度"),
    # 案例 3: 低置信度
    (4, [0.15, 0.2, 0.25, 0.25, 0.15], THRESHOLDS, DOWNGRADE_LEVELS, 2, "低置信度"),
    # 案例 4: 低置信度，但原始分较低
    (2, [0.3, 0.2, 0.2, 0.15, 0.15], THRESHOLDS, DOWNGRADE_LEVELS, 1, "低置信度，原始分较低"),
    # 案例 5: 边界情况 - 刚好等于高阈值
    (3, [0.1, 0.1, 0.6, 0.1, 0.1], THRESHOLDS, DOWNGRADE_LEVELS, 3, "边界情况 - 等于高阈值"),
    # 案例 6: 边界情况 - 刚好等于中阈值
    (3, [0.2, 0.2, 0.3, 0.15, 0.15], THRESHOLDS, DOWNGRADE_LEVELS, 2, "边界情况 - 等于中阈值"),
    # 案例 7: 细分阈值 - 中高置信度
    (4, [0.1, 0.1, 0.2, 0.55, 0.05], THRESHOLDS_FINE, DOWNGRADE_LEVELS_FINE, 3, "细分阈值 - 中高置信度"),
    # 案例 8: 细分阈值 - 中低置信度
    (4, [0.15, 0.15, 0.2, 0.35, 0.15], THRESHOLDS_FINE, DOWNGRADE_LEVELS_FINE, 2, "细分阈值 - 中低置信度"),
    # 案例 9: 概率和异常 (预期行为是打印警告并继续计算)
    (3, [0.1, 0.1, 0.5, 0.1, 0.1], THRESHOLDS, DOWNGRADE_LEVELS, 3, "概率和异常"),
]

@pytest.mark.parametrize("initial_score, probabilities, thresholds, downgrade_levels, expected_score, description", test_cases)
def test_adjust_score_tiered(initial_score, probabilities, thresholds, downgrade_levels, expected_score, description):
    """ 测试 adjust_score_tiered 函数在各种情况下的表现 """
    print(f"测试案例: {description}")
    print(f"  输入: score={initial_score}, probs={probabilities}, thresholds={thresholds}, levels={downgrade_levels}")
    adjusted_score = adjust_score_tiered(initial_score, probabilities, thresholds, downgrade_levels)
    print(f"  输出: adjusted_score={adjusted_score}, 预期: {expected_score}")
    assert adjusted_score == expected_score

# 测试非法输入
def test_adjust_score_invalid_input():
    """ 测试非法输入是否按预期引发 ValueError """
    # initial_score 无效
    with pytest.raises(ValueError, match="initial_score 必须在 1 到 5 之间"):
        adjust_score_tiered(0, [0.2]*5, THRESHOLDS, DOWNGRADE_LEVELS)
    with pytest.raises(ValueError, match="initial_score 必须在 1 到 5 之间"):
        adjust_score_tiered(6, [0.2]*5, THRESHOLDS, DOWNGRADE_LEVELS)

    # probabilities 长度无效
    with pytest.raises(ValueError, match="probabilities 列表必须包含 5 个元素"):
        adjust_score_tiered(3, [0.2]*4, THRESHOLDS, DOWNGRADE_LEVELS)
    with pytest.raises(ValueError, match="probabilities 列表必须包含 5 个元素"):
        adjust_score_tiered(3, [0.1]*6, THRESHOLDS, DOWNGRADE_LEVELS) # 总和也不为1

    # # probabilities 和不为 1 (现在是警告，不抛异常)
    # with pytest.raises(ValueError, match="probabilities 中元素的和必须接近 1.0"):
    #     adjust_score_tiered(3, [0.1]*5, THRESHOLDS, DOWNGRADE_LEVELS)

    # downgrade_levels 长度无效
    with pytest.raises(ValueError, match="downgrade_levels 的长度必须比 thresholds 的长度多 1"):
        adjust_score_tiered(3, [0.2]*5, THRESHOLDS, [0, 1])
    with pytest.raises(ValueError, match="downgrade_levels 的长度必须比 thresholds 的长度多 1"):
        adjust_score_tiered(3, [0.2]*5, THRESHOLDS, [0, 1, 2, 3])

    # thresholds 不是降序
    with pytest.raises(ValueError, match="thresholds 列表必须是降序排列的"):
        adjust_score_tiered(3, [0.2]*5, [0.3, 0.6], DOWNGRADE_LEVELS)

    # downgrade_levels 包含负数
    with pytest.raises(ValueError, match="downgrade_levels 中的降级幅度不能为负数"):
        adjust_score_tiered(3, [0.2]*5, THRESHOLDS, [0, -1, 2]) 