import math


# TODO 未使用
def adjust_score_tiered(
    initial_score: int, probabilities: list[float], thresholds: list[float], downgrade_levels: list[int]
) -> int:
    """
    根据大模型给出评分时的概率，对原始评分进行分级置信度调整。

    Args:
        initial_score: 大模型给出的原始评分 (整数 1 到 5)。
        probabilities: 包含 5 个评分 (1 到 5) 概率的列表。
                       例如 [P(1), P(2), P(3), P(4), P(5)]。
        thresholds: 一个降序排列的概率阈值列表，定义置信度区间边界。
                    例如 [0.6, 0.3]。
        downgrade_levels: 与 thresholds 对应的降级幅度列表，长度比 thresholds 多 1。
                          定义了每个置信度区间的降级数。例如 [0, 1, 2]。

    Returns:
        经过置信度调整后的最终评分 (整数 1 到 5)。

    Raises:
        ValueError: 如果输入参数不合法（例如概率列表长度不对，阈值未降序等）。
    """
    # --- 输入校验 ---
    if not (1 <= initial_score <= 5):
        raise ValueError("initial_score 必须在 1 到 5 之间。")
    if len(probabilities) != 5:
        raise ValueError("probabilities 列表必须包含 5 个元素。")
    # 检查概率和是否接近 1 (允许小的浮点误差)
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
        print(f"警告: 概率之和 {sum(probabilities)} 不接近 1.0。请检查概率来源。")  # 打印警告而非直接报错
        # raise ValueError("probabilities 中元素的和必须接近 1.0。")
    if len(downgrade_levels) != len(thresholds) + 1:
        raise ValueError("downgrade_levels 的长度必须比 thresholds 的长度多 1。")
    if any(thresholds[i] < thresholds[i + 1] for i in range(len(thresholds) - 1)):
        raise ValueError("thresholds 列表必须是降序排列的。")
    if any(level < 0 for level in downgrade_levels):
        raise ValueError("downgrade_levels 中的降级幅度不能为负数。")

    # --- 算法核心 ---
    # 1. 获取选中分数的概率
    # 列表索引从0开始，所以评分 s 对应的索引是 s-1
    try:
        p_chosen = probabilities[initial_score - 1]
    except IndexError:
        # 这个错误理论上不应发生，因为 initial_score 已校验在 1-5 之间
        raise ValueError(f"无法从 probabilities 列表获取索引 {initial_score - 1} 的值。")

    # 2. 确定降级幅度
    downgrade = downgrade_levels[-1]  # 默认为最低置信度区间的降级幅度
    # 遍历阈值列表 (从高到低)
    for i in range(len(thresholds)):
        if p_chosen >= thresholds[i]:
            downgrade = downgrade_levels[i]  # 找到对应的置信度区间
            break  # 停止遍历

    # 3. 计算调整后的评分
    preliminary_score = initial_score - downgrade
    adjusted_score = max(1, preliminary_score)  # 确保分数不低于 1

    # 4. 返回结果
    return adjusted_score
