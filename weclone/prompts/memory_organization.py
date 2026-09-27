import json


DIMENSIONS = {
    1: "人口学与基本身份：B的身份信息、社会位置和家庭角色等相对稳定事实。",
    3: "长期需求、动机与目标：B持续追求的结果、需求及明确表达的动机；具体待办和约定归计划与承诺，不从一次行动推定长期目标，但多次可以推测。单纯的喜恶、兴趣、习惯、偏好不属于本维度",
    4: "资源禀赋与生活条件：B自身拥有或可支配的经济、物质、健康、时间等条件及限制；区分长期条件与阶段变化，具体情境施加的外部限制归环境约束。只收录正文明确描述的资源、可用条件或自身限制；单纯的经历、知识、行为方式不收录",
    5: "重要计划与承诺：B需要跟进的关键计划或明确承诺，其履行、变更会显著影响目标推进、资源安排或他人事务；必须要保证重要性，不因出现计划措辞、具体日期或日常待办就入选。排除普通行程、例行加班和泛泛观望，除非有上述实质影响。完成、延期和取消关联对应事项，区分意向与兑现。",
    6: "社会关系网络：具体人物与B之间的关系、互动模式或关系变化。",
    7: "个人外部世界：提取与B生活有关的场所、人物与组织、设施资源、制度规则、信息与宏观环境的具体事实。正文须明确给出环境对象及其属性、关系、运行条件或状态变化；仅有话题提及、个人评价，未给出具体环境事实时不收录。B自身状况、主观倾向或活动过程不单独整理；其中明确建立或改变的外部环境事实可以提取。",
    8: "兴趣与偏好：B对具体对象、活动、方式或相处体验的喜恶、兴趣和选择倾向。保留偏好对象、方向及适用条件；不将一次行为或评价推广为长期偏好，不将他人的偏好归给B。输入中的peer是来源聊天对方编号，仅正文明确指向聊天对方时用于标识偏好对象。",
}

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {
                                "anyOf": [{"type": "integer", "enum": list(DIMENSIONS)}, {"type": "string"}]
                            },
                        },
                    },
                },
                "required": ["id", "fields"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

ATTRIBUTE_HIERARCHY_SCHEMA = {
    "type": "object",
    "properties": {"items": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "topic": {"type": "string"},
                "attr": {"type": "string"},
            },
            "required": ["id", "topic", "attr"],
            "additionalProperties": False,
        },
    }},
    "required": ["items"],
    "additionalProperties": False,
}

ATTRIBUTE_HIERARCHY_RULES = """将人物身份画像的属性名整理为上层主题和归一属性名。
输入id是属性编号，attr是原属性名，不包含具体事实。
根据属性描述的生活领域生成简洁中文主题，不预设类别列表；相关属性可归入同一主题，只有含义等价的属性才统一名称。
保留属性名中的对象、阶段和范围区别；含义不明确时保留原名，不推测取值。例如“本科专业”和“研究方向”可属于同一主题，但不能合并。
每个属性只归入一个最直接的主题，同一个归一属性名使用同一个主题。
只输出JSON对象：{"items":[{"id":1,"topic":"主题名","attr":"归一属性名"}]}。
items覆盖每个输入id且各一次；topic是上层浏览主题，attr是归一后的属性名。
"""


def attribute_hierarchy_prompt(attributes: list[dict]) -> str:
    return ATTRIBUTE_HIERARCHY_RULES + "\n属性：\n" + json.dumps(attributes, ensure_ascii=False, separators=(",", ":"))


PREFERENCE_ATTRIBUTE_SCHEMA = {
    "type": "object",
    "properties": {"items": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "attrs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "attrs"],
            "additionalProperties": False,
        },
    }},
    "required": ["items"],
    "additionalProperties": False,
}

PREFERENCE_ATTRIBUTE_RULES = """为人物B的每条偏好画像生成简洁中文属性名，用于后续分组归并。
输入id是记忆编号，content是正文；peer是来源聊天对方编号，仅正文明确指向对方时用于识别对象。
属性名描述偏好的信息类别，具体对象、喜恶方向及条件留在正文；换成该类别的其他取值时属性名应保持不变。例如喜欢或不喜欢甜食均可使用“口味偏好”。按含义统一同义属性名，不预设类别列表。
只提取正文支持的B的偏好属性，不把他人偏好或明确的玩笑当作B的偏好。
只输出JSON对象：{"items":[{"id":"输入编号","attrs":["属性名"]}]}。
items覆盖每个输入id且各一次；attrs是该条记忆涉及的不重复属性名列表，无可用偏好时为空。
"""

CLASSIFY_RULES = """你负责将已有人物B的画像和事件记忆分类到本次给定的维度，并为每条记忆提出中文属性名。
目标是保留对当前维度的人物建模有实际作用的信息，不追求覆盖全部输入；普通操作和零散细节若不影响对B的个人特征、现实条件或重要事项的理解，则不收录。
输入是已抽取记忆，id是记忆编号，content是正文；可选的event_time是原抽取的事件时间，status是计划或目标状态。
逐一对照本次维度定义，判断记忆事实是否直接提供了该维度描述的信息；匹配后为这部分信息生成属性名。
属性名用简洁中文描述信息类别，不限定词表；取值是该类别下的具体内容。自检：换成其他可能取值时，属性名应保持不变。例如“人工智能”和“机械工程”均使用“专业或研究方向”。
明确的玩笑、调侃内容不分类；同条记忆中的非玩笑事实仍可分类。
同一条可支持多个维度或属性，但每个归类都须有正文中的具体信息支持；各维度独立判断，没有匹配维度时fields为空数组，不为覆盖维度而凑结果。
只输出JSON对象：{"items":[{"id":"输入记录编号","fields":[[1,"职业"]]}]}。
items覆盖每个输入id且各一次；id原样复制。fields中的每项是[维度数字ID,中文属性名]。
"""

SUMMARY_CONTENT_RULES = """将已有记忆整理为给定维度下的属性与事实。
目标是保留对当前维度的人物建模有实际作用的信息，不追求覆盖全部输入；普通操作和零散细节若不影响对B的个人特征、现实条件或重要事项的理解，则不收录。
输入：原记忆的id是记忆编号，content是正文；可选的sample_time是来源聊天时间，event_time是事件时间，status是计划或目标状态。
仅归纳本次候选属性涉及的信息，排除明确的玩笑、调侃内容，不将其改写为事实。
归纳步骤：
1. 统一属性名：attr用简洁中文描述信息类别，value写该类别下的具体内容。自检：换成其他可能取值时，attr应保持不变；不能把具体取值加上“身份、状态、方向”等后缀当作属性名。例如“人工智能专业方向”改为“专业或研究方向”，人工智能写入value。按正文含义合并同义属性名，不同属性分别保留，不因都描述身份就统一成“身份”。
2. 整理取值：在保留独立事实、必要条件和重要变化的前提下尽量合并；同一属性的同义取值、同一事项的互补信息集中表述，不因措辞或候选组不同而重复输出，也不因主题相同就混合不同对象或事项。不同取值按当前维度的时间规则处理。需要选取最新状态时默认比较sample_time，正文或event_time明确指向历史或未来的记录不覆盖现状；仅在要求取最新但无法判定互斥值先后时，在同一条value中列出冲突值并注明“无法确定最新值”。
3. 保留依据：value写明主体、对象及必要时间和条件。source_ids汇集支持所保留事实的全部输入来源，不只选代表性来源。

结果字段：facts是事实列表，无符合条件的信息时为空；attr是中文属性名，value是具体事实，source_ids是支持该事实的原记忆编号，只能引用本次输入提供的来源。
"""

DIMENSION_TIME_RULES = {
    1: "保留身份候选：同一属性的同义取值合并，不同取值分别输出，包括互斥值和新旧值。每条value只写一个候选取值，保留其时间和条件，不将所有候选都写成当前身份；source_ids汇集支持该取值的全部来源。",
    3: "保留仍有效的需求、动机与目标。新目标明确替代旧目标时更新，明确完成或放弃的目标不再写成当前追求；不同目标可并存，不因出现较新的目标或旧目标未再提及就删除旧目标。",
    4: "同一资源、对象和统计范围保留最新状态，不同资产或资源分别保留。对象或范围不能对齐时分开表述，不将不同日期或对象的数值拼成范围、总量或包含关系。",
    5: "依据具体目标、对象和时间关系识别同一事项，将其计划、准备和进展合并，保留最新进展及必要变更，包括完成、延期和取消。不同对象或不同轮次的事项分别保留。",
    6: "peer是本次全部来源记忆的聊天对方编号，不代表正文中的所有人物。value中的关系对象须明确：正文指向聊天对方时写peer编号，第三人保留姓名或关系称谓；指向不明时保留不确定性，不归到peer名下。同一关系对象的关系状态取最新，有意义的互动经历与关系变化保留其时间；不同人物分别整理，不用一次互动覆盖整段关系，也不将历史互动写成持续现状。",
    7: "同一环境或外部条件保留最新状态，不同场景分别保留。临时地点、故障或限制保留适用时间，不因它是最后一条记录就视为持续现状。",
    8: "同一对象、方向和适用条件下的同义偏好合并，不同对象或条件分别保留。偏好明确改变时保留变化前后及时间，不因旧偏好未再提及就删除；相反表述无法确定先后或适用条件时保留冲突，不按出现次数选取。针对具体人物的偏好保留对象，不推广为对所有人的偏好。",
}

SUMMARIZE_RULES = (
    SUMMARY_CONTENT_RULES
    + """
只输出JSON对象：{"facts":[{"attr":"职业","value":"B从事算法研发工作。","source_ids":["输入编号"]}]}。
"""
)

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "attr": {"type": "string"},
        "value": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["attr", "value", "source_ids"],
    "additionalProperties": False,
}
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": FACT_SCHEMA},
    },
    "required": ["facts"],
    "additionalProperties": False,
}
BATCH_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, **SUMMARY_SCHEMA["properties"]},
                "required": ["id", "facts"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


def batch_summary_prompt(tasks: list[dict]) -> str:
    reduced = tasks[0]["stage"] == "reduce"
    groups, memories = [], {}
    for i, task in enumerate(tasks):
        group = {"id": i, "attrs": task["attrs"], "source_ids": task["source_ids"]}
        if reduced:
            group["facts"] = task["rows"]
        else:
            memories.update((row["id"], row) for row in task["rows"])
        groups.append(group)
    payload = {"groups": groups}
    if not reduced:
        payload["memories"] = list(memories.values())
    rules = """本次包含多个候选属性组。groups中id是组编号，attrs是候选属性，source_ids是该组候选来源；memories是各组共用、按id去重的原记忆。
若组内提供facts，则它们是该组分批归纳结果，合并时保留原source_ids。
同次请求内跨组统一同义属性、合并重复事实；依据正文选择支持结论的来源，可引用其他组的来源，不得引用输入外的编号。每条合并事实只放入一个相关组，其余组不重复输出；组内仍有其他事实则保留，全部并入其他组时facts为空数组。覆盖所有组id且各一次。
只输出JSON对象：{"groups":[{"id":0,"facts":[{"attr":"中文属性","value":"具体取值","source_ids":["原记忆编号"]}]}]}。
"""
    return (
        SUMMARY_CONTENT_RULES
        + f"\n当前维度{tasks[0]['dim']}的时间规则：{DIMENSION_TIME_RULES[tasks[0]['dim']]}\n"
        + rules
        + f"\n维度 {tasks[0]['dim']}: {DIMENSIONS[tasks[0]['dim']]}\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def classify_prompt(records: list[dict], dimensions: tuple[int, ...]) -> str:
    if dimensions == (8,):
        return PREFERENCE_ATTRIBUTE_RULES + "\n记忆：\n" + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    definitions = "\n".join(f"{dim}: {DIMENSIONS[dim]}" for dim in dimensions)
    return (
        CLASSIFY_RULES
        + "\n本次维度：\n"
        + definitions
        + "\n记忆：\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )


def summarize_prompt(
    dim: int, attrs: list[str], records: list[dict], *, reduced: bool = False, peer: str | None = None,
) -> str:
    extra = "\n本轮输入是分批归纳的facts；合并时保留原source_ids，不用分批编号替代来源。\n" if reduced else ""
    if peer is not None:
        extra += "\npeer：" + json.dumps(peer, ensure_ascii=False) + "\n"
    return (
        SUMMARIZE_RULES
        + f"\n当前维度{dim}的时间规则：{DIMENSION_TIME_RULES[dim]}\n"
        + extra
        + f"\n维度 {dim}: {DIMENSIONS[dim]}\n候选属性："
        + json.dumps(attrs, ensure_ascii=False)
        + "\n输入：\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )
