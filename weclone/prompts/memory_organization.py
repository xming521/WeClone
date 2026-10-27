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

PROFILE_DIMS = (1, 3, 4, 5, 8)
# Privacy levels, from shareable with strangers to never leaving the chat it came from.
SENSITIVITY_LEVELS = ("public", "normal", "private", "sensitive")
# How a fact holds over time: never changes, latest known state, ended or replaced, tied to one point in time.
TIME_SCOPES = ("stable", "current", "superseded", "dated")


def dimension_name(dim: int) -> str:
    return DIMENSIONS[dim].split("：")[0]


DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["domains"],
    "additionalProperties": False,
}

TAG_DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "domain": {"type": "integer"}},
                "required": ["id", "domain"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
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
                            "type": "object",
                            "properties": {
                                "dim": {"type": "integer", "enum": list(DIMENSIONS)},
                                "domain": {"type": "integer"},
                                "attr": {"type": "string"},
                            },
                            "required": ["dim", "domain", "attr"],
                            "additionalProperties": False,
                        },
                    },
                    "sensitivity": {"type": "string", "enum": list(SENSITIVITY_LEVELS)},
                },
                "required": ["id", "fields", "sensitivity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "items"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["topics"],
    "additionalProperties": False,
}

PEER_SCHEMA = {
    "type": "object",
    "properties": {"relation": {"type": "string"}},
    "required": ["relation"],
    "additionalProperties": False,
}

DOMAIN_RULES = """你在整理 B 的记忆。查找记忆时，先判断要查的内容属于 B 生活的哪个方面，再到那个方面里找需要的记忆。
下面是从 B 的聊天记忆中抽出的检索标签及出现次数。请据此把 B 的生活划分为若干领域。

领域的要求：
1. 领域是 B 生活的一个方面，按"聊天时在聊什么"划分：只看一句话或一个问题，就能判断话题属于哪个领域。
2. 一个领域应能同时容纳身份、目标、条件、计划、偏好、经历等各类信息。只描述一类信息、不对应具体话题的标签（例如沟通偏好、时间安排、风险偏好）不单独成领域。
3. 按 B 的实际情况和话题粒度切分：B 经常聊、内容多的方面可以单独成领域，内容很少的方面并入相近领域。宽泛标签（例如人际关系、职业发展）覆盖面大，要按具体标签反映的话题拆分，不直接照搬为领域。领域数量 10 到 16 个，标签很少时可以更少。单个领域涵盖的内容不超过全部的四分之一。
4. 领域之间尽量互斥。姓名、年龄、籍贯这类基本身份信息归入"个人背景"类领域。
5. 领域名用简洁中文，不超过 10 个字。每个领域写一句描述，说明包括哪些内容，带上 B 在这个领域的具体对象或话题词，便于根据聊天内容识别，不超过 50 字。

只输出 JSON：{"domains":[{"name":"领域名","description":"描述"}]}

## 标签
格式：标签 出现次数
"""

TAG_DOMAIN_RULES = """你在整理 B 的记忆。下面是 B 的生活领域，以及从 B 的聊天记忆中抽出的检索标签。
把每个标签归入最相关的一个领域，填领域编号。宽泛标签填它最常对应的领域。
只输出 JSON：{"items":[{"id":标签编号,"domain":领域编号}]}。items 覆盖每个标签编号且各一次。
"""

PEER_RULES = """你在整理 B 的记忆。下面是从 B 与一位聊天对象的聊天中抽取的记忆，正文里的聊天对象编号就是这个人，sample_time 是来源聊天时间。
用一句话写出此人与 B 的关系，供给记忆分领域时参考：先写关系类型，例如家人、恋人、前任、暧昧对象、朋友、同学、室友、同事、导师、客户、网友；关系变过时按时间写出变化，例如"B的同学：2020年起同班，2023年起成为恋人"。
只依据正文，不推测。不写两人聊过什么，也不写此人的其他情况。不超过 60 个字。看不出关系时 relation 为空字符串。
只输出 JSON：{"relation":"关系"}
"""

SENSITIVITY_RULES = """sensitivity是隐私等级，按B愿意对谁说来判断，取下面四档之一：
- public：对陌生人也会说，例如口味、爱好、在读研究生、所在城市。
- normal：熟人之间会聊，例如学校和专业、求职进展、近期出行计划、和室友的日常。
- private：只对亲近的人说，例如感情状况、家庭矛盾、收入和花销、一般健康问题、对具体某人的负面评价、他人的私事。
- sensitive：只留在原来的聊天里，不告诉其他人，例如疾病诊断和心理健康、性取向、证件号和账号、精确住址、违法违规、他人托付的秘密。
拿不准时取较高的一档。
"""

PEER_DOMAIN_RULES = """正文里的P1这类编号是聊天对象，聊天对象列表给出此人与B的关系。B与聊天对象之间相处、沟通、陪伴、情感需求、交往边界、送礼这类关于两人关系本身的信息，按这层关系选领域，例如对恋人或前任的归入感情类领域，不因正文没写明关系就归入亲友类领域；在聊学习、工作、游戏、投资等具体事情时，仍按那件事选领域。
"""

CLASSIFY_RULES = (
    """你负责将已有人物B的画像和事件记忆分类到本次给定的维度，为每条记忆提出中文属性名，并选出所属的B的生活领域。
目标是保留对当前维度的人物建模有实际作用的信息，不追求覆盖全部输入；普通操作和零散细节若不影响对B的个人特征、现实条件或重要事项的理解，则不收录。
输入是已抽取记忆，id是记忆编号，content是正文；可选的event_time是原抽取的事件时间，status是计划或目标状态。
逐一对照本次维度定义，判断记忆事实是否直接提供了该维度描述的信息；匹配后为这部分信息生成属性名。
属性名用简洁中文描述信息类别，不限定词表；取值是该类别下的具体内容。自检：换成其他可能取值时，属性名应保持不变。例如“人工智能”和“机械工程”均使用“专业或研究方向”。属性名不含人名或P1这类聊天对象编号，涉及的人物留在正文。
每个归类再从领域列表中选一个领域，填领域编号。领域按这部分信息在聊天时属于什么话题来判断，同一话题下的身份、目标、条件、计划、偏好归入同一领域，不按维度区分。
"""
    + PEER_DOMAIN_RULES
    + """明确的玩笑、调侃内容不分类；同条记忆中的非玩笑事实仍可分类。
同一条可支持多个维度或属性，但每个归类都须有正文中的具体信息支持；各维度独立判断，没有匹配维度时fields为空数组，不为覆盖维度而凑结果。
每条记忆另给一个sensitivity，按正文判断，fields为空时也要给。
只输出JSON对象：{"items":[{"id":"输入记录编号","fields":[{"dim":1,"domain":2,"attr":"职业"}],"sensitivity":"normal"}]}。
items覆盖每个输入id且各一次；id原样复制。fields中的每项是一个归类：dim是维度数字ID，domain是领域编号，attr是中文属性名。
"""
    + SENSITIVITY_RULES
)

PREFERENCE_RULES = (
    """为人物B的每条偏好画像生成简洁中文属性名，用于后续分组归并，并选出所属的B的生活领域。
输入id是记忆编号，content是正文；peer是来源聊天对方编号，仅正文明确指向对方时用于识别对象。
属性名描述偏好的信息类别，具体对象、喜恶方向及条件留在正文；换成该类别的其他取值时属性名应保持不变。例如喜欢或不喜欢甜食均可使用“口味偏好”。按含义统一同义属性名，不预设类别列表。
只提取正文支持的B的偏好属性，不把他人偏好或明确的玩笑当作B的偏好。
每个属性再从领域列表中选一个领域，填领域编号。领域按偏好对象在聊天时属于什么话题来判断。
"""
    + PEER_DOMAIN_RULES
    + """每条记忆另给一个sensitivity，按正文判断，fields为空时也要给。
只输出JSON对象：{"items":[{"id":"输入编号","fields":[{"dim":8,"domain":2,"attr":"属性名"}],"sensitivity":"public"}]}。
items覆盖每个输入id且各一次；fields是该条记忆涉及的不重复偏好属性，dim固定为8，无可用偏好时为空数组。
"""
    + SENSITIVITY_RULES
)

TOPIC_RULES = """你在整理 B 的记忆。B 的记忆已经分成下面这些领域。查找记忆时，先选领域，再在领域里选主题，然后读主题下的条目。

现在把其中一个领域里的属性分成主题：
1. 按话题分：主题是"聊天时具体在聊什么"，例如"读博申请""实习与工作经历"。同一话题下的身份、目标、条件、计划、偏好放进同一个主题，不按维度分开。
2. 每个属性后面括号里是它的维度和记忆条数，"例"后面是其中一条记忆的开头。每个主题的记忆总数一般在 {min_memories} 到 {max_memories} 条之间：超过 {max_memories} 条的话题拆成几个并列主题，主题名要能互相区分；记忆很少的零散属性并入话题最接近的主题，不单独成主题。整个领域记忆很少时可以只有一个主题。
3. 含义相同、只是说法不同的属性放进同一个主题。
4. 主题名用简洁中文，概括主题里的全部属性，不超过 12 个字。
5. 每个属性只放进一个主题，全部属性都要放进主题。

只输出 JSON：{{"topics":[{{"name":"主题名","items":[属性编号]}}]}}
"""

TIME_SCOPE_RULES = """time_scope是时效，取下面四档之一：
- stable：不会随时间变，多久以前说的都仍成立，例如性别、家乡、出生年份、身高、已取得的学历和毕业院校、童年经历、已结束的经历本身（曾任某职务、曾在某单位工作），以及不随B处境变化的地点客观情况（某地气候、景点、车站位置）。
- current：会变的身份、状态、条件、目标、偏好或关系，是本次输入里这件事最新的说法，且没有结束的迹象，例如职业、任职阶段、住处、恋爱状态、持有的资产、仍在追求的目标。
- superseded：会变的状态已经结束，或被本次输入里更晚的说法明确取代，例如在读研究生之后已说毕业、试用期之后已说转正、已完成或放弃的目标。一个状态结束时，依附于它的状态和目标随之结束，例如毕业后，在读期间的实验室、课题组、学生职务和找实习这类求学阶段的目标；离职后，原单位的部门和岗位。与多次出现的说法矛盾、只出现一次或像玩笑的说法，不取代多次出现的说法，本身标superseded。后来没再提到不算结束，仍标current。
- dated：属于某个具体时间的计划、约定、安排、单次事件或临时情况，例如某天的考试、答应别人的事、一次出行。value写明时间。
"""

SUMMARY_CONTENT_RULES = (
    """将已有记忆整理为给定维度下的属性与事实。
目标是保留对当前维度的人物建模有实际作用的信息，不追求覆盖全部输入；普通操作和零散细节若不影响对B的个人特征、现实条件或重要事项的理解，则不收录。
输入：原记忆的id是记忆编号，content是正文，sensitivity是原记忆的隐私等级；可选的sample_time是来源聊天时间，event_time是事件时间，status是计划或目标状态。
仅归纳本次候选属性涉及的信息，排除明确的玩笑、调侃内容，不将其改写为事实。
归纳步骤：
1. 统一属性名：attr用简洁中文描述信息类别，value写该类别下的具体内容。自检：换成其他可能取值时，attr应保持不变；不能把具体取值加上“身份、状态、方向”等后缀当作属性名。例如“人工智能专业方向”改为“专业或研究方向”，人工智能写入value。按正文含义合并同义属性名，不同属性分别保留，不因都描述身份就统一成“身份”。
2. 整理取值：在保留独立事实、必要条件和重要变化的前提下尽量合并；同一属性的同义取值、同一事项的互补信息集中表述，不因措辞或候选组不同而重复输出，也不因主题相同就混合不同对象或事项。不同取值按当前维度的时间规则处理。需要选取最新状态时默认比较sample_time，正文或event_time明确指向历史或未来的记录不覆盖现状；仅在要求取最新但无法判定互斥值先后时，在同一条value中列出冲突值并注明“无法确定最新值”。
3. 保留依据：value写明主体、对象及必要时间和条件。source_ids汇集支持所保留事实的全部输入来源，不只选代表性来源。
4. 定隐私等级：sensitivity按value实际写出的内容判断，参考来源记忆的等级。value略去了来源里的私密细节时可以降级，例如来源是“B和女友吵架后去吃火锅”，事实“B喜欢吃火锅”可以是public；合并了更私密的来源时随之升级。
5. 定时效：time_scope按这件事会不会随时间变、写的是不是最新情况来判断。年龄、年级、工龄这类随时间推算的属性，用来源时间换算成出生年份、入学年份、入职年月等起点写入value，标stable。

结果字段：facts是事实列表，无符合条件的信息时为空；attr是中文属性名，value是具体事实，source_ids是支持该事实的原记忆编号，只能引用本次输入提供的来源，sensitivity是事实的隐私等级，time_scope是事实的时效。
"""
    + SENSITIVITY_RULES
    + TIME_SCOPE_RULES
)

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
只输出JSON对象：{"facts":[{"attr":"职业","value":"B从事算法研发工作。","source_ids":["输入编号"],"sensitivity":"normal","time_scope":"current"}]}。
"""
)

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "attr": {"type": "string"},
        "value": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "sensitivity": {"type": "string", "enum": list(SENSITIVITY_LEVELS)},
        "time_scope": {"type": "string", "enum": list(TIME_SCOPES)},
    },
    "required": ["attr", "value", "source_ids", "sensitivity", "time_scope"],
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
        group = {"id": i}
        if task.get("topic"):
            group["topic"] = task["topic"]
        group.update(attrs=task["attrs"], source_ids=task["source_ids"])
        if reduced:
            group["facts"] = task["rows"]
        else:
            memories.update((row["id"], row) for row in task["rows"])
        groups.append(group)
    payload = {"groups": groups}
    if not reduced:
        payload["memories"] = list(memories.values())
    rules = """本次包含多个候选属性组。groups中id是组编号，topic是该组所属主题（可缺省），attrs是候选属性，source_ids是该组候选来源；memories是各组共用、按id去重的原记忆。
若组内提供facts，则它们是该组分批归纳结果，合并时保留原source_ids，按合并后的value重新定sensitivity和time_scope。
同次请求内跨组统一同义属性、合并重复事实；依据正文选择支持结论的来源，可引用其他组的来源，不得引用输入外的编号。每条合并事实只放入一个相关组，其余组不重复输出；组内仍有其他事实则保留，全部并入其他组时facts为空数组。覆盖所有组id且各一次。
只输出JSON对象：{"groups":[{"id":0,"facts":[{"attr":"中文属性","value":"具体取值","source_ids":["原记忆编号"],"sensitivity":"normal","time_scope":"current"}]}]}。
"""
    return (
        SUMMARY_CONTENT_RULES
        + f"\n当前维度{tasks[0]['dim']}的时间规则：{DIMENSION_TIME_RULES[tasks[0]['dim']]}\n"
        + rules
        + f"\n维度 {tasks[0]['dim']}: {DIMENSIONS[tasks[0]['dim']]}\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def domain_block(domains: list[dict]) -> str:
    return (
        "\n## 领域\n"
        + "\n".join(f"{i}. {domain['name']}：{domain['description']}" for i, domain in enumerate(domains, 1))
        + "\n"
    )


def peer_block(relations: dict[str, str]) -> str:
    if not relations:
        return ""
    return "\n## 聊天对象\n" + "\n".join(f"{peer}：{relation}" for peer, relation in relations.items()) + "\n"


def domain_prompt(tags: list[tuple[str, int]]) -> str:
    return DOMAIN_RULES + "\n".join(f"{tag} {count}" for tag, count in tags)


def tag_domain_prompt(domains: list[dict], tags: list[tuple[int, str]]) -> str:
    return (
        TAG_DOMAIN_RULES
        + domain_block(domains)
        + "\n## 标签\n格式：标签编号 标签\n"
        + "\n".join(f"{i} {tag}" for i, tag in tags)
    )


def peer_prompt(peer: str, records: list[dict]) -> str:
    # Rules come first so every peer request shares the cached prefix.
    return (
        PEER_RULES
        + f"\n聊天对象编号：{peer}\n记忆：\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )


def classify_prompt(
    records: list[dict],
    dimensions: tuple[int, ...],
    domains: list[dict],
    relations: dict[str, str] | None = None,
) -> str:
    # Rules, the domain list and the peer relations come first so every batch of one person shares the cached prefix.
    if dimensions == (8,):
        rules, definitions = PREFERENCE_RULES, ""
    else:
        rules = CLASSIFY_RULES
        definitions = "\n本次维度：\n" + "\n".join(f"{dim}: {DIMENSIONS[dim]}" for dim in dimensions) + "\n"
    return (
        rules
        + domain_block(domains)
        + peer_block(relations or {})
        + definitions
        + "\n记忆：\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )


def topic_prompt(
    domains: list[dict], domain: int, attributes: list[dict], min_memories: int, max_memories: int
) -> str:
    current = domains[domain - 1]
    return (
        TOPIC_RULES.format(min_memories=min_memories, max_memories=max_memories)
        + domain_block(domains)
        + f"\n## 当前领域\n{current['name']}：{current['description']}\n"
        + "\n## 属性\n格式：属性编号 属性名（维度，记忆条数）例：记忆开头\n"
        + "\n".join(
            f"{row['id']} {row['attr']}（{dimension_name(row['dim'])}，{row['count']} 条）例：{row['example']}"
            for row in attributes
        )
    )


def summarize_prompt(
    dim: int,
    attrs: list[str],
    records: list[dict],
    *,
    reduced: bool = False,
    peer: str | None = None,
    topic: str | None = None,
) -> str:
    extra = (
        "\n本轮输入是分批归纳的facts；合并时保留原source_ids，不用分批编号替代来源，按合并后的value重新定sensitivity和time_scope。\n"
        if reduced
        else ""
    )
    if peer is not None:
        extra += "\npeer：" + json.dumps(peer, ensure_ascii=False) + "\n"
    if topic is not None:
        extra += "\n主题：" + json.dumps(topic, ensure_ascii=False) + "\n"
    return (
        SUMMARIZE_RULES
        + f"\n当前维度{dim}的时间规则：{DIMENSION_TIME_RULES[dim]}\n"
        + extra
        + f"\n维度 {dim}: {DIMENSIONS[dim]}\n候选属性："
        + json.dumps(attrs, ensure_ascii=False)
        + "\n输入：\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )
