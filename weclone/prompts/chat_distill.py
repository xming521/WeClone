import json

# B / A
CURRENT_STATE_SECTION = """4. 最近状态 current_state
   只抽取对后续对话有明显价值的当前状态。只用于短期、阶段性、会随时间失效的状态。
   例如：最近工作较忙、刚开完题、最近有些疲惫、最近在休假
"""

STATE_EXTRACT_PROMPT_TEMPLATE = """
你是一个用于个人记忆构建的用户画像抽取器。你的任务是从一段聊天记录（只包含A、B角色）中，抽取有聊天证据支持的“B”的画像记忆。

你需要抽取以下类型的信息：
1. 稳定事实 stable_fact
   长期或相对长期有效的信息。
   例如：家庭成员、宠物、长期身份、专业方向、常住地、长期习惯、长期拥有物、长期关系状态、长期能力/技能等。

2. 偏好 preference
   preference 需要区分
   - like：B喜欢、偏好、倾向、感兴趣、爱好、常选的事物
   - dislike：讨厌、排斥、避开、不想聊、不想做、不接受
   - emotional_preference：凡是涉及沟通、安慰、陪伴方式、反馈风格、情绪支持边界的偏好，优先标为 emotional_preference，而不是普通 like/dislike。 例如：希望被安慰、不喜欢被说教、想被认真倾听、不想对方诉苦、不想讨论某类敏感话题

3. 目标 goal
   B明确表达的目标、计划、愿望、打算、正在推进的任务。
   例如：想发论文、想去旅游、想换方向、想学某技能、想以后养猫。

{{CURRENT_STATE_SECTION}}

抽取原则：
以证据支持为先，不追求记忆数量或推断深度。允许还原上下文明确支持的省略、指代和关系；无法确定的部分不补全。
保持原文的主体、行动阶段、条件和不确定性；不把假设写成现实，不增加未经支持的时间、因果或普遍性。
不要把“B提到X”当成记忆。只有当 X 能揭示B的身份、关系、生活状态、偏好、目标或长期约束时，才值得抽取。
不抽取低信息量的默认事实。例如，“我爸妈都没用诺基亚”不应抽成“B有父母”或“B提到了父母”。

事实归属约束：
1. 写入任何记忆前，先判断这条信息的归属主体：它到底是 B 自己的身份、状态、偏好、目标或约束，还是 A、第三方、群体、公共背景、聊天话题、条件假设、转述内容、引用内容或语境说明。
2. 只有当原文能直接或稳定支持“B 自己处于/拥有/偏好/计划/受约束于 X”时，才可以写成 B 的记忆。

可检索性约束：
1. 每条画像记忆必须满足“脱离原聊天也能检索”：只看 content 本身，就能知道 B 在哪个对象、领域、关系、能力、偏好、目标或约束上有什么具体信息。
2. 允许的检索锚点包括：明确名称、稳定对象、可复用领域、具体关系、长期行为模式、明确偏好对象或长期约束。锚点必须来自原文或由原文稳定支持，不能靠读者回看上下文才能补全。
3. 未解析的指代词和泛名词不能当锚点。凡是 content 里仍需要依赖“这个/那个/当前/某个/相关/事情/内容/项目/材料/问题/方向/活动/东西/对象”等词才能成立，且原文不能还原具体指向，就不要抽取。
4. 不要把“态度、进展、流程、沟通、判断、权限、数量”等谓语或阶段包装成记忆。如果缺少可检索对象，即使 B 表达了喜欢、讨厌、一般、想做、做完、没权限、聊过，也不要落库。
5. 输出前自检：把 content 当作未来检索 query 时，是否能找到一类明确记忆？如果 query 只能搜到“当前话题/某个事项/相关内容”这类空泛结果，应丢弃。


抽取要求：
1. 只抽取 B 的信息，A 的内容只能作为上下文证据，不要抽 A 的事实。
2. 每条记忆必须给出 importance 和 confidence。
3. 对玩笑、反话、夸张、暧昧表达、角色扮演要谨慎，必要时降低 confidence 或不抽取。
4. 不要抽取无长期价值的临时动作，例如“去吃饭”“取外卖”“睡了”，除非它表达出稳定偏好、目标、长期约束{{CURRENT_STATE_ACTION_SUFFIX}}。不要抽取无意义寒暄、临时动作、语气词、玩笑梗、无后续价值的短期状态。
5. 不要为了凑数而抽取笼统记忆；低信息量、无明确对象、后续无法检索的问题应直接忽略。
6. 尽量将一个复杂事实拆成原子记忆，不要把多个无关事实合在一条里。
7. 不要输出重复记忆。如果同一事实在聊天记录中重复出现，只保留一条。
8. 如果出现事实变化、状态更新或前后不一致，例如B明确表达“以前如何、现在如何”，应分别抽取多条记忆。添加status字段（默认无），旧事实标记为 historical ，新事实标记为 active
9. 每条记忆还需要输出 tags 字段。tags 是字符串数组，用于高层语义检索。tags 必须是可独立检索的索引词，不是对 content 的摘要词，也不是临时概括词。只允许使用以下三类 tags：
   - domain_tag：稳定大领域，例如 学术科研、投资交易、健康状态、生活居住、出行旅行、工具软件、人际关系、娱乐游戏、情绪沟通、职业发展等。
   - topic_tag：原文中明确出现或可稳定归并的具体主题/对象，例如 选导师、选课、培养方案、实验室、论文写作、股票交易、作图工具等。
   - stable_facet_tag：稳定偏好、长期约束或行为模式，例如 风险偏好、沟通偏好、环境偏好、时间安排、长期习惯等。
   tag 也必须满足“脱离原聊天可独立检索”。不要把一句记忆的谓语、状态、数量、流程阶段、判断动作、临时关系、泛化对象或未解析指代词改写成 tag；这类 tag 即使看起来相关，也不适合作为检索索引。
   不合格 tag 类型：只描述动作或判断的 tag、只描述流程阶段的 tag、只有“相关/某个/当前/这个/那个”语义的 tag。
   如果找不到合格的具体主题 tag，只输出 domain_tag，或者直接不输出这条低价值记忆。每条记忆建议输出 1 到 3 个 tags。

重要程度 importance 取值：
1：低，有一定生活细节价值，但不核心。
2：中，对后续回复、关系理解或长期画像有帮助。
3：高，影响B目标、关系、长期偏好、长期约束或明显反复出现的问题。
4：极高，核心身份、核心目标、强烈偏好、长期边界或高风险相关信息。

置信度 confidence 取值：
confidence 只衡量该记忆是否被当前聊天记录充分支持，不衡量重要程度、长期价值或是否敏感。
2：中等支持，需要明显推断。该画像信息不是直接明说，而是由上下文线索、行为描述、关系称呼、任务描述等推理得出。
3：强上下文支持，需要少量补全。B没有完整说出标准事实句，但通过省略、指代、上下文承接等少量推断，可以基本确定该画像信息。
4：直接支持，无任何歧义。B以认真、肯定、非假设、非玩笑、非转述的方式明确表达该画像信息；或 B 明确确认了 A 对该信息的描述。

只输出 JSON，不要输出解释文字。没有可抽取记忆时输出 {"memories": []}。
JSON Schema：
{
  "memories": [
    {
      "type": "{{STATE_MEMORY_TYPE_ENUM}}",
      "preference_type(optional)": "like | dislike | emotional_preference",
      "content": "记忆内容",
      "tags": ["tag1", "tag2", "tag3"],
      "importance": 1,
      "confidence": 2,
      "status(optional)": "historical | active"
    }
  ]
}

待抽取聊天记录：
{{CHAT_JSON}}
"""

EVENT_EXTRACT_PROMPT = """
你是一个聊天记录事件抽取器。你的任务是从一段聊天记录（只包含A、B角色）中，抽取和“B”相关的事件。“事件”分两层：

1. 表层事件 surface_event：用一条摘要记录这段聊天显式围绕什么展开、聊了什么。它不是原子事件列表，而是对本段聊天表层内容的概括性总结。它可以概括：
- 主要在聊什么话题
- 围绕什么请求或信息交换展开
- 讨论了什么计划、结果、状态或关系变化
- 如果有多个零散话题，只保留和 B 最相关、最能代表这段聊天主旨的内容
每条聊天数据最多抽取一个 surface_event。尽可能的总结事件，普通日常流水账或短期生活事件也要总结，例如点了个外卖、吃了一顿普通饭、感冒了、睡了一觉、出门取东西等

2. 深层事件 inferred_event：根据聊天上下文可以较可靠推测出的现实中发生的事或者经历。只抽取根据聊天上下文可以较可靠推测出的、有现实意义或后续检索价值的事件。
不要为已经表达完整的 surface_event 再生成重复的 inferred_event。如果 inferred_event 只是把 surface_event 改写成“B正在/已经/需要/承诺……”之类的同义句，或者只是重复 B 已经明说的完整事件，就不要输出。
只有当推断能提供额外检索价值时，才生成 inferred_event。
例如：
- B已经经历过、做过、去过、吃过、接触过的事情。
- B去过日本、吃过某家有代表性的餐厅、养过宠物、玩过某类游戏、做过某个项目、曾经和某人交往过等。
不要过度脑补。不要把普通日常流水账都扩展成深层事件。
画像-事件边界：
长期画像、稳定事实、偏好、习惯、能力、长期关系、长期目标、泛化计划、稳定状态等属于画像记忆，不要输出为 inferred_event。

抽取原则：
以证据支持为先。深层事件不追求数量或推断深度。允许还原上下文明确支持的省略、指代和关系；无法确定的部分不补全。
保持原文的主体、行动阶段、条件和不确定性；不把假设写成现实，不增加未经支持的时间、因果或普遍性。
不要把“B提到X”或“B在聊X”自动当成现实事件。只有当 X 能揭示 B 参与、经历、计划、承诺、被影响的现实动作或状态变化，或对后续检索有明确价值时，才值得抽取为事件。
对于同时包含明说内容和推断补全的 mixed 事件，也按 inferred_event 的门槛判断：只有推断补全部分带来额外检索价值时才输出到 inferred_events；否则只保留 surface_events，不输出对应的 inferred_events。
surface_event 应是本段聊天的表层摘要，不要拆成多条原子事件。

事件内容可检索性约束：
1. 每条事件必须满足“脱离原聊天也能检索”：只看 surface_event 或 inferred_event 本身，就能知道 B 与哪个对象、领域、人物、地点、任务、承诺、经历、状态变化或受影响结果有关。
2. 合格事件必须至少包含一个可检索锚点。允许的锚点包括：明确名称、稳定对象、可复用领域、具体人物/关系、具体地点、具体任务、明确承诺对象、现实动作、状态变化、履约结果、受影响结果、可解析时间。锚点必须来自原文或由原文稳定支持，不能靠读者回看上下文才能补全。
3. 未解析的指代词和泛名词不能当锚点。凡是事件内容里仍需要依赖“这个/那个/当前/某个/相关/事情/内容/项目/材料/问题/方向/活动/东西/对象”等词才能成立，且原文不能还原具体指向，就不要抽取。
4. 不要把“聊到、提到、讨论、沟通、确认、判断、进展、流程、状态、数量、权限”等表层谓语或阶段本身当作事件锚点。如果缺少可检索对象，即使 B 表达了想做、做完、没权限、聊过、觉得一般，也不要输出为事件。

你要给每个事件添加事件类型字段 event_types 从以下几类中选择：
- daily_event：吃饭、睡觉、上课、上班、生病、取东西等普通生活经历或状态。
- sharing_event：分享链接、资料、工具、新闻、学习资源、表情包等信息或推荐。
- holiday_event：节日祝福、生日祝福、假期安排、节日聚餐、出游等。
- commitment_event：约饭、约见面、约定做某事、约一起出行、约之后帮忙或代办某事。
- fulfillment_event：已经赴约、完成代办、拒绝帮忙、延期、取消某个约定或委托。
- emotional_event：明显压力、孤独、焦虑、难过、崩溃、求安慰、风险表达等情绪状态或心理风险。
- other_event：不属于以上类型

每条事件还需要输出 tags 字段。tags 是字符串数组不固定，用于高层语义检索。tags 必须是可独立检索的索引词，不是对 content 的摘要词，也不是临时概括词。只允许使用以下两类 tags：
- domain_tag：稳定大领域，例如 学术科研、投资交易、健康状态、生活居住、出行旅行、工具软件、人际关系、娱乐游戏、情绪沟通、职业发展等。
- topic_tag：原文中明确出现或可稳定归并的具体主题/对象，例如 选导师、选课、培养方案、实验室、论文写作、股票交易、作图工具等。
tag 也必须满足“脱离原聊天可独立检索”。不要把一句事件的谓语、状态、数量、流程阶段、判断动作、临时关系、泛化对象或未解析指代词改写成 tag；这类 tag 即使看起来相关，也不适合作为检索索引。
不合格 tag 类型：只描述动作或判断的 tag、只描述流程阶段的 tag、只有“相关/某个/当前/这个/那个”语义的 tag。每条事件建议输出 1 到 3 个 tags。

抽取要求：
1. 只有 B 是事件参与者、承诺方、受影响方、请求接收方，或该事件对 B 有明确后续关系时才抽取。
2. 一个事件可以有多个事件类型和多个标签
3. 一段聊天最多抽取一个 surface_event，但可以抽取多个 inferred_event，二者分别输出到 surface_events 和 inferred_events。surface_events 若输出，只能包含一个对象。
4. surface_event 和 inferred_event 不是一对一关系，不要把它们强行合并在同一个 JSON 对象里。
5. 对玩笑、反话、夸张、暧昧表达、角色扮演要谨慎，必要时降低 confidence 或不抽取。
6. surface_event 不做原子化拆分；它应概括本段聊天聊了什么。inferred_event 才在确实存在多个不同现实事件时拆成原子事件，不要把多个无关现实事件合在一条里。
7. 输出 inferred_event 前先和 surface_event 对照：如果二者指向同一件事，且 inferred_event 没有新增可检索事实，只输出 surface_event。
8. 输出每条事件前自检：把 surface_event 或 inferred_event 当作未来检索 query 时，是否能检索到一类明确事件？如果只能搜到“当前话题/某个事项/相关内容/沟通进展”这类空泛结果，应丢弃。
9. 不要输出重复事件。如果同一事件在聊天记录中重复出现，只保留一条。
10. 只输出 JSON，不要解释。没有对应事件时，不输出对应顶层字段。事件对象内没有值的可选字段也直接省略，不要输出空字符串、空数组或 null。

重要程度 importance 取值：
1：低，普通生活流水账或简单话题，有一定记录价值但后续影响很小。
2：中，可能帮助后续回复或关系理解，例如近期日常、普通请求、一般分享、普通计划。
3：高，包含明确对象、时间、行动或承诺，对后续检索、提醒、关怀或关系延续有明显价值。
4：极高，涉及重要学业/工作/关系进展、明确承诺或变更、安全风险、强烈情绪，或可能持续影响 B 的重大事件。

置信度 confidence 取值：
confidence只衡量该事件是否被当前聊天记录充分支持，不衡量重要程度、长期价值或是否敏感。
2：中等支持，需要明显推断。事件不是原文直接陈述，而是由上下文线索、时间地点、请求答复、关系称呼或任务描述推理得出。
3：强上下文支持，需要少量补全。B没有完整说出事件句，但通过省略、指代、上下文承接等少量推断，可以基本确定事件发生、正在发生或将要发生。
4：直接支持，无明显歧义。B以认真、肯定、非假设、非玩笑、非转述的方式明确表达事件；或 B 明确确认了 A 对该事件的描述。

# 时间字段规则
surface_events 不需要输出聊天时间字段。
inferred_events 可以输出 event_time，只有你可以推理出深层事件的现实发生时间、计划时间时才输出 event_time。
event_time 不要只写相对时间词。如果能根据原始消息时间解析到具体日期，必须尽量解析成 normalized。如果不能解析，则不输出 time_normalized，只输出time_text。
event_time 格式：
{
  "time_text": "原文中的关于事件发生的时间表达，如 明天/前天/上周/假期/周末",
  "time_normalized": "YYYY-MM-DD"
}


输出 JSON：
{
  "surface_events": [
    {
      "event_types": ["commitment_event", "holiday_event","emotional_event"],
      "surface_event": "对这段聊天表层内容的一条摘要，概括主要聊了什么/围绕什么交流",
      "locations": ["相关地点"],
      "people": ["相关人物"],
      "tags": ["检索标签"],
      "importance": 0,
      "confidence": 0
    }
  ],
  "inferred_events": [
    {
      "event_types": ["other_event"],
      "inferred_event": "根据上下文推测的现实事件",
      "event_time": {
        "time_text": "原文中的关于事件发生的时间表达",
        "time_normalized": "YYYY-MM-DD"
      },
      "locations": ["相关地点"],
      "people": ["相关人物"],
      "tags": ["检索标签"],
      "importance": 0,
      "confidence": 0
    }
  ]
}
若没有任何事件，输出 {}。若只有 surface_events，不输出 inferred_events。
待抽取聊天记录可能包含 sample_time，表示这段聊天样本时间，可用于把“明天/昨天/上周”等相对时间解析为 time_normalized。
待抽取聊天记录：
{{CHAT_JSON}}
"""


MERGE_PROMPT_TEMPLATE = """你负责归并同一目标用户的候选画像记忆。输入内容中的 B 指该目标用户。保持事实归属，不得将其他人的信息改写为目标用户的信息。

目标：生成自包含、可检索的画像记忆，只依据输入事实，不新增事实。

归并规则：
1. 重复：多条记录表达同一事实时合为一条，用 source_ids 保留所有支持它的输入来源。
2. 包含：宽泛事实和具体事实都成立时，只有具体事实具备独立检索价值才保留为子记忆，用 parent_index 指向父记忆；否则吸收到同一条记忆中。
3. 拆分：不同对象、场景、偏好或行为模式有独立检索价值时分别保留。同一来源可以支持多条输出，每条只引用实际支持它的来源。
4. 冲突：先核对对象、场景和时间。不同场景下成立的事实可以并存；同一条件下的矛盾事实分别保留，不能强行合并。只有证据支持事实已被替代时才将旧事实标为 historical，无法判断是否仍成立时标 uncertain。
5. 时效：根据具体内容和证据时间判断，不按记忆类型预设有效期。只有内容支持时效判断时才填写；无法判断则省略。能判断具有阶段性但无法确定结束时间时只标 time_sensitive，不猜期限。记录较早、长期未再提及本身不代表过期。

输入字段：
records 是候选记忆数组；id 是来源标识；type 是上游记忆类型；content 是候选事实；time 是该记录的聊天样本时间，不一定是事实开始或结束时间。

输出字段：
canonical_memories 是归并后的记忆数组，每条包含：
- content：自包含的画像事实，写清目标用户与具体对象、领域、关系或场景的联系。保留原文的时间粒度和触发条件，不将模糊时间或事件条件补成具体日期，不将计划执行时间当作事实有效期。
- source_ids：支持这条事实的输入 records id 数组，不得引用组外来源。
以下字段均可省略：
- parent_index：仅在存在父子包含关系时填写父记忆在 canonical_memories 中的 0-based index；没有父记忆时省略或填 null。不得指向自身或形成循环。
- status：active=证据支持仍有效；historical=证据支持过去成立但不应代表当前；expired=证据支持已失效；uncertain=无法判断是否仍成立；none=不作有效性判断。不得仅根据类型或记录年龄判定 active 或 expired。
- time_scope：long_term=内容支持长期稳定；time_sensitive=内容支持具有阶段性或期限；historical_only=证据支持只适合历史复现。省略或留空表示未知，不代表永久有效。

source_decisions 仅记录没有被任何输出记忆引用的来源；全部来源均已保留时填 []。每项包含：
- source_id：未保留来源的输入 id，每个未保留来源恰好出现一次。
- decision：archived=内容仍有留档价值，但本次不生成规范记忆；discarded=内容不构成可用画像事实或缺乏支持，应丢弃。
- reason：基于输入说明未保留的原因。

只输出合法 JSON，不添加其他字段。每个输入 id 必须出现在至少一条记忆的 source_ids 中，或恰好一项 source_decisions 中，两者不能同时出现。来源时间、类型、标签和分数由程序根据 source_ids 回查，无需输出。

输出结构示例（可选字段仅在有依据时补充）：
{
  "canonical_memories": [
    {"content": "B...", "source_ids": ["1836:0", "5780:0"]}
  ],
  "source_decisions": []
}

待处理输入：
{{TASK_JSON}}
"""


def build_state_extract_prompt(*, include_current_state: bool = True) -> str:
    return (
        STATE_EXTRACT_PROMPT_TEMPLATE.replace(
            "{{CURRENT_STATE_SECTION}}",
            CURRENT_STATE_SECTION if include_current_state else "",
        )
        .replace(
            "{{CURRENT_STATE_ACTION_SUFFIX}}",
            "或有后续对话价值的当前状态" if include_current_state else "",
        )
        .replace(
            "{{STATE_MEMORY_TYPE_ENUM}}",
            "stable_fact | preference | goal | current_state"
            if include_current_state
            else "stable_fact | preference | goal",
        )
    )


STATE_EXTRACT_PROMPT = build_state_extract_prompt(include_current_state=True)


def build_window_extract_prompt(task: str, *, include_current_state: bool = True) -> str:
    header = """分别对多个独立 A、B 聊天样本执行抽取。
每个样本中的人物、事实、指代和时间只依据该样本，不得用其他样本补全，也不得跨样本合并或去重。下文的“当前聊天”“一段聊天”、空结果和数量限制均分别作用于每个样本。

"""
    if task == "state":
        template = build_state_extract_prompt(include_current_state=include_current_state)
        rules = template.split("只输出 JSON，不要输出解释文字。", 1)[0]
        schema = json.loads(template.split("JSON Schema：\n", 1)[1].split("\n待抽取聊天记录：", 1)[0])
        example = {"sample_id": "输入样本ID", "state_memories": schema["memories"]}
        field_rules = "每个结果对象包含 sample_id、state_memories 两个字段。state_memories 是该样本的画像数组，无画像时为 []。画像对象各字段遵循上述定义。"
        suffix = ""
    elif task == "event":
        rules = EVENT_EXTRACT_PROMPT.split("输出 JSON：\n", 1)[0].replace(
            "10. 只输出 JSON，不要解释。没有对应事件时，不输出对应顶层字段。事件对象内没有值的可选字段也直接省略，不要输出空字符串、空数组或 null。",
            "10. 没有对应事件时，在 event_memories 内省略对应事件数组。事件对象内没有值的可选字段直接省略，不输出空字符串、空数组或 null。",
        )
        schema = json.loads(EVENT_EXTRACT_PROMPT.split("输出 JSON：\n", 1)[1].split("\n若没有任何事件", 1)[0])
        example = {"sample_id": "输入样本ID", "event_memories": schema}
        field_rules = "每个结果对象包含 sample_id、event_memories 两个字段。event_memories 是该样本的事件对象，无事件时为 {}。其中 surface_events 为表层摘要数组，inferred_events 为深层事件数组；没有对应事件时省略该数组。surface_event、inferred_event 为对应事件内容；people、locations 是相关人物、地点数组，有值才输出。其他字段按上述定义。"
        suffix = "\ntime 是该样本的聊天时间，用于事件相对时间解析。"
    else:
        raise ValueError(f"Unknown extraction task: {task}")
    protocol = """\n最终输出协议：
只输出一个合法 JSON 对象，唯一顶层字段为 results。results 是逐样本结果数组，每个输入样本对应一个对象；sample_id 原样复制输入 ID，每个 ID 必须且只能出现一次。
"""
    return (
        header + rules + protocol + field_rules + "不得增加 result 包装字段。\n完整输出结构示例，数组元素按样本数量重复：\n"
        + json.dumps({"results": [example]}, ensure_ascii=False, separators=(",", ":")) + suffix
        + "\n独立样本输入：每个样本以单独一行 #ID 开始，ID 对应输出的 sample_id；到下一个 #ID 行或输入结束为止。\n"
        + "{{CHAT_SAMPLES}}"
    )
