"""Checklist ownership; extraction groups may serve several style categories."""

PROTOCOL_VERSION = "style-taxonomy-v5"
CATEGORIES = {
    "lexical_choice": "词汇选择 Lexical Choice",
    "syntactic_style": "语法结构与句式 Grammatical & Syntactic Style",
    "orthographic_style": "书写与副语言形式 Orthographic & Paralinguistic Style",
    "register_style": "整体语体 Register",
    "pragmatic_style": "语气、立场与交际基调 Pragmatic Tone & Stance",
    "affective_style": "情绪表达 Affective Style",
    "rhetorical_devices": "修辞手法 Rhetorical Devices",
    "discourse_style": "篇章组织与信息展开 Discourse Organization & Information Elaboration",
    "interactional_style": "对话互动与消息节奏 Interactional Style",
    "contextual_style": "情境比较层 Contextual Style Adaptation",
}
# Stable merged IDs retain every source number; category membership follows the checklist.
CATALOG = {
    "001": ("高频词：高频词 Top-K、核心常用词", "lexical_choice"),
    "002": ("功能词 / 虚词：的、了、着、就、还、都、而、但是、然后、嘛、呀、呢、吧等", "lexical_choice"),
    "003": ("内容词偏好：哪类实词更常用", "lexical_choice"),
    "004,006": ("个人惯用表达：偏好措辞与口头禅", "lexical_choice"),
    "005": ("回避词：很少/从不用哪些词", "lexical_choice"),
    "007": ("网络流行语：yyds、emo、破防、绝绝子等", "lexical_choice"),
    "008": ("黑话 / 圈内词：特定社区表达", "lexical_choice"),
    "009": ("专业术语：程序员、学术圈等职业语言", "lexical_choice"),
    "010": ("缩写：yyds、xswl、tbh、btw 等", "lexical_choice"),
    "011": ("中英夹杂：“update 一下”“push 一把”“debug 下”", "lexical_choice"),
    "012": ("方言词：东北、四川、河南、粤语等", "lexical_choice"),
    "013": ("自造词：个人创造或非常规 token", "lexical_choice"),
    "016": ("叠词：看看、等等、慢慢、试试", "lexical_choice"),
    "017": ("人称代词：我/你/他/我们/咱们比例", "lexical_choice"),
    "018": ("指示代词：这、那、这个、那个", "lexical_choice"),
    "020": ("词汇丰富度：TTR、MTLD、HD-D", "lexical_choice"),
    "021": ("平均词长：中文可看平均字数/词", "lexical_choice"),
    "022,288": ("高频集中程度：Top-50/Top-100 占全部 token 的比例", "lexical_choice"),
    "023": ("Zipf 特征：rank-frequency 曲线、Zipf slope", "lexical_choice"),
    "024,289": ("n-gram 习惯：1/2/3/4-gram，个人常用短语和搭配", "lexical_choice"),
    "245": ("粗口使用", "lexical_choice"),
    "178-179": ("网络梗使用：表达、出现率与密度", "lexical_choice"),
    "181": ("梗年代", "lexical_choice"),
    "025,027-030": ("词性分布与密度（POS 比例）", "syntactic_style"),
    "026": ("名词化倾向", "syntactic_style"),
    "031-032": ("实词 / 虚词比例；词汇密度", "syntactic_style"),
    "033": ("各词性的句中位置分布", "syntactic_style"),
    "034": ("句首/句尾特定词性偏好", "syntactic_style"),
    "035": ("功能词在前 30% / 后 30% 的密度", "syntactic_style"),
    "036": ("时态、体貌倾向——英语文体评价里常用，中文可以弱化处理", "syntactic_style"),
    "163": ("强度副词：非常、太、巨、特别、稍微……", "syntactic_style"),
    "037-040": ("句长：字符数 / 词数、平均句长、句长方差、长短句分布", "syntactic_style"),
    "041": ("单句 / 复句比例", "syntactic_style"),
    "042": ("从句比例", "syntactic_style"),
    "043,060": ("句法完整度；短省略句，例如“吃了？”、“在？”", "syntactic_style"),
    "044": ("是否习惯完整主谓宾", "syntactic_style"),
    "045": ("是否显式带主语（原：主语省略）", "syntactic_style"),
    "046": ("谓词是否带宾语（原：宾语省略）", "syntactic_style"),
    "047": ("是否使用谓语式表达（原：谓语/判断词省略）", "syntactic_style"),
    "048": ("分句是否显式连接（原：连接词省略）", "syntactic_style"),
    "049": ("上下文依赖型省略", "syntactic_style"),
    "050": ("代词替代程度", "syntactic_style"),
    "051": ("陈述句", "syntactic_style"),
    "052,211": ("疑问句形式与使用比例", "syntactic_style"),
    "053": ("感叹句", "syntactic_style"),
    "054": ("祈使句", "syntactic_style"),
    "055": ("反问句", "syntactic_style"),
    "056": ("设问句", "syntactic_style"),
    "057": ("倒装句", "syntactic_style"),
    "058-059,062": ("前置成分：话题前置、宾语前置及位置分布", "syntactic_style"),
    "061": ("Subject–Verb–Object 顺序", "syntactic_style"),
    "063": ("主谓距离", "syntactic_style"),
    "064": ("动宾距离", "syntactic_style"),
    "065": ("状语—谓词距离", "syntactic_style"),
    "066": ("补语—谓词距离", "syntactic_style"),
    "067": ("预置状语数量", "syntactic_style"),
    "068-069": ("依存关系分布与各关系比例", "syntactic_style"),
    "070,290": ("Dependency Position Distribution", "syntactic_style"),
    "071": ("句法树深度", "syntactic_style"),
    "072": ("平均分支因子", "syntactic_style"),
    "073": ("平均依存距离", "syntactic_style"),
    "074": ("最大依存距离", "syntactic_style"),
    "075": ("Structural Skeleton / 句法骨架", "syntactic_style"),
    "076": ("POS / Dependency sequence n-gram", "syntactic_style"),
    "078": ("逗号使用率", "orthographic_style"),
    "079": ("句号使用率", "orthographic_style"),
    "080": ("问号使用率", "orthographic_style"),
    "081": ("感叹号使用率", "orthographic_style"),
    "082": ("省略号使用率", "orthographic_style"),
    "083": ("波浪号 `~`", "orthographic_style"),
    "084": ("冒号 / 分号", "orthographic_style"),
    "085": ("连续标点：`？？`、`！！！`、`?!`", "orthographic_style"),
    "086": ("完全不用标点", "orthographic_style"),
    "087": ("用空格代替标点", "orthographic_style"),
    "088": ("用换行代替逗号/句号", "orthographic_style"),
    "089": ("标点总密度", "orthographic_style"),
    "090": ("各类标点的相对分布", "orthographic_style"),
    "091": ("句末标点习惯", "orthographic_style"),
    "092": ("数字使用习惯", "orthographic_style"),
    "093": ("大小写习惯（中英混合时）", "orthographic_style"),
    "094": ("特殊符号", "orthographic_style"),
    "095": ("字符表情：XD、orz、:)", "orthographic_style"),
    "014": ("错别字：错字、同音字、漏字", "orthographic_style"),
    "097-098": ("语气助词与叹词的使用偏好", "orthographic_style"),
    "099-105": ("文本副语言表达及语境功能", "orthographic_style"),
    "106-107,109": ("连续重复形式与用途", "orthographic_style"),
    "108": ("拟声词的使用偏好", "orthographic_style"),
    "110": ("书面拉长音的使用方式", "orthographic_style"),
    "111": ("Emoji", "orthographic_style"),
    "112": ("Emoji 类别偏好", "orthographic_style"),
    "113": ("Emoji 使用密度", "orthographic_style"),
    "114": ("Emoji 出现间隔", "orthographic_style"),
    "115": ("Emoji 通常出现在句首/句尾", "orthographic_style"),
    "116": ("字符表情 vs Unicode Emoji 偏好", "orthographic_style"),
    "019,233": ("书面语 vs 网络聊天语", "register_style"),
    "234": ("文言化倾向", "register_style"),
    "237": ("年龄代际语言", "register_style"),
    "244": ("互联网化程度", "register_style"),
    "117-119": ("命题确定性分布", "pragmatic_style"),
    "120-127": ("主观立场限定与缓和策略", "pragmatic_style"),
    "128-131": ("礼貌行为与人际照顾方式", "pragmatic_style"),
    "132-134": ("敬意与自谦表达", "pragmatic_style"),
    "135": ("客套表达程度", "pragmatic_style"),
    "136-137": ("行动指向与约束强度", "pragmatic_style"),
    "138": ("立场坚持与表达直接性", "pragmatic_style"),
    "139": ("拒绝中的缓冲结构", "pragmatic_style"),
    "146-150": ("强调功能与实现方式", "pragmatic_style"),
    "246": ("攻击性语言", "pragmatic_style"),
    "174-175": ("调侃（含自嘲）", "pragmatic_style"),
    "176": ("冷幽默", "pragmatic_style"),
    "177": ("段子", "pragmatic_style"),
    "151": ("正向 / 负向情感词比例", "affective_style"),
    "152": ("情绪强度", "affective_style"),
    "153,160": ("情绪表达方式", "affective_style"),
    "158": ("抱怨密度", "affective_style"),
    "159": ("吐槽倾向", "affective_style"),
    "161": ("情绪基调", "affective_style"),
    "162": ("情绪波动范围", "affective_style"),
    "166": ("比喻", "rhetorical_devices"),
    "167": ("拟人", "rhetorical_devices"),
    "168": ("夸张", "rhetorical_devices"),
    "169": ("反讽表达（总体用法）", "rhetorical_devices"),
    "170": ("排比", "rhetorical_devices"),
    "metonymy": ("借代：不用事物本名，而用相关事物代替。", "rhetorical_devices"),
    "antiphrasis": (
        "反语：字面评价与语境支持的传达评价方向相反；作为169反讽的实现方式。",
        "rhetorical_devices",
    ),
    "171": ("对比", "rhetorical_devices"),
    "172": ("反差", "rhetorical_devices"),
    "173": ("谐音", "rhetorical_devices"),
    "140-144": ("元话语功能与位置", "discourse_style"),
    "145": ("认同、让步与反驳的组织方式", "discourse_style"),
    "183": ("Elaboration：展开 / 补充", "discourse_style"),
    "185": ("Cause：原因", "discourse_style"),
    "186": ("Evidence：证据", "discourse_style"),
    "187": ("Contrast：对比", "discourse_style"),
    "188": ("Purpose：目的", "discourse_style"),
    "189": ("Nucleus–Satellite 核心/附属关系", "discourse_style"),
    "190": ("RST 树深度", "discourse_style"),
    "191": ("修辞关系分布", "discourse_style"),
    "192": ("核心信息放前还是放后", "discourse_style"),
    "193": ("先结论后解释 vs 先铺垫后结论", "discourse_style"),
    "194": ("是否习惯提供理由", "discourse_style"),
    "195": ("是否喜欢举例", "discourse_style"),
    "196": ("是否进行论证", "discourse_style"),
    "197": ("是否展开背景", "discourse_style"),
    "198": ("是否补充细节", "discourse_style"),
    "256": ("是否提供额外建议", "discourse_style"),
    "263": ("信息压缩程度", "discourse_style"),
    "burst_pattern": ("连发模式", "interactional_style"),
    "210": ("是否追问", "interactional_style"),
    "212": ("一次追问几个问题", "interactional_style"),
    "214": ("追问的措辞与表达形式偏好", "interactional_style"),
    "218": ("长篇独白倾向", "interactional_style"),
    "219": ("是否主动延展话题", "interactional_style"),
    "220": ("是否经常等对方继续", "interactional_style"),
    "221": ("结束话题方式", "interactional_style"),
    "227": ("是否复述对方内容", "interactional_style"),
    "228": ("是否先回应情绪再回应事情", "interactional_style"),
    "229": ("是否用昵称", "interactional_style"),
    "231": ("是否习惯显式附和", "interactional_style"),
    "015": ("错字修正习惯：打错后是否马上补一句纠正", "interactional_style"),
    "264": ("工作群 vs 私聊", "contextual_style"),
    "265": ("家人 vs 朋友", "contextual_style"),
    "266": ("熟人 vs 陌生人", "contextual_style"),
    "267": ("老师/领导 vs 同辈", "contextual_style"),
    "268": ("正式场景 vs 日常场景", "contextual_style"),
    "269": ("忙碌 vs 空闲", "contextual_style"),
    "154-157,270": ("开心 vs 生气", "contextual_style"),
    "271": ("白天 vs 深夜", "contextual_style"),
    "272": ("紧急问题 vs 随便聊天", "contextual_style"),
    "273": ("句长变化", "contextual_style"),
    "274": ("礼貌变化", "contextual_style"),
    "275": ("个人惯用表达变化", "contextual_style"),
    "276": ("Emoji 变化", "contextual_style"),
    "277": ("信息密度变化", "contextual_style"),
    "278": ("解释长度变化", "contextual_style"),
    "279": ("调侃程度变化", "contextual_style"),
    "280": ("提问率变化", "contextual_style"),
    "281": ("连发程度变化", "contextual_style"),
    "282": ("词汇正式度变化", "contextual_style"),
}

# Rows merged only at report level; extractors keep these output keys. 288-290 were folded into
# 022/024/070 and 019 into 233; 270 has no extractor of its own.
PARTS = {
    "022,288": ("022",),
    "024,289": ("024",),
    "031-032": ("031", "032"),
    "037-040": ("037", "038", "039", "040"),
    "043,060": ("043", "060"),
    "070,290": ("070",),
    "019,233": ("233",),
    "174-175": ("174", "175"),
    "154-157,270": ("154", "155", "156", "157"),
}


def parts(key):
    return PARTS.get(key, (key,))


def source_ids(key):
    if key in ("metonymy", "antiphrasis", "burst_pattern"):
        return [key]
    values = []
    for part in key.split(","):
        bounds = [int(n) for n in part.split("-")]
        values.extend(f"{n:03}" for n in range(bounds[0], bounds[-1] + 1))
    return values


def extractor_for(key):
    if key in ("metonymy", "antiphrasis"):
        return "rhetorical_devices"
    if key == "burst_pattern":
        return "interactional_style"
    if key in PARTS:
        return extractor_for(PARTS[key][0])
    n = int(source_ids(key)[0])
    for low, high, module in (
        (1, 24, "lexical_choice"),
        (25, 36, "morphosyntax"),
        (37, 77, "syntactic_style"),
        (78, 95, "orthographic_style"),
        (97, 116, "paralinguistic_style"),
        (117, 150, "pragmatic_style"),
        (151, 163, "affective_style"),
        (166, 181, "rhetorical_devices"),
        (210, 231, "interactional_style"),
        (233, 246, "register_style"),
        (256, 263, "information_density"),
    ):
        if low <= n <= high:
            return module
    return None


def selected_categories(chapter):
    return list(CATEGORIES) if chapter == "all" else [chapter]


def extractors_for(categories):
    return list(
        dict.fromkeys(
            module
            for key, (_, category) in CATALOG.items()
            if category in categories
            for part in parts(key)
            if (module := extractor_for(part)) is not None
        )
    )


def metadata(key):
    name, category = CATALOG[key]
    result = {"name": name, "category": category, "source_ids": source_ids(key)}
    if key == "antiphrasis":
        result.update(parent="169", independent_score=False)
    return result
