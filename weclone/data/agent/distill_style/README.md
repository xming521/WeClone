# 风格抽取

```bash
python -m weclone.data.agent.distill_style
# 或
weclone-cli distill-style
```

只抽取本人回复的风格特征，不读入也不比较模型回复。

默认输入与画像抽取相同：`dataset/res_csv/agent/people/*.json`（跳过 manifest），复用画像的聊天读取规则。每条源记录作为一个完整对话段，抽取其中全部 B 发言，不会将一段拆成多个前缀样本。模型、并发、批量配置沿用现有设置。

可用 `--input-dir` 指定输入目录或文件，`--output-dir` 指定新输出目录；默认输出到 `dataset/res_csv/agent/distill/style/` 下的新运行目录，缓存放在相邻的 `style_cache/`。本地 LTP 模型默认位于 `models/LTPbase`，可用 `--ltp-model` 覆盖。`--limit` 在时间窗之后取前 N 个聊天段，`--no-llm` 只运行本地统计；默认语义抽样数量保持原来的 2000，`--llm-samples 0` 使用全部聊天段。时间窗（`--window-days`，默认 365）、统计单元和联系人分组见“统计单元、时间窗与联系人分组”。

# 指标说明

正式分类与 `WC-exp_风格评估清单_精简版.md` 一致：九类表层风格和独立的情境比较层，共 180 个清单条目。`taxonomy.py` 保存编号和唯一归属，`reporting.py` 组织输出；原抽取模块继续复用，同一抽取组可以服务多个类别。清单的抽取单元表按本包实际调用分组，并注明未实现、暂缓和停用的项；每次运行的覆盖情况以 `manifest.json` 为准。

| 分类 | `--chapter` | 主要共用抽取模块 |
| --- | --- | --- |
| 词汇选择 | `lexical_choice` | lexical_choice、rhetorical_devices、register_style |
| 语法结构与句式 | `syntactic_style` | morphosyntax、syntactic_style、affective_style |
| 书写与副语言形式 | `orthographic_style` | orthographic_style、paralinguistic_style；014 暂缓 |
| 整体语体 | `register_style` | register_style |
| 语气、立场与交际基调 | `pragmatic_style` | pragmatic_style、rhetorical_devices |
| 情绪表达 | `affective_style` | affective_style |
| 修辞手法 | `rhetorical_devices` | rhetorical_devices |
| 篇章组织与信息展开 | `discourse_style` | pragmatic_style 的 140–145、information_density 的 256 |
| 对话互动与消息节奏 | `interactional_style` | interactional_style、lexical_choice 的 015 |
| 情境比较层 | `contextual_style` | `154-157,270` 复用 affective_style（只有 155 已抽取），其余按实际覆盖标记 |

`--chapter all` 运行所有抽取组，每组只调度一次，共用解析器与缓存。

运行目录下的 `person/metrics.json` 是全体语料，`group-<chat_with_id>/metrics.json` 是各联系人组，均以合并编号为键，记录类别、原编号、覆盖状态、测量值和 `measurement_scope`。有组语料时另有 `group_variation.json`。`manifest.json` 列出类别、唯一条目、语料、时间窗、分组及抽取来源。`extractors/<模块>/` 保留原文、证据、各模块的统计及协议，供追溯。下面模块说明中的语料文件位于 `extractors/<模块>/<语料>/`，模块的 `manifest.json` 位于 `extractors/<模块>/`。

合并后的输出键为 `004,006`、`025,027-030`、`052,211`、`058-059,062`、`068-069`、`153,160`、`178-179`。属性保留在合并项内部，不重复输出独立指标；004,006 读取表达集合后会对相同表达去重。精简版清单另合并的 `022,288`、`024,289`、`031-032`、`037-040`、`043,060`、`070,290`、`019,233`、`174-175`、`154-157,270` 只在报告层合并：抽取模块的输出键和缓存不变，报告行的 `parts` 列出各部分的覆盖状态，测量值按部分编号分列。各部分状态一致时沿用该状态，全部已测量记 `extracted`，部分已测量记 `partial`，都未测量时取最接近完成的状态。288–290 已并入 022、024、070。184 背景关系与 232 整轮正式度已移除；197 背景展开和 233 整体语体保留；003 内容词分类、019 单独词汇正式度停止抽取，在模块覆盖表中标记 not_run，019 在报告中并入 `019,233`，由 233 测量。反语 `antiphrasis` 的父项为 169，不作独立综合分数。172 只观察文本有证据的表达落差，不推断人格或真实想法。

原来暂缓或尚无抽取器的项目仍显式标记 `deferred` / `not_implemented`。输出不合成总分。

## 运行

在 WeClone 根目录激活已有环境。入口顶部有默认路径和参数，命令行可以覆盖。

```bash
source .venv2/bin/activate
# 前 20 个聊天段验证全部类别
python weclone/data/agent/distill_style/run.py --chapter all --limit 20

# 只验证基础统计，不调用 LLM
python weclone/data/agent/distill_style/run.py --no-llm

# 少量聊天段验证真实标注链路，并复抽 2 条检查稳定性
python weclone/data/agent/distill_style/run.py --limit 20 --qa-samples 2

# 语法类；同样支持 --no-llm、--limit 和 --qa-samples
python weclone/data/agent/distill_style/run.py --chapter syntactic_style

# 发现个人惯用表达（004、006），默认读取同一批聊天；输出目录必须是新目录
python weclone/data/agent/distill_style/run.py discover --output /path/to/discovery

# 用发现的表达集合统计 004、006
python weclone/data/agent/distill_style/run.py \
  --frozen /path/to/discovery/frozen_expressions.json
```

改代码后运行 `pytest tests/test_distill_style.py`。它用合成聊天和一个总是失败的假 judge 跑完抽样、本地统计、QA、discover、`--frozen` 和批量请求，再与 `tests/tests_data/distill_style_snapshot.json` 比对每个输出文件和全部 LLM 请求。需要本地 `models/LTPbase`。有意改变输出或提示词时，用 `UPDATE_STYLE_SNAPSHOT=1` 重新生成快照，并检查报出的变化文件是否都在预期之内。假 judge 不返回有效标注，所以语义汇总只走到失败分支。

默认输出到 `dataset/res_csv/agent/distill/style/<时间>_<类别>`，所有运行新建目录，缓存单独放在 `dataset/res_csv/agent/distill/style_cache`。LLM 后端跟随 `settings.jsonc` 的 `agent_distill_args.llm_provider`，未配置时使用 `codex_exec`；模型、推理强度、超时和命令等读取 `codex_exec_args`，调用项目已有的 Codex CLI 客户端，失败不会转用 API。`--judge-model` 可覆盖模型；显式指定 `--llm-provider api` 时读取 `make_dataset_args`，此时才支持 `--base-url`，API key 不写入产物。运行不会安装包。`discover` 默认读取 `--input-dir` 的同一批聊天，也可用 `--training` 另给一份语料（SFT 消息格式或下文的规范格式）。未提供 `--frozen` 时 004,006 显式记为 `not_run`；`--no-llm` 下语义项同样记为 `not_run`。

默认 `--llm-samples 2000 --sample-seed 42`：按“联系人×月份”比例分配名额、按样本 ID 的固定哈希抽样，再给每个联系人组补足下限；没有时间或联系人的记录单独成层。`sampling.json` 保存选中 ID、各层与各组的人数和入选概率。语义统计是描述性样本统计，不将未抽中的记录记为 absent。`--llm-samples 0` 使用全量语义标注。

本地指标仍计算全量，保存在 `surface/<模块>/`；抽样语义结果在 `extractors/<模块>/`。汇总报告的 `measurement_scope` 按语料给出本地统计与语义统计各自的窗口数和非空消息数，语义项的词数等分母取同一抽样子集。LLM 最多合并 `--llm-batch-size 8` 条独立聊天输入；输入预算默认 16000 字符，输出估算预算默认 32000 字符，估算依据为提示词结构复杂度与输入证据量。超预算的单条任务独立发送。批量响应按任务 ID 拆分缓存，缺失或重复 ID 只重试受影响任务；语义结构修复仍沿用各模块的验证器。`llm/progress.json` 记录实际请求数、任务数、缓存命中、字符量及服务端返回的 token 用量，`joint_requests` 是跨模块请求数，`request_inputs` 是累计发送的聊天输入数（含重试）。完整批次响应保存在 `llm/batches/`。条数、字符预算与原有 LTP batch size、并发数是不同参数。

各模块同时运行，共用一个 LLM 客户端，所以 `codex_exec_args.request_interval_seconds` 的启动间隔和容量冷却对整次运行生效，`--workers` 仍是每个模块的并发数。LTP 解析一次只给一个模块。会发送别的模块请求的模块之间保持先后：语体在联合批次里顺带发送词汇和修辞的请求，所以词汇、修辞等语体完成后再跑；情感用副语言提示词重标笑声，所以排在副语言之后。后跑的模块直接读缓存，同一请求不会发两次。surface 本地统计与 LLM 模块同时进行。

批次按提交顺序填满，所以所有进入 LLM 队列的样本（逐窗口标注、词汇的话题分类、情感的回复前条件、004/006 的候选发现）都先按 `sample_id` 的固定哈希顺序打乱，种子是 `extraction.QUEUE_SEED`。输入文件按联系人和时间排列，不打乱时一批几乎都是同一联系人相邻的窗口。p0 默认 2000 段抽样按 8 段一批切，94% 的批只有一个联系人，打乱后每批平均 5.5 个联系人、5.6 个月份。客套程度、情绪强度、语体、独白这类相对判断会以同批样本为参照，同源批次会把这种参照偏差变成系统偏差。结果仍按原顺序写回，顺序只影响批次组成。批量协议因此升为 `style-chat-batch-v8`，同源批次留下的旧缓存键不再命中。

语体、词汇、修辞联合抽取时，输入为 A/B 纯文本，提取全部 B 发言，A 仅作为上下文，用分隔线区分对话段，不发送对话或消息编号，每段原文只发送一次，直接返回一份按类别组织的结果，提示词只列出本批请求的类别（“语体”“词汇”“修辞”之一或多个）。同一类别在一批内只能有一套规则，例如 015 是否适用会改变词汇规则，规则不同的任务分到不同批次。本批各段统一抽取提示词中定义的类别，不再逐段发送 fields、boundary，也不发送 tasks/module/input_id 映射；程序只接收内部待处理类别，其余类别不重复计入统计。输出 results 按输入顺序对应；数量不一致时整批拒收，避免漏项后错位。程序内部拆分到原有校验器和缓存；缺失类别只重试对应部分。引用原文由程序恢复内部消息定位，输入不暴露消息编号。批量参数不变。空格、换行保留本地数量及位置统计，停用其语境用途分类；字符表情的 095/116 及语体复用均不再调用 LLM。Unicode Emoji 的本地识别仍保留。

超长 LTP 输入优先按句末标点切分，单句仍超限时按 tokenizer 边界切分，合并时恢复原文与 token 偏移。`chunks` 保留切分位置，依存根保持独立，不虚构跨块依存关系；强制切开的长句若不构成单根树，句法结构统计按解析失败处理。

## 统计单元、时间窗与联系人分组

三种单元的定义随 `manifest.json` 的 `units` 保存：

| 单元 | 含义 |
| --- | --- |
| 消息（message） | 本人的一个回合，即分身一次生成的单位。默认聊天输入已把连续发送合并成一个回合。 |
| 气泡（bubble） | 消息内按换行切开的非空片段，近似一条微信消息。不带换行直接拼接的连发仍算一个气泡。 |
| 窗口（window） | 一条输入记录及其中全部本人消息。窗口长短由数据处理的切段规则决定，不是本人的属性。 |

主比例一律按消息报，字段名以 `message_` 开头。逐消息标注的项（语用、情感、副语言）用可判定的消息作分母，按窗口标注的项用可判定窗口里的非空消息作分母，命中数为 `hit_messages`。窗口级比例只作诊断，统一命名为 `window_*`（如 `window_occurrence_rate`、`valid_windows`、`hit_windows`），不能当作本人习惯直接使用。长度同时给出消息、气泡、窗口三级分位数（`length_summary`）。233 语体由 LLM 按窗口判断，因为单条短消息很少能看出语体。`window_proportions` 是窗口比例，`message_weighted_proportions` 按窗口内非空消息数加权，表示本人消息落在各语体中的比例。

命中率随回复长度上升：回复越长，越容易带上逗号、语气词或某种修辞。只看总比例时，回复长短不同的两组会显得风格不同。所以每条按消息计的比例旁边另存 `by_length`，按单条消息的非空白字符数分成 1–2、3–4、5–8、9–16、17–32、33+ 六档（全空白的消息记作 0 档），每档给出命中数、消息数和比例。各档消息数相加就是该比例的分母。本人的字数分布在 218 的 `message_length_counts` 和 `message_length_shares` 里。分档定义随 `manifest.json` 的 `length_buckets` 保存。233 语体是窗口标签，窗口越长越容易同时出现两种语体而判成 mixed，按单条消息长度分档看不出这一点，所以按整个窗口全部目标消息的非空白字符数分档：1–8、9–16、17–32、33–64、65–128、129+。窗口里每条非空消息在该窗口所在的档里各计一次，结果存在 `by_length.message_weighted_proportions` 的各标签下，各档相加等于 `message_label_counts` 和 `labeled_messages`。窗口分档定义随 `length_buckets` 的 `window_buckets` 保存。树深度、依存距离这类按句段计算的数值也不分档。

出现次数少的比例主要是随机波动，拿去比较像不像时会把噪声当成风格差别。所以每条带 `by_length` 的比例旁边另存 `reliability`，218 的 `message_long_rate` 和 `message_length_shares` 也有。记录给出命中数、消息数、窗口数，以及较少一侧结果出现的次数 `events`。较少一侧记在 `rare_outcome`：比例不超过 0.5 时是命中（occurrence），超过时是未命中（non_occurrence）。`interval` 是比例的 95% 区间，取两种区间中更宽的一端。一种是按窗口有放回重抽 1000 次的百分位区间，因为同一窗口里的消息不独立，话题和口头禅会扎堆。另一种是按消息的 Clopper–Pearson 精确区间，因为次数很少时重抽给不出可靠的上界。`effective_events` 是 `events` 除以设计效应（重抽方差与二项方差之比，下限为 1），命中越扎堆，有效次数越少。只有一个窗口时无法估计扎堆程度，`effective_events` 为空。`rare_upper` 是较少一侧比例的上界。据此分三类：

| 类别 | 条件 | 用法 |
| --- | --- | --- |
| `estimable` | `effective_events` ≥ 30 | 比例和区间可以直接用。该项内部以命中为分母的分布（形式、位置、功能等）也可以用，其中单个类别次数太少时只描述。 |
| `rare` | 不到 30，但 `rare_upper` ≤ 1% | 较少一侧几乎不出现：`rare_outcome` 为 occurrence 时是几乎不用，为 non_occurrence 时是几乎总是。只当上限用，比较时只查是否用多了，内部分布不用。 |
| `undetermined` | 其余，包括分母为 0 | 证据不够，只描述，不用来判断像不像。常见原因是适用的消息太少，或语义项只标了抽样的窗口。 |

阈值和重抽设置在 `stats.py` 顶部（`MIN_EVENTS`、`RARE_UPPER`、`CONFIDENCE`、`BOOTSTRAP_SAMPLES`、`BOOTSTRAP_SEED`），随 `manifest.json` 的 `reliability` 保存。记录保留了次数、有效次数和上界，换阈值时直接重新分类即可，不用重抽。运行目录的 `reliability.json` 汇总 `person` 语料的全部记录，按编号和比例路径列出，`classes` 是三类各有几条。各组语料的记录在各自的 `metrics.json` 里。233 语体只有按窗口长度分的 `by_length`，没有 `reliability`。011 的 `mixed_rate` 两者都没有。

需要 LLM 判断的比例，`reliability` 还记录 `unknown_messages`、`failure_messages`、`missing_share` 和 `bounds`。前两个是因窗口或消息判为 unknown、或标注失败而没进分母的目标消息数，`missing_share` 是它们在“分母加未判定”中的占比。`bounds` 是把这些消息全算未命中和全算命中时的比例，即 [命中数, 命中数+未判定数] ÷ (分母+未判定数)。两组的未判定比例差得多时，比较的是不同的子集，这时看上下界：差别在两端都成立才算。按窗口判断的项，整个窗口判为 unknown 时其中的非空消息都计入。窗口判为 present、但个别单元判不了的消息仍按未命中计，没有单独计入。失败也算未判定，因为它同样让分母变成另一个子集。三类划分只看已判定的消息，不受未判定数影响。

`--window-days` 默认 365，只保留最新一条记录之前 365 天内的窗口，0 表示使用全部历史。`--limit` 在时间窗之后取前 N 段。

联系人分组默认开启。时间窗内非空消息数不少于 `--min-group-messages`（默认 300）的联系人各自成为一个语料 `group-<chat_with_id>`，与全体语料 `person` 并列输出，`--no-groups` 关闭。`--contact-groups map.json` 可以把 `chat_with_id` 映射到关系标签（如 `{"chat_with_0016": "partner"}`），同一标签的联系人合并成一组，未映射的联系人仍各自成组。

语义抽样在 `--llm-samples` 预算之内完成。先按“联系人×月份”比例抽取 `person` 语料，它对本人全体窗口自加权，可直接当作全体估计。再在每个组内按同一哈希顺序补足到 `--group-floor`（默认 100 个窗口），补足的窗口只进入组语料。下限最多占预算的一半，组多时自动降低，实际值记在 `sampling.json` 的 `group_floor`。同一窗口出现在多个语料时只标注一次。`--qa-samples` 只在 `person` 语料上复抽。

有组语料时，运行目录另写 `group_variation.json`。它把各组每条按消息计的比例恢复成命中数和分母（有 `by_length` 时直接用分档计数），只检验期望命中和期望未命中都不少于 5 的组。参加检验的组都有分档计数时，在同一字数档内比较各组，用广义 Cochran–Mantel–Haenszel 检验。233 语体按窗口长度档比较。比值用实际命中除以期望命中，期望命中按该组自己的字数分布和参照比例算出。参照比例取 `person` 语料在各档的比例，`person` 在某档没有消息时改用各组合并的比例。这样，只因为和某个联系人聊得更长而多出来的命中，不会算成关系差异。没有分档计数的比例，以及本身就是长度的比例（`message_long_rate`、`message_length_shares`），仍用不分档的卡方齐性检验，原因写在 `length_adjustment`。分档时，不分档的比值和检验保留在 `unadjusted` 里。全部检验一起做 Benjamini–Hochberg 校正，再按较少一侧结果的组间最大/最小比值分类：

| 类别 | 条件 |
| --- | --- |
| `relationship_dependent` | 比值 ≥ 1.5 且 q < 0.01 |
| `uncertain` | 比值 ≥ 1.5 但 q ≥ 0.01 |
| `stable` | 比值 < 1.5 |
| `insufficient` | 可检验的组少于 2 个，或命中、未命中合计少于 30 |

一个编号下只要有一条比例属于 `relationship_dependent`，该编号就归为关系相关，画像里应按联系人或关系给出。全部可检验比例都是 `stable` 的编号可以作为本人的稳定特质，用 `person` 语料的值。`person` 的值逐条列出作参照，不参与检验。不分档时比值用 Jeffreys 平滑（(k+0.5)/(n+1)），分档时用（实际+0.5）/（期望+0.5），都只在可检验的组之间计算。

同一文件也是按字数和联系人控制特征时的目标表。每条比例下，`person.by_length` 是本人在各字数档的比例。各组的 `length_adjusted_ratio` 是（命中数+0.5）/（期望命中数+0.5），即按字数分布调整后该联系人相对本人的倍数，不可检验的小组也给出。某联系人在某档的目标比例等于本人该档比例乘以该联系人的倍数，超过 1 时取 1。稀有特征在组内再分档后计数太少，所以借用本人的分档比例。组内各档的原始计数在 `group-<id>/metrics.json` 的 `by_length` 里，消息多的格子可以直接用。

## lexical_choice 抽取模块

| 编号 | 实现 | 主要输出 |
| --- | --- | --- |
| 001 | LTP 词级频率、Top-100 | `lexical.words/top_words` |
| 002 | 以当前 POS 识别功能词 | `function_words/function_rate` |
| 001–024 共用 | 笑声、哭声、迟疑这类拟声串（`vocal_token`：一半以上非空白字符落在哈/呵/嘿/嘻/呜/嗷等声音字的连续重复里，或拉丁 hh/haha/emmm）不算词，不进词频、功能词、丰富度、词长、Zipf 和 n-gram，n-gram 在这里断开。被排除的词形和数量见 `excluded_vocal_tokens/excluded_vocal_forms`。这些形式由 099–105 测 | `lexical.excluded_vocal_*` |
| 003 | 停止抽取，不再发送实词候选 | `coverage.003=not_run` |
| 004、006 | `discover` 用 LLM 从聊天回复提出候选，程序核验跨会话复现后写入表达集合，`--frozen` 读入后统计 | `frozen_expressions.json`、`personal.json` |
| 007–010 | 网络用法、圈内词、专业术语/专名、缩写。007 不算 hh、emmm 和平台表情（归副语言），010 只算 nb、yyds 这类缩写，不算 hh 一类笑声写法、国庆/北邮这类已成常用词或专名的汉语缩略语和 ID/APP 这类通用英文缩写 | `semantic.007` 至 `010` |
| 011 | 原始中文/拉丁片段；LLM 仅输出实际中英混用的消息片段；纯中文/纯英文为 absent，英文只是软件、产品、人名地名等专名或 ID/APP/wifi 这类通用缩写/外来词时也为 absent，分母为有效非空目标消息 | `lexical.per_message.script`、`semantic.011.mixed_rate/switches` |
| 012 | 有明显地域色彩的方言用法，区分本人表达和引用/模仿；啥、咋这类全国口语通用的词不算 | `semantic.012.labels/label_events_per_100_words` |
| 015 | 同人紧邻两次发送之间的明确更正（同一回合相邻两行，或 `boundary=message` 时相邻两条消息），区分已确认错误与前文错误不确定。回合里有两行以上才调用，程序校验 source 在 evidence 的上一行 | `semantic.015` 的 `adjacent_sends`、`corrections_per_adjacent_send`、同回合更正数、可用秒数及覆盖量 |
| 016 | 有语法功能的叠词 | `semantic.016` |
| 017 | POS=r 与封闭人称代词集合交集，按完整词形计数 | `personal_pronouns/personal_pronoun_rate` |
| 018 | 当前用作指示代词的证据跨度 | `semantic.018` |
| 019 | 停止单独抽取，保留 233 整体语体，报告并入 `019,233` | `coverage.019=not_run` |
| 020 | 词级 TTR、双向 MTLD、HD-D，按等长块计算 | `lexical.diversity` |
| 021 | 非标点词的 Unicode 字符长度，保留数字/英文 | `word_lengths/mean_word_length` |
| 022 | Top-50/100 词频和 ÷ 全部词 token | `top_mass` |
| 023 | 全部词 token 的 rank–frequency 点、OLS 斜率、截距、R² | `lexical.zipf` |
| 024 | 独立词级 1–4 gram、汉字 1–5 gram | `word_ngrams/han_character_ngrams` |

每个语义项保留 `present/absent/unknown/na/failure` 数量、有效样本数、未知率、失败率、用法发生率、标签/表达 Counter 和每百词事件密度。发生率分母仅为 present+absent，同时保留全量覆盖率。主发生率为 `message_occurrence_rate`（命中消息 ÷ 可判定窗口的非空消息），`window_occurrence_rate` 只作诊断。`011.window_occurrence_rate` 指存在可抽取自然语言的比例，中英混用看按消息计的 `mixed_rate`。`personal.json` 的 004,006 同样以 `message_occurrence_rate` 为主，并给出每个表达的 `message_counts`。专业术语与专名、方言本人用法与引用、两种更正状态分别给出标签计数和密度，不混作同一种现象。

## 数据与统计口径

默认读取画像聊天文件，`--limit` 取前 N 个聊天段。只把可见的 A/B 聊天文本交给风格 judge，A 发言仅作前文。

换数据可使用 `--format canonical --input-dir records.jsonl`，单条格式如下：

```json
{
  "sample_id": "p0:conversation-id:segment-id",
  "person_id": "p0",
  "conversation_id": "conversation-id",
  "boundary": "response",
  "context": [{"message_id": "c0", "role": "user", "text": "可见前文"}],
  "messages": [{"message_id": "m0", "role": "assistant", "text": "原始回复"}]
}
```

sample_id 标识一个对话段，conversation_id 标识原会话，004,006 用它判断跨会话复现；person_id 是本人的 ID，不传给用法 judge。context 与 messages 按顺序拼成完整对话段，其中全部 assistant 发言都是抽取目标，user 发言只作前文。message_id 在单条记录内唯一。只有每条 messages 本身就是一次发送时，才用 `boundary="message"`，并按发送顺序提供，可各带 ISO 8601 `timestamp`。默认聊天输入是 `boundary="response"`：一条 message 是本人一个回合，`qa_generator` 合并连续发送时一律用换行连接，所以回合内的每个非空行（bubble）就是一次发送，086、088、091、015、114 都按行计算。真实消息内的换行（原始消息中约 0.12%）在合并后无法和发送边界区分。缺时间戳则只输出消息和行的间隔，不猜秒数。

LTP 的 POS 映射依据 [863 标签定义](https://ltp.ai/docs/appendix.html)：本实现把 p/c/u/e/o/d 作为广义功能词（含副词），n/nh/ni/nl/ns/nt/nz/v/a/i/j 作为内容词候选。这个分类是可复核的操作性口径，不是唯一语言学划分。人称代词用封闭语法类加 POS 条件，保留“我/我们/咱们”等词形；不会在“我们”内额外命中“我”。新语言或新解析器必须修改相应映射并重新运行，不能沿用中文 POS 规则。

词级 token 排除 POS=wp、纯标点和符号；数字、英文计为词并按 Unicode 字符计长，区分大小写。LTP 某些 token 带空白，程序去掉边缘空白并回定位到原文，不修改消息本身。若解析器截断或丢字，直接报错，不用残缺解析代表原回复。词 n-gram 不跨标点、换行和消息；词汇多样性按语料顺序连接同人回复后分块，不把单条短消息硬算成一个语料块。

默认块长 200、MTLD 阈值 0.72、HD-D 抽样数 42；依据 [McCarthy & Jarvis (2010)](https://doi.org/10.3758/BRM.42.2.381) 的方法定义。HD-D 是无放回抽样中每个词型至少出现一次的概率之和除以 42。MTLD 没有可用 factor 时 null。只使用等长完整块，尾部 token 数、块数和每块值均输出。Zipf 用全部词 token、全部词型、顺序秩和自然对数拟合；这是描述性斜率，不是幂律成立检验。两者的数值都随语料规模变化。

## 语义抽取与规则来源

只共享一次词汇抽取调用，不为每个指标单独调用。话题另由回复之前的可见前文粗分一次，各语义项另按话题分组汇总（`by_topic`）。

LLM 传输格式由 `wire.py` 统一定义：`states` 为按指定项目顺序排列的字符串，必须逐项返回 `p/a/u/n`（明确存在/确定没有/无法判断/不适用），提问清单另用 `k` 表示完整检查；详情只保留标签与原文证据，不生成解释。项目键以 JSON 字符串数组指定，`"153,160"` 是不可拆分的单个键。逐消息指标的状态字符串严格对应输入消息顺序；只有一条目标消息时详情直接放 `items[项目键]`，多条时再按消息 ID 定位。`items.questions` 直接为问题数组。无命中的形态句法结果可只返回 `{"states":"aa"}`。证据使用 `[消息ID,逐字原文,出现序号]`，输入消息使用 `[消息ID,role,原文]`。

163 的候选判定按输入顺序输出 decisions 字符串（0 非程度、1 非情绪程度、2 情绪程度、u 程度不明、? 情绪用途不明），不复述候选或证据；程序由候选目标恢复定位并核对完整覆盖。语用事件的空引用、空条件及未表达的 choice/insistence 省略，由程序恢复默认值。

程序将紧凑结果还原后执行原文定位、候选覆盖和跨项一致性校验；漏项目、漏消息、未知 ID、重复 JSON 键或缺少必要事件均为失败，不自动补 absent。修复响应可以返回指定的失败项目，或返回完整原始项目表；两者都必须覆盖完整，且不得出现未知项目。缓存键与 manifest 记录 `style-state-table-v3` 和批量协议 `style-chat-batch-v8`，旧缓存保留，原协议结果不会被当成新协议结果。

网络语、方言、圈内词、缩写、专业术语和叠词没有手写样本词表，提示词只用少量例子划出边界（如 012 的啥、咋不算方言）。固定的是粗标签和统计公式；这是跨样本协议。004,006 的候选由 `discover` 用 LLM 从聊天回复中提出，至少跨两个独立会话复现才接纳：阈值用于排除单片段偶发词串，不证明作者独占或心理偏好。实体/话题词单列，未计入惯用表达集合。更换本人或输入语料应重新发现。发现调用/证据失败时 artifact 标为 incomplete，`--frozen` 拒绝使用。

`manifest.json` 记录输入摘要、规范化语料摘要、解析模型身份、judge 后端、模型及后端参数、提示词摘要、阈值和运行范围。LLM 缓存含原始响应、实际返回模型名和证据；单条请求保存后端返回的 usage，批量请求的 usage 保存在 `llm/batches/` 并汇总到 `llm/progress.json`，不重复分摊到样本缓存；缓存键包括协议、模型、端点、提示词和输入，Codex 缓存还包括后端与配置参数，与原 API 缓存隔离。需要可复现结果时应指定可固定的 judge 版本；服务商别名更新后应使用新的缓存目录。原文、证据和缓存均是本地私人数据，不宜直接提交到公共仓库。

`--qa-samples N` 用固定随机种子复抽 N 个样本，保存原始/复抽结果与逐项一致性；`--review-model` 可增加另一模型的同样本复核。它们检验机器标注的稳定性，不证明标注准确。

## morphosyntax 抽取模块

运行 `run.py --chapter syntactic_style`，该抽取组产物位于 `extractors/morphosyntax/`。复用已有 LTP 缓存，LLM 使用 `morphosyntax-v2` 协议，仅为 026/036 共用一次标注。该类别同时调度句法和程度修饰抽取。

每个语料输出 `morphosyntax.json`，其中 `metrics` 按稳定编号（含合并编号）保存指标，`sentences` 保存逐句段数值及原文跨度。开启 LLM 时另输出 `annotations.json`、`semantic.json`。`manifest.json` 保存来源、代码、提示词摘要和 `morphosyntax_protocol`。

| 编号 | 口径与输出 |
| --- | --- |
| `025,027-030` | `counts/rates` 为非标点词的 POS 数量和比例，`group_counts/group_rates` 合并名词子类；`densities` 保存 a/d/v/r 的同分母密度及原编号。另保存逐句段密度；含标点的原始计数只作诊断。 |
| 026 | LLM 识别动作/事件被当作名词性成分指称的当前用法，提供目标跨度及包含它的局部语法依据；不把 POS=n 或“的/化”字面命中直接当名词化。 |
| 031 | 复用`lexical_choice`实词、功能词映射，报告 content/function/other 数量、各自占词数比例及 content/function；function=0 时该比值为 null，不丢弃对应句段。 |
| 032 | content 数量 ÷ 非标点词数，保存全语料值和逐句段分布。 |
| 033 | 按原始 POS 分别累计条件位置分布 P(位置\|POS)，分成 5 桶。 |
| 034 | 每个有效句段的严格首词/末词 POS Counter 和概率。 |
| 035 | 按词的中心位置筛选前 30% 和后 30%，各区间功能词数 ÷ 该区间词数；全语料汇总与逐句段分布分别输出。 |
| 036 | LLM 抽取完成、经历、进行、持续、习惯、将要、状态变化，以及过去/当前/未来时间用法。`aspect_labels` 与 `time_labels` 分开统计；`other_uses` 保存“了/过/着”在普通词等非体貌/时间用法中的证据，不计入用法发生率。 |

POS 分组遵循 [LTP 的 863 标签表](https://ltp.ai/docs/appendix.html)。形容词密度严格使用 a，不把 b/z 自动改归形容词；名词大类覆盖 n/nd/nh/ni/nl/ns/nt/nz。实词/功能词继续采用`lexical_choice`已定义的操作性映射，余下类别放在 other，不能假定两类穷尽全部词。这些映射来自解析器标签与已接受的统计约定，不是从这批样本回复挑选的词表；更换解析器或语言时需重新核对映射。

句段在消息边界、原始换行、LTP 标为 wp 且含 `。！？!?；;` 或恰为 `.` 的 token 处结束；逗号不截断句段，数字 token 内的小数点不截断。LTP 仍按原消息解析，句段仅用于统计位置，不声称是语义分句或重新解析出的句法树。无标点消息整体作为一个操作性句段。句段中位置按去除标点/纯符号后的词序重新编号：`(i+0.5)/L`。前区间为 `<=0.3`，后区间为 `>=0.7`，用整数不等式判边界以避免浮点误差。单词句保留在 0.5，前后区间分母为零而返回 null；纯标点句段、空消息的数量仍报告。逐句段平均密度与语料总体密度分别计算，不能相互替代。

026/036 的抽取字段在 `morphosyntax.py` 中定义。每个事件有 `label`、`evidence`、`basis`；程序校验原文出现序号及 basis 对 evidence 的包含关系，允许同一跨度承担不同体貌/时间标签，拒绝同标签同跨度重复。两项分别保留 present/absent/unknown/na/failure、有效样本量、发生率、每百词事件数及标签/表达分布。这里只验证引文和输出结构，不把机器语义判断当作人工真值。

真实调用发现过 JSON 遗漏必需字段。对收到合法 JSON 但字段/证据校验失败的项目，最多追加一次结构修正调用；已通过的项目保留首轮结果，仍失败的项目继续计为 failure。首轮、修正轮原始响应分别缓存。修正轮的缓存键在首轮请求之外加上轮次（含失败原因的修正请求还包括失败原因），所以重跑时直接复用，结果可复现。修正后仍有失败的样本，两轮缓存都标记为校验失败，下次运行会重新请求这两轮；`--qa-samples` 的复测每次都是新请求。逐项记录 `item_cache_keys`，汇总中报告 `initial_failure_count` 与 `repaired_count`，不在程序中补空字段或把失败改成 absent。修正提示词与首次提示词都在运行清单中记录摘要。

## syntactic_style 抽取模块

`syntactic_style.py` 覆盖 037–076 中的低、中难度及已有指标；暂缓高难度 049、050、055。042 和 047 在聊天上验证后停止抽取，记 `not_run`：LTP 在聊天句段上不输出条件角色，“如果……就……”被析成并列，042 无从计算；047 把“好的”“对”“行”这类应答算成述谓，和 043 重复。入口为 `run.py --chapter syntactic_style`，默认输出 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。

```bash
# 规则/解析器指标 + 上下文句法标注
python weclone/data/agent/distill_style/run.py --chapter syntactic_style

# 数值链路验证
python weclone/data/agent/distill_style/run.py --chapter syntactic_style --no-llm --limit 20
```

每个语料的 `syntactic_style.json` 保存按编号组织的指标、逐句段原文及结构、消息/窗口长度、分句连接证据。LLM 的 `annotations.json` 保存语义句段、语法完整度、句式和交际用法，`semantic.json` 汇总有效数量、未知和失败。

| 编号 | 实现口径 |
| --- | --- |
| 037–040 | 规则句段的原文 Unicode 字符数、非空白字符数、非标点词数、全部 CWS token 数；均值、总体方差及原始长度分布。消息与窗口长度另存，不能混称句长。 |
| 041 | dep/SDP 支持的单句、复句、短语片段及未知；保存谓词、嵌入候选、事件关系和操作性分句的 token 集合。单复句比例另输出。 |
| 043 | LLM 结合可见前文判完整、省略、碎片或未知；语气/应答词在此项归碎片，原始细标签保留。 |
| 044–046 | 同一谓词的显式 SBV 及 VOB/IOB/FOB；成分 token 集合及其包络跨度、主谓宾组合、带主语分句率、带宾语谓词率和宾语数量。拟声串（`vocal_token`）不算主语或宾语，LTP 常把 emmm 析成前置宾语。未检测到成分不等于省略。 |
| 048 | 同回复相邻已识别分句对；右侧分句的前置连词需同时经 dep 附着与 SDP mRELA 指向其谓词。只连接名词的连词不计；左侧标记或附着不一致的作用范围保留未知，单分句不适用；单个句段边界不明也记未知，不当作单分句。 |
| 051–054 | LLM 的陈述、疑问、感叹、祈使形式。单独成段的叹词、詈语、评价词或应答语（好家伙、nb）记为其他，不算感叹句。逐句段保存原文证据和独立的交际用法说明，汇总消息命中率、窗口发生率与有效语义句段比例。疑问形式使用合并键 `052,211`，其消息命中率按逐消息句式状态计算；句段、消息分母和未知覆盖分开。 |
| 056 | 同轮设问—自答配对，验证问题先于自答、两侧均来自目标回复，不引用前文代替目标问题。 |
| 057、`058-059,062` | 改由语义句段标注给出语序：`order` 为 fronted（话题或受事成分在主语或谓语之前）、dislocated（主干说完后补出主语、宾语等，即易位句）或 unknown，省略即常规语序，只对完整句和省略句判断；fronted/dislocated 另给被移动成分 `moved`，程序校验它在本句段内。057 取 dislocated，058、059、062 合为前置成分取 fronted。分母是含完整句或省略句的消息（`applicable_messages`、`applicability_rate`），`moved_position` 是被移动成分中心在句段内的相对位置。LTP 的 TPC 只在 0.5% 的句段出现，典型话题句被析成 FOB，FOB 又混入 emmm 误析，因此不再用。 |
| 061 | 同一谓词显式主宾的 SVO/SOV/OSV 等顺序；缺成分组合另列，多个主宾按组合计数并报告完整谓词数。V 表示谓词，不限动词词性。 |
| 060 | 复用语义完整度，按短句长度筛选后区分省略、完整、语气回应、碎片及未知。默认最多 5 个非标点词，`--short-words` 可改；完整短句不计作省略。 |
| 063–067 | SBV/VOB/ADV/CMP 的有向与绝对依存距离；逐谓词前置 ADV 依存项数。 |
| `068-069` | 一个依存标签 Counter 和比例向量，不额外输出 069 别名。 |
| 070 | 逐规则句段的原始依存标签条件位置分布。 |
| 071–074 | 词汇根深度为 0 的最大树深度；非叶节点/全部词汇节点两个分支因子；逐句段非 ROOT 边的平均/最大绝对距离。环、自指、越界、多根失败计入覆盖率；单节点树深度 0，无依存边的距离为 null。 |
| 075 | POS 类别:DEP 类别:指向方向与距离桶构成的原有骨架；保存完整 Counter、重复率、熵和 Top-10。句段保留短句。 |
| 076 | 每句段独立的原始 POS/DEP/SDP 1–3 gram Counter，不跨句段拼接。 |

第 3 章使用本地 LTP 的原文 `cws/pos/dep/sdp` 联合解析，不跑 SRL（聊天上程度、话题和条件角色几乎不出现，已没有指标使用），缓存在独立的 `ltp_syntax` 命名空间。原消息用来确定规则边界和复现整消息指标，规则句段再独立解析，使句法树不会跨换行或句末；不直接裁剪整消息的 head 数组。本地预分词批量接口可能返回多余的位置，因而使用原文接口并核验 token 的原文覆盖和所有结构长度。标点保留在树、依存距离、位置与结构序列中；人工 ROOT 不作为 token 或距离边。

规则依据 [LTP 标签定义](https://ltp.ai/docs/appendix.html) 和本地模型实际标签集，不来自样本回复中的触发词。041 要求同层 COO/IS 结构，COO 还需要指向一致的 eCOO/ePREC/eSUCC；各自带 SBV 的并列述谓分开，共享显式主语的并列谓词合并。d/r 嵌套或补语结构不直接算独立分句；兼语、无主语多谓词或依附不明确的连动结构记未知。ADV 下的形容词/动词还须有 SBV 或嵌套关系才列入谓词，避免把一般状语当作独立述谓。此规则是保守操作性分类；解析成功不代表句法分类准确，未知与覆盖率必须和比例一起查看。成分的 `token_indices` 是精确集合，`start/end` 仅是包络范围，可能包含非连续成分之间的文字；这些偏移相对句段，外层句段提供消息内偏移。

固定的语法标签、距离公式及分句原则可跨样本复用，更换解析器/语言需重核标签和角色。5 词的“短句”是显式的工程约定，不是从当前结果挑选的最优参数。

LLM 一次共享调用同时覆盖 043、051–054、056、060，不重复抽取同一语义句段。语义分句覆盖目标消息的全部非空白字符，程序拒绝遗漏、重叠、乱序、不可定位证据和缺字段；至多一次结构修正，已通过的分句或问答部分保留首轮结果。`units` 是 LLM 语义分句，与解析器的规则句段分别输出；其长度单列在 `semantic_lengths`，不替换 037–040。短句词数按语义跨度与原消息 CWS token 的相交计算，若模型在一个 token 内划界，该 token 可能分别计入两侧，因而不能把这些词数累加替代规则词数。机器判断和原文对齐均不构成人工准确率验证。

## orthographic_style 抽取模块

`orthographic_style.py` 实现 078–095 共 18 项，入口为 `run.py --chapter orthographic_style`，默认输出 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。全部是本地统计，不调用 LLM；数字识别复用 LTP 的 CWS/POS，其他表面统计直接读取原文。

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter orthographic_style
python weclone/data/agent/distill_style/run.py --chapter orthographic_style --no-llm
```

每个语料输出 `orthographic_style.json`（按编号保存指标，`messages` 保存原文偏移和逐消息观察）、`records.json`；本模块不再生成 LLM 标注。`manifest.json` 保存输入、代码、提示词、Unicode/regex 版本及统计协议。

| 编号 | 统计口径 |
| --- | --- |
| 078–084 | 逗号、句号、问号、感叹号、省略号、波浪号、冒号/分号的事件数、原始形式、每千非空白字符频率和消息命中率。全半角归入同类，原始形式仍分别保存。 |
| 085 | 最大连续标点串的完整形式、字符长度及消息比例；不将一个长串拆成多个重叠短串。 |
| 086 | 非空回合中完全没有标点的比例；`bubble_no_punctuation_rate` 按行（一次发送）算，分层按行长。纯空白消息为 NA。 |
| 087 | 程序定位最大空格/制表符游程，保存表面边界频率；不再判断语境用途。 |
| 088 | 聊天输入里换行就是发送边界，因此 088 测“用分条发送代替逗号/句号”：回合内每个行间边界按前一行的结尾归类（`preceding_endings`），`unpunctuated_boundary_rate` 是前一行无结尾标点的比例，按前一行长度分层。原始换行数另存为 `newline_count`。 |
| 089 | 标点字符数 ÷ 非空白字符数；标点事件数另存。单个省略号事件可能包含多个标点字符。 |
| 090 | 所有标点事件的类别分布；类别计数之和等于事件数。 |
| 091 | 每行（一次发送）去尾部空白后，恰好位于末尾的最后一个标点事件；否则为无标点结尾或 Unicode Emoji 呈现簇。`counts` 是全部行，`nonfinal_counts` 是后面还有下一行的行，`turn_final_counts` 是每个回合的最后一行（空文本另列）。不再进行字符表情的语境分类。 |
| 092 | LTP m 数词、nt 时间表达及相邻 q 量词，分别记录数字、数量和时间用法；按原文区分十进制数字、汉字数词、混合、非数字时间词及其他。合法年月日/时钟格式另列。 |
| 093 | 原文最大 Latin 字母及组合附标片段的全大写、首字母大写、全小写或混合；单个大写字母归全大写，保留词形及大小写。 |
| 094 | Unicode Sc/Sm/Sk/So 符号字符的全量形式、密度及类别分布，保留低频项。 |
| 095 | 已移除 LLM 字符表情识别，标记 not_run；保留原文标点/符号统计。 |

“标点”的表面定义为 Unicode `P*` 加上三种波浪号 `~～〜`。后者是清单单列的书写符号，其中一些不属于 Unicode P 类，所以同时输出严格 Unicode 标点字符数。其余类别由 Unicode General Category 补齐，不使用样本高频符号名单。连续 `…`、至少三个英文/全角点及相邻的上述片段按一个最大省略号事件计数；两个点仍是两个句点。数字两侧之间的点归 `numeric_dot`，不计 079；网址、缩写等其他点仍属于句点的表面形式，本章不判断它是否承担语义句末功能。冒号也不据字面推断是停顿或时钟用途。

计长使用原文 Unicode code point，不按 UTF-8 字节。默认频率基数为 1000；命中率以非空白的观测文本单元为分母，空白文本不算“没有标点”的语言样本。087/088 的原始边界总数仍包含空白消息内的边界。数据只有整轮回复时，指标的 `message` 表示该观测文本单元，不能解释为真实发送次数；只有 `boundary=message` 才累计实际跨消息边界。CRLF 是一个换行，孤立 CR/LF/NEL/LS/PS 各为一个换行；连续空行不并成一次发送。

句末 Emoji 检测用现有 `regex` 的扩展字素簇与 Unicode Emoji 呈现属性，保留 ZWJ、肤色、旗帜和键帽序列的末尾簇；文本呈现选择符 VS15 不作 Emoji。这个检查用于结尾类型，不是完整 RGI 序列合法性验证，也不是`paralinguistic_style`的 Emoji 计数实现。呈现/文本变体区分依据 [Unicode UTS #51](https://www.unicode.org/reports/tr51/)。字符级标点/符号统计与字素簇类型是不同统计轴，Emoji 组件、波浪号等可能同时出现在不同轴中，不应将它们相加当作互斥事件。

092 的数词识别由 LTP 标签决定，不建设汉字数词词表；例如普通名字内的数字字形不会仅凭字符命中就变成数词。日期表面格式支持 `YYYY-MM-DD`、斜线/点分隔同格式、`YYYY年M月D日`，校验公历日期；时钟支持 24 小时 `HH:MM[:SS]`。格式匹配只映射对应全角 ASCII，逐字偏移和原文不变。这些格式与 POS 数词是独立观察轴，不能累加成总数；不声称识别所有地区日期格式，或推断价格、编号、网络暗语。更换语言、解析器或目标日期格式时需更新协议后重跑。

087/088 只保留代码计算的边界数、频率与位置，语境用途标为 partial；095 和 091 的字符表情识别标为 not_run，不触发 LLM 请求。

这些类别、Unicode 属性和格式是跨样本的操作性规则，不是从样本回复挑出的触发词；字符表情与替代标点的功能没有手写白名单。更换 Unicode 库版本、语言或 LTP 模型后应记录新的版本及协议，机器用法判断仍需独立质量检查，原文跨度校验只能证明证据可定位。

## paralinguistic_style 抽取模块

`paralinguistic_style.py` 按当前清单的合并版实现：原 097–110 合为五项，111–116 为六项，共 11 项低、中难度指标。JSON 使用 `097-098`、`099-105`、`106-107,109` 等合并键，`source_ids` 保留原编号映射，不为某个汉字、字母、数字或重复长度单设指标。

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter orthographic_style
python weclone/data/agent/distill_style/run.py --chapter orthographic_style --no-llm
```

默认输出 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。每个语料有 `paralinguistic_style.json`（词类、重复、完整 Emoji 与逐窗口观察）、`records.json`；开启 LLM 后另有 `annotations.json`、`semantic.json`。清单记录代码/输入/提示词、解析器及 Emoji 数据摘要。默认 `--limit` 表示在时间窗内按原顺序取前 N 段工程验证样本。

| 合并项 / 编号 | 口径与输出 |
| --- | --- |
| 097–098 | 叹词和语气助词都按封闭语法类取，不看 `pos=e`：叹词是重复单元属于叹词集合、词性不是 u/v 的词；`pos=u` 必须有有效 RAD 附着、位于分句末尾且属于语气助词集合，才计语气助词。两个集合在聊天上核对过，语气助词补了哦、噢、喔、嘞、哈、捏。拟声串（`vocal_token`）两类都不算，LTP 会把“啊哈哈哈”标成 e、“哈哈”标成 o 或 z、hh 和 emmm 标成 ws。两类分别保存原始词形、频率及消息内位置；歧义助词另计，不默认归入语气词。 |
| 099–105 | LLM 从完整消息提取即时文本副语言表达，声音/语音状态与语境功能分开，保存 `function_by_writing`、实际词形和功能标签；不使用例词或重复形式预筛。 |
| 106–107、109 | 任意 Unicode 字符单元的最大连续重复跨度、最短重复单元、次数和书写长度；语气助词/叹词重复为跨度对齐后的子集。LLM 复用相同候选，区分构词、语法重叠、声音模拟、强化、节奏、其他及未知。 |
| 108 | 取 099–105 同一次标注里模拟声音的事件（paralinguistic=yes，sound 为笑声、哭腔、迟疑、感叹、其他人声或非人声），输出词形、位置、消息命中率和声音分布，不调 LLM 时不产出。`pos=o` 在聊天里七成是笑声，不再使用。机器/动物/环境声响的描写不自动视为人的即时情绪反应。 |
| 110 | LLM 判断书面延音作用；程序从确认跨度统计承载字符、符号、重复次数、长度。重复或波浪号本身不等于延音，书写长度不代表真实发音时长。 |
| 111 | 每个扩展字素簇与固定 Unicode 17.0 完整序列表精确匹配，保留 qualification 和原始形式；平台元数据事件另列，无法识别的序列和未核实的括号文本也另列。 |
| 112 | Unicode 表面 group/subgroup 与 LLM 语境功能分别保存。未知平台 ID 的语境功能可以未知，不把符号类别视为实际情绪。 |
| 113 | 完整 Emoji 事件数、消息命中率与每百原文字符事件数，复用 111，不再提取一次。 |
| 114 | 同一窗口内相邻 Emoji 事件的字符间隔、回合距离、行距离（`bubble_index_distance`、`intervening_bubbles`）；默认输入的窗口按顺序列出同一会话里本人的全部回合，所以跨回合的间隔也计入，另报同回合对数和行距离分布。时间只比较带时区的 ISO 消息时间戳，另报可用对数。 |
| 115 | 原始消息中的首/尾/中间/整条消息位置；只含一个 Emoji 的消息归整条，不重复计首尾。 |
| 116 | 已停止 LLM 抽取，标记 not_run；不把未识别字符表情算作零。 |

词类规则以 [LTP 的 e/o/u、RAD 定义](https://ltp.ai/docs/appendix.html) 为结构基础。语气助词的冻结集合在模块顶部 `PARTICLES`：核心句末语气形式、其常规语音变体及句末固定表达，按语法身份确定，不从样本回复的高频字符串挑选。实际词形仍来自 CWS；同一形式的重复使用最短字形单元判集合归属。句末的“了/的”有多功能，不仅凭 RAD 当成语气助词，未覆盖的句末 u 也保留为 `unknown_particle`；“过/着/地/得”等结构或体貌形式排除。助词之后允许句末标点、符号或其他语气助词，换行或分句标点结束当前范围。此保守口径不覆盖所有中文语气系统；更换语言、解析器或扩展集合应在独立开发样本核对并重新冻结。

重复提取保留大小写、数字、符号和空白，要求连续整数次 `k>=2`。同一跨度取最短单位；嵌套或重叠候选按最长跨度优先、起点优先保留非重叠集合，防止同一重复被多个 n-gram 重算。交错循环串的部分尾单元不强行补成一次重复。这里只测字形，不推断不同字形是否同音。原始统计保留每个解析 token/重复跨度的独立观察；`event_registry` 进一步合并相交的文本跨度，保留各来源的原始边界、词类、声音、功能、延音及重复用途。相邻而不相交的表达不合并，属性未判断时为 unknown。注册表的事件总数不能用各指标计数相加替代，也不是副语言质量总分。

声音标签描述呈现的声音或发声状态，功能标签描述当前互动作用，两者可以不同；例如笑声可以用于缓和或反讽，而不证明真实开心。粗功能类别按作用定义，未覆盖类别保留 `other`，不生成类别解释，也不按词形硬映射到已有类。所有标签和字段含义在共享提示词中说明，模型同时看到完整消息、可见前文及程序候选，文本事件不局限于候选的命中跨度。原始书写类型由 Unicode 字符类别计算，与模型功能无关。

当前环境没有安装 Emoji 专用包，本章使用随代码保存的 [Unicode 17.0 emoji-test.txt](https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt)，文件和授权文本位于 `resources/`，保留原始版权/版本信息。该官方数据的 fully-qualified、minimally-qualified、unqualified 和 component 状态分别保存；已列出的未带呈现选择符形式也可以匹配，不能统称为 fully-qualified。匹配整个字素簇，不将未列出的 ZWJ 串拆成多个已知 Emoji；VS15 文本呈现不算 Emoji。数据和库更新后应重新运行，不能沿用旧数据摘要声称覆盖新版。

规范输入可在单条消息中提供 `platform_emojis=[{"emoji_id":"平台ID","start":0,"end":4}]`，偏移为原文 Unicode 字符区间；整条表情消息可用 `message_type="emoji"` 与 `emoji_id`，允许文本为空。只依据这些明确元数据识别平台表情，已覆盖跨度优先于文字中识别的 Unicode Emoji。没有元数据的 `[文字]` 仅是 `unverified_text_labels`，不计入 Emoji。输入适配器不凭文本猜平台 ID；默认聊天输入没有平台元数据，该类别没有观测量。

频率以原文 Unicode code point（含空白）为分母，使用率以观测到的目标消息单元为分母；空文本平台表情仍有一个事件，字符密度为 null。与`orthographic_style`的“非空白字符”口径不同，清单明确记录这个分母，不能直接混合两章数值。Emoji 间隔只使用同一窗口内可见的目标消息序列，不跨聊天段构造连续会话。消息时间戳之差只是发送时间之差，不代表单个 Emoji 的产生时间；缺失、无时区、格式错误或逆序时间保留 null。

主标注只调用一次，分别验证 text、repeat_usage、emoji_usage 三个部分；重复和 Emoji 候选必须逐个标注，文本跨度不得重叠或吞入已知 Emoji。至多一次结构修正，保留首次成功部分，原始/修正缓存可追溯。116 不再复用 095，也不发起请求。`--no-llm` 下 099–105、110、116 为 not_run，重复用途和 Emoji 语境功能为 partial；声音、功能未知不会被当作否定证据或零频次。

## pragmatic_style 抽取模块

`pragmatic_style.py` 共用抽取语气与篇章组织功能，其中 140–145 的正式归属为篇章类。原 117–150 按功能归为 11 项，沿用合并键并保存 `source_ids`；不为清单例词建立触发名单。

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter pragmatic_style
python weclone/data/agent/distill_style/run.py --chapter pragmatic_style --limit 20
python weclone/data/agent/distill_style/run.py --chapter pragmatic_style --no-llm
```

默认输出 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。`pragmatic_style.json` 保存原文字符数、词数、重复、标点和大小写；开启 LLM 后生成逐条 `annotations.json` 和聚合 `semantic.json`。不启用 LLM 时，146–150 只有形式观察、标记 partial，其余为 not_run，不能将形式频次当作语用功能。

| 合并项 | 统计对象和输出 |
| --- | --- |
| 117–119 | 显式确定性 strong/medium/weak/unknown；前三类另有 `certainty_evidence` 支持强弱判断，普通陈述的默认断言力不算显式强确定。未显式表达不生成事件；分母为显式确定性表达，不能解释成全部命题中的比例。 |
| 120–127 | 降低承诺、限定视角/信息、缩小程度/范围三种策略；缓和 yes/no/unknown 独立于策略；汇总消息命中、策略共现及各策略下的缓和。命题通过事件 ID 复用 117–119。 |
| 128–131 | 请求负担照顾、感谢、致歉和其他礼貌作用。一般功能按消息统计；`requests.request_politeness_rate` 以 136–137 中实际 request 为分母，其他行动类型不混入。 |
| 132–134 | 敬意和礼貌性自谦分别统计消息命中、原文措辞和每百词密度，允许同一表达带两种关系属性。 |
| 135 | 每条消息一个 none/light/marked 等级，unknown 单列；等级不从句长、词数或感谢次数推算。 |
| 136–137 | 行为类型 command/request/suggestion/permission 与 compulsory/optional/unexpressed/unknown 选择空间分开；按行为类型分别统计约束分布。permission 单列对已讨论行动明确授权选择，避免混成请求或拒绝。显式条件保留原文证据，不将有条件等同于可选。 |
| 138 | fixed/adjustable/unexpressed/unknown 坚持程度与 explicit/implicit/unknown 直接性分别统计，另统计联合分布；引用相应行动、缓和事件。 |
| 139 | 每个拒绝及其缓冲片段；只在缓冲关系可判定的拒绝中计算 `refusals.buffer_rate`，报告候选数、有效分母、未知比例和前后结构。 |
| 140–144 | 判断形成、表达修正、结构预告、态度转向和其他元话语；保存作用内容、功能及消息内位置分布。 |
| 145 | 每个反驳与同一对象下的认同/让步片段；`rebuttals.window_prior_support_rate` 以含反驳、且全窗口关系标注完整可判定的窗口为分母。关系未知不算“只有反驳”。 |
| 146–150 | 每个被强调内容只建一个事件，多种实现方式保存在 methods；统计方式共现、消息命中，`form_attributes` 对齐原文重复、标点和大小写观察。 |

共享 TONE 标注读取完整目标消息、可见前文及程序形式，不只检查命中某种形式的片段。每个事件保存当前表达的证据、作用对象、作用范围原文和功能标签；scope 可以引用可见前文，计数证据只能来自目标消息。缓冲、认同和让步也必须来自当前回复的目标消息，不能把前文中别人的行为计给当前说话者。作用对象隐含且无可定位原文时 scope 可为空。

每条目标消息在每项下都有 present/absent/na/unknown 状态；failure 由程序记录结构或调用失败，不由模型冒充 absent。一般功能的 `message_rate` 为 present/(present+absent)，一条消息同一功能只命中一次。117–119、136–137、138、139、145 没有相应命题、对方行动要求、立场/行动意图、拒绝或反驳时为 na，改报 `applicability_rate`（分母 `applicability_denominator`）；具体属性和行为率使用表中的条件分母。消息内任何该属性标签未知时，该消息不进入该属性的消息命中分母，但已确认事件与未知标签数量仍保留。请求礼貌率要求该窗口各消息的礼貌标注均可判定；另报行动检测未知消息数，避免把漏判机会当无请求。

每百词密度采用共享 `lexical_choice.word_token` 的 LTP CWS/POS 词口径，排除 wp 和不含字母/数字的 token；分母只来自同一指标 present/absent 消息，不能混入未知或不适用消息。密度是事件数，不是被覆盖词的比例；一个多属性事件不在事件总数中重复累计。前/中/后位置采用证据跨度中点在去首尾空白消息中的相对位置，分三等份，保留原文偏移及连续位置。缓冲和认同/反驳的 before/after/overlap 则由原始消息顺序与跨度严格计算，不从提示词中的“先”推断，不跨聊天段拼接。

功能分类由归并清单的“改变什么、如何改变”定义，并在模块顶部和提示词中冻结；不是从当前样本提取的字符串规则。独立边界样例检查同词不同功能、换写法、条件与选择空间、引用归属及跨消息结构。新的礼貌、元话语或强调方式用 other 保留具体功能说明；若需要稳定的新子类，应在独立开发数据上定义新协议后重跑。换语言或语域时需复核这些操作性类别；更换分词器影响每百词分母。代码、提示词、模型及输入摘要写入 manifest，已有标注缓存不会跨协议或提示词复用。

跨项共享关系使用类型明确的事件 ID；程序验证引用存在、请求照顾指向实际请求，并传播失效引用。已识别行动、拒绝或反驳的消息在 138 下必须适用，不能标 na；两条属性仍可各自未知，不从已适用自动推定坚持或直接程度。最多一次结构修复，只替换失败项，保留首轮有效项及缓存来源；再次失败保留 failure。类别和联合分布中的未知类别单列比例，频率保留各自分母，不将这些维度合成一个总分。`--qa-samples` 沿用既有随机复标接口，衡量可重复性；原文跨度和结构校验不等于人工确认语用标签正确。

## affective_style 抽取模块

`affective_style.py` 共用抽取 151、152、153,160、155、158、159、161、163；155 归情境层 `154-157,270`，163 归语法类；154、156、157、162 保留 deferred。这里只抽取可观察的表达，不恢复真实心理状态或推断人格。

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter affective_style
python weclone/data/agent/distill_style/run.py --chapter affective_style --limit 20
python weclone/data/agent/distill_style/run.py --chapter affective_style --no-llm
```

输出位于 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`：`conditions.json` 保存回复前条件，各语料的 `affective_style.json` 保存原文形式和 LTP 程度证据，`annotations.json`、`semantic.json` 保存共享情感标注与统计。155 适用时保存复用`paralinguistic_style`的 `laughter_annotations.json`，其中保留原始及修复缓存键。

| 编号 | 统计口径 |
| --- | --- |
| 151 | 语境中的正/负情感词语跨度与词级比例分别统计。最小完整跨度保留否定与当前用法；LTP 词只要与跨度有字符交集便对齐，每个词仅计一次；跨越正负两种跨度的词单列 mixed_alignment。分别报告正负跨度占比、情感词内占比及全部有效词中的比例。 |
| 152 | 每个消息一个表达强度 0–3；0 是明确未表达，1 轻微、2 清楚且较强、3 极强或爆发。未知为 null，不补零；不按感叹号、长度或副词直接赋分。 |
| `153,160` | 直接说出感受、符号/拟声承载、依赖语境的隐含表达三类可共现；同一消息同类只命中一次。未表达和未知分别保存。 |
| 155 | 仅在独立 happy 条件内统计感叹号、完整 Emoji 和笑声的频次、消息命中及每百字符密度，不设置“很多”的样本特定阈值。缺少适用条件时 unavailable，比例为 null。 |
| 158 | 抱怨行为的事件密度、消息发生率及窗口发生率；客观问题报告不由关键词自动计入。 |
| 159 | 吐槽作为戏谑/揶揄等方式的负向评价单独判断，不与所有负向情感或平实抱怨等同；允许与 158 共现。 |
| 161 | positive/negative/mixed/calm/unmarked/unknown 表达基调，保存消息分布；另按相同可见前文或独立 scene_id 分组，保存各组的基调分布 `context_groups`。无前文且无场景依据的消息不分组，数量单列。 |
| 163 | 程度副词和程度补语按封闭语法类取候选，`fixed` 成员按规则确认为程度修饰，`ambiguous` 成员由同次 AFFECT 标注区分程度与其他用途。是否修饰情感对象另标。 |

情感、强度、方式、抱怨、吐槽、基调和程度用途共享一次标注，每项必须覆盖所有目标消息。证据只取本条目标消息，并校验逐字定位和同项非重叠；程序检查强度、表达方式和基调的明显矛盾。最多一次结构修复，保留首轮有效项及缓存来源，失败保留 failure。默认消息发生率使用 present+absent 分母，空白 na、未知、失败单列；窗口发生率要求该窗口所有目标消息可判定且至少一条非空。基调/强度按消息计一次，不按证据片段数计等级频次。字符分母含原文空白；词分母复用 LTP CWS/POS 与 `word_token`，比例均保留实际分母。

155 的条件先独立计算。默认条件分类只接收 context，不能读取目标回复；happy/not_happy 需要目标说话者在前文自己明确给出的当前状态依据，中奖、祝贺等事件及对方的猜测不足以证明其开心。无依据、时效不明或冲突未解决为 unknown，分类失败另列。规范输入也可以在记录中提供独立场景设定：

```json
{"affect_scenario":{"source":"independent_scenario","scene_id":"预先定义的场景ID","happiness":"happy","description":"回复之前的独立设定及其来源"}}
```

该元数据必须由独立场景资料提供，不能由目标回复反推。`happiness` 可为 happy/not_happy/unknown。描述字段记录外部来源声明，不代表程序能验证设定的真实性。没有元数据时，161 使用前文摘要作为分组依据。

155 的感叹号复用`orthographic_style`原文标点事件，Emoji 复用`paralinguistic_style`固定 Unicode 完整序列解析，平台表情只依据明确元数据。笑声复用`paralinguistic_style` 099–105 的原提示词、协议、验证器及缓存；只对 happy 回复补齐该抽取，不能用笑声是否出现反过来改变条件。笑声属性未知或抽取失败的回复不加入笑声分母，表面符号仍可统计；因此两部分分别报告覆盖率。`--no-llm` 时仍可使用独立提供的场景设定统计表面形式，笑声用途未抽取；仅有普通前文时不能自动取得 happy 分组。

163 的候选是程度副词（在被修饰词前，dep=ADV）和程度补语（在被修饰词后，dep=CMP），被修饰词须是有效的非标点词。两组都是封闭语法类，词形表见 `affective_style.DEGREE_ADVERBS/DEGREE_COMPLEMENTS`，也写进 manifest。`fixed` 成员（很、太、挺、有点、最、更、比较等）只作程度修饰，标注只判情绪用途，不能标 0 或 u。`ambiguous` 成员（好、真、还、多、这么、那么、死等）另有频率、时间、方式、焦点、数量或结果用法，由标注先判是否程度修饰。还、这样、那样、如此只在修饰形容词时进候选。不用 SRL：本地模型在聊天句上一次 ARGM-DGR 都不输出，“太好了”“特别累”都标成 ARGM-ADV；旧做法因此把全部 d+ADV（不、也、都、就……）送去消歧。p0 最近一年的 13,000 条消息里，候选从 11,893 个降到 2,456 个，其中 1,045 个是 `fixed`，这么、那么、累死这类旧做法漏掉的程度修饰也进了候选。纯 LTP 输出报告 `fixed` 下界、按来源的候选词形和 ADV/CMP 数量。

163 的完整密度只使用程度用途全部可判定的消息及其词数，未知候选不当作非程度；保留全部候选数、已标注覆盖率、可判定覆盖率及情绪用途未知数量。换解析器、语言或语域时需重新核对两组词形和附着关系。

以上标签按功能定义冻结，开发样例用于检查否定、引用、同词不同功能、无显式情绪词和条件独立性，不是触发词来源。manifest 记录代码、输入、情感/条件提示词、解析器与 Emoji 数据摘要。高难度的真实状态条件下克制、悲伤表达及长期波动等未随本次实现扩大范围。

## rhetorical_devices 抽取模块

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter rhetorical_devices
# 前 20 个聊天段验证标注链路
python weclone/data/agent/distill_style/run.py --chapter rhetorical_devices --limit 20
```

默认输出至 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。使用可见前文与目标消息做一次共享 LLM 抽取，不需要加载 LTP；最多一次结构修复，保留原始输出缓存和逐项来源。`--no-llm` 仅保存输入及字符/消息数量，语义项均为 `not_run`。

| 编号 / 输出键 | 内容 |
| --- | --- |
| 166、167、168、169、170 | 比喻、拟人、夸张、反讽、排比 |
| `metonymy`、`antiphrasis` | 清单未编号的借代、反语；用稳定语义键，不另占原编号 |
| 171、172、174、175 | 对比、反差、自嘲、调侃 |
| `178-179` | 同一批去重网络梗事件的出现率、每百消息/每百字符密度，以及可空的名称、来源描述 |

173 谐音、176 冷幽默、177 段子、181 梗年代标为 `deferred`。类别定义按相似性、关联替代、评价反转、结构安排、语境中的表达作用区分，完整定义随 manifest 保存；排比采用至少三个结构相近且共同推进或强调的单元这一操作标准。没有触发词表。仅提及、分析或引用某种说法，不自动计为说话者使用修辞。反语属于本协议反讽子类，要求反讽证据覆盖对应反语；其他标签可共现，不能加总为综合分数。174、175 仍分别抽取，报告中合为 `174-175`（调侃，含自嘲）。

`annotations.json` 中每项以整个窗口给出 `present/absent/unknown/na/failure`，每个事件保留一段或多段原文证据及解释。多段、跨消息的排比或对比仍算一次事件；不跨窗口拼接。证据只定位目标消息，前文仅供理解。完全相同的事件跨度集合去重；同项不同事件的跨度重叠会进入修复，避免重复计数；同一表达在不同位置出现仍分别计数。

`semantic.json` 以涉及消息的发生率 `message_occurrence_rate` 为主，另存有效窗口数、窗口发生率、事件数及每百消息/字符密度。分母限于该项完整可判断的 `present+absent` 窗口，其非空消息及原始 Unicode 字符长度；字符包含标点、空白和 Emoji 码位。全部为空白的窗口为 NA。未知、失败不补零，并单独报告覆盖率；没有有效样本时比例为 null。跨消息事件只计一次，但可涉及多条消息。178-179 的梗名与来源仅作为待核验的可空描述，不作为来源事实，也不推测年代。合并项不额外调用 LLM，不按事件重复累加同一条消息的命中率。

## interactional_style 抽取模块

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter interactional_style
# 工程抽样与长消息阈值覆盖
python weclone/data/agent/distill_style/run.py --chapter interactional_style --limit 20 --long-characters 200
```

默认输出至 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。按当前精简版清单实现以下 11 项；高难度 220 真实等待行为暂缓。原清单已删去的长度指标不另行恢复编号。`--no-llm` 可运行连发形式与原始长度统计；218 标为 `partial`，其余语义项为 `not_run`。该抽取模块不加载 LTP；类别中的其他共用模块按需解析。

| 输出键 | 实现与口径 |
| --- | --- |
| `burst_pattern` | 按 LF、CRLF、CR 拆分非空文本片段，保留原文偏移、片段数量和长度；同一消息有至少两个非空片段记为换行连发形式。原消息数独立保存。 |
| 210 | 依据前文识别追问，保留完整求答措辞、独立求答目标和前文证据；与新问题、反问、设问自答区分。 |
| 212 | 复用独立问题数组，逐消息统计追问数，给出每条有效消息的均值和分布。逐窗口的追问数、全部求答问题数与窗口均值另存作诊断。一个问号不等于一个问题。 |
| 214 | 复用确认的追问，统计口语/中性/正式/混合措辞、是否重述对象、语气词、实际表达及字符长度。未知属性保留并附覆盖率。 |
| 218 | 消息、气泡、窗口三级长度分位数，按非空白字符数分档的消息数 `message_length_counts` 与比例 `message_length_shares`，长消息比例 `message_long_rate`，窗口级 `window_long_rate` 作诊断；语义标注分别报告展开说明 `explanation`、独白式延展 `monologue` 的消息命中率。 |
| 219、221 | 依据前文区分主动同主题扩展/新支线；分别标出告别、离开、语言收束等结束表达。 |
| 227、229、231 | 复述对方信息、直接昵称、显式认同；分别排除仅共享关键词、第三人称呼、单纯收悉/执行承诺。 |
| 228 | 情绪回应和事情回应各自对齐对方前文；程序按消息顺序和字符偏移计算情绪先、事情先、交错/重叠、仅一类或两类皆无。 |

`annotations.json` 中 `items.questions` 服务 210/212/214；问句形式统一由 syntactic_style 的 `052,211` 提取。其余 7 项与之共用一次请求，结构错误最多修复一次，有效部分保留。追问必须有可见前文依据；缺少前文不能把问题自动当作追问或新问题。227/228/231 的前文证据必须来自对方，不能用自己先前的话代替对方内容。所有证据逐字回定位，不用固定词表筛选。

`interactional_style.json` 保存全量表面统计。换行口径反映文本拆分习惯，没有时间戳时不声称是真实连续发送次数；字面字符串 `\n` 不拆分，空行和末尾换行不制造消息。长消息默认阈值为 200 个原始 Unicode 码位，这是可调整的描述阈值，不是经验证的独白边界；完整长度数组同时保留。

`semantic.json` 逐项报告消息、字符、窗口分母和未知/失败比例。主发生率为消息命中率，只用 `present+absent` 窗口；未知追问数为 null，不补零。228 的“先情绪后事情”条件比例只在两类回应都出现的窗口中计算，单类回应不补为相反顺序。无后续记录时，221 仅表示结束表达。

## register_style 抽取模块

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter register_style
python weclone/data/agent/distill_style/run.py --chapter register_style --limit 20
```

默认输出至 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。共享抽取 233、234、244、245 四项，其中 245 归词汇类；237 年龄代际、246 攻击性语言保留为 `deferred`。`--no-llm` 仅检查输入并统计字符、LTP 词和消息数量，四项语义抽取均为 `not_run`。

| 编号 | 实现 |
| --- | --- |
| 233 | 整个窗口的书面、口语、网络聊天、混合语体。混合需要至少两种语体的证据；无法区分时未知。与正式度分别判断。输出 `window_proportions` 与按消息加权的 `message_weighted_proportions`，后者另按窗口长度分档。 |
| 234 | 文言式表达的本人用法、引用/讨论用法分别统计，引用古句实施当下表达可属本人用法，不能只按引号分类。 |
| 244 | 直接复用 007 网络用语、010 缩写、178-179 梗；各自报告消息/窗口发生率、事件数、每百字符密度、实际表达和覆盖率，不合成总分。 |
| 245 | 语境中的粗口及谐写/改写/遮蔽变体，区分本人用法与引用/讨论，排除普通同形词、代码标识符和普通符号。不给攻击性或关系判断。 |

233/234/245 在一次共享请求中抽取，结构不符最多修复一次，保留有效项目。244 调用原章节的 `annotate` 函数，使用原协议、原输入和缓存键；缓存未命中时执行原章节共享抽取，不改写原章定义。网语、缩写、网络梗三个分项分母独立，失败或未知不会补成零；缩写、字符表情也不被声称为网络独有特征。

`annotations.json` 保存本节的新标注；`reused_annotations.json` 保存复用的三项及原缓存、修复来源。`semantic.json` 中 234/245 的主发生率只计本人用法，`all_usage` 保留包含引用的统计；仅有引用的窗口仍是本人用法发生率的有效负例。有效分母为该项完整判断的 `present+absent` 窗口中的非空消息及原始 Unicode 字符。`states` 保留原抽取状态，`hit_messages` 才是主发生率的命中数，`hit_windows` 是窗口级诊断。

语体不是作者身份判断，也没有“越正式越好”的奖励分数。`--qa-samples` 复测本模块新抽取的三项；复用项保留原章节的缓存来源。

## information_density 抽取模块

```bash
source .venv2/bin/activate
python weclone/data/agent/distill_style/run.py --chapter discourse_style
python weclone/data/agent/distill_style/run.py --chapter discourse_style --limit 20
```

默认输出至 `dataset/res_csv/agent/distill/style/<时间>_<类别>/extractors/<模块>`。256 在一次请求中抽取建议单元及其与对方需求的关系，结构错误最多修复一次。263 依赖省略、冗余等未完成的综合输入，保留 `deferred`。该抽取模块不加载 LTP；`--no-llm` 只保存输入和字符/消息数量，256 为 `not_run`。

每个建议事件保留当前回复的一个或多个原文跨度、提议释义、前文请求证据和关系依据。关系为 `requested`（直接满足请求）、`extra`（超出请求范围）或 `unknown`。前文证据必须来自对方；没有可见请求范围时，可确认存在建议但不能确认额外性。宽泛求方案后收到多个建议，不因此把后面的都算额外；事实回答、自己的计划、承诺执行和纯引用也不自动成为建议。

`annotations.json` 的 `state` 描述建议是否被完整识别；`semantic.json` 中 `256.states` 描述额外建议是否可判断，`extraction_states` 保留原提取状态。主发生率为消息命中率，它和建议数量、长度都只使用提取完整且所有关系均可判断的窗口；未知数量为 null，无额外建议才为 0。任一建议关系未知的窗口退出主分母，其中已确认的额外建议仍保留在标注明细，并以 `confirmed_extra_events_outside_complete_windows` 报告数量。含任何建议的消息比例另报 `message_advice_occurrence_rate`。

长度按额外建议原文的 Unicode 码位计算，不让 LLM 估计字数，不包含前文及非证据间隔。跨消息的同一提议算一个事件，同项同一跨度集合去重，不同位置再次提出可以分别计数，因此不称为去重后的语义行动数量。输出额外建议逐单元长度、逐窗口数量/字符数、原文字符占比，以及“有建议时包含额外建议”的窗口条件比例。窗口覆盖率与事件关系覆盖率分别保留。这里描述表达展开方式，不提供信息量、信息压缩率或建议质量总分。

默认按整段聊天抽取全部 B 发言，保留 A/B 原始交替顺序；标注、证据定位及统计分母使用同一批 B 发言。
