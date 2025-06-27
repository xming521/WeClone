import re
from dataclasses import dataclass
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# from presidio_analyzer.analyzer_engine import logger as presidio_logger
from weclone.utils.log import logger


@dataclass
class PIIResult:
    """PII检测结果"""

    entity_type: str
    start: int
    end: int
    score: float
    text: str


class PIIDetector:
    """PII检测器，基于presidio库"""

    def __init__(self, language: str = "en", threshold: float = 0.5):
        self.language = language
        self.threshold = threshold

        self._init_engines()
        self.anonymizer = AnonymizerEngine()
        self.not_filtered_entities = ["DATE_TIME", "PERSON", "URL", "NRP"]
        self.supported_entities = self.get_supported_entities()
        self.filtered_entities = [
            entity for entity in self.supported_entities if entity not in self.not_filtered_entities
        ]
        logger.info(f"隐私过滤的实体类型: {self.filtered_entities}")

    def _init_engines(self):
        """初始化presidio分析和匿名化引擎"""
        model_mapping = {
            "zh": "zh_core_web_sm",
            "en": "en_core_web_sm",
            "es": "es_core_news_sm",
            "fr": "fr_core_news_sm",
            "de": "de_core_news_sm",
        }

        model_name = model_mapping.get(self.language, "en_core_web_sm")

        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": self.language, "model_name": model_name}],
        }

        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        self._add_custom_recognizers()

        # self.anonymizer = AnonymizerEngine()

        logger.info(f"Presidio引擎初始化成功，使用语言: {self.language}, 模型: {model_name}")

    def _add_custom_recognizers(self):
        """添加自定义识别器"""
        # 创建数字ID识别器 - 匹配5位以上的纯数字或数字间夹着-符号的模式
        numeric_id_patterns = [
            Pattern(name="numeric_id", regex=r"\b(?:\d{5,}|\d+-\d+(?:-\d+)*)\b", score=0.8),
        ]

        numeric_id_recognizer = PatternRecognizer(
            supported_entity="NUMERIC_ID",
            patterns=numeric_id_patterns,
            name="numeric_id_recognizer",
            context=["id", "编号", "号码", "代码", "code", "number", "序号"],
        )

        # 注册自定义识别器到analyzer
        self.analyzer.registry.add_recognizer(numeric_id_recognizer)

        logger.info("已添加自定义数字ID识别器")

    def has_pii(self, text: str, entities: Optional[List[str]] = None) -> bool:
        """
        检测文本中是否包含PII信息

        Args:
            text: 待检测的文本
            entities: 指定检测的实体类型，默认检测所有支持的类型

        Returns:
            是否包含PII信息
        """
        pii_results = self.detect_pii(text)
        return len(pii_results) > 0

    def detect_pii(self, text: str) -> List[PIIResult]:
        """
        检测文本中的PII信息

        Args:
            text: 待检测的文本
            entities: 指定检测的实体类型，默认检测所有支持的类型

        Returns:
            检测到的PII信息列表
        """
        if not text or not isinstance(text, str):
            return []

        # 执行PII分析
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.filtered_entities,
            score_threshold=self.threshold,
        )

        # 转换为自定义结果格式
        pii_results = []
        for result in results:
            pii_result = PIIResult(
                entity_type=result.entity_type,
                start=result.start,
                end=result.end,
                score=result.score,
                text=text[result.start : result.end],
            )
            pii_results.append(pii_result)

        if pii_results:
            logger.debug(f"检测到 {len(pii_results)} 个PII实体")

        return pii_results

    def anonymize_text(self, text: str, entities: Optional[List[str]] = None) -> str:
        """
        匿名化文本中的PII信息

        Args:
            text: 待匿名化的文本
            entities: 指定匿名化的实体类型，默认匿名化所有检测到的类型

        Returns:
            匿名化后的文本
        """
        if not text or not isinstance(text, str):
            return text

        try:
            # 先检测PII
            analyzer_results = self.analyzer.analyze(
                text=text, language=self.language, entities=entities, score_threshold=self.threshold
            )

            # 执行匿名化
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results)

            logger.info(f"成功匿名化 {len(analyzer_results)} 个PII实体")
            return anonymized_result.text

        except Exception as e:
            logger.error(f"文本匿名化失败: {e}")
            return text

    def get_supported_entities(self) -> List[str]:
        """获取支持的实体类型"""
        return self.analyzer.get_supported_entities(language=self.language)


class ChinesePIIDetector(PIIDetector):
    """中文PII检测器，扩展了对中文特有PII的识别"""

    def __init__(self, threshold: float = 0.5):
        super().__init__(language="zh", threshold=threshold)
        self._init_chinese_patterns()

    def _init_chinese_patterns(self):
        """初始化中文特有的PII模式"""
        self.chinese_patterns = {
            "CHINESE_ID_CARD": re.compile(r"\b\d{15}|\d{18}|\d{17}[Xx]\b"),
            "CHINESE_PHONE": re.compile(r"\b1[3-9]\d{9}\b"),
            "CHINESE_NAME": re.compile(r"[\u4e00-\u9fff]{2,4}"),
            "QQ_NUMBER": re.compile(r"\b[1-9]\d{4,10}\b"),
            "WECHAT_ID": re.compile(r"\bwxid_[a-zA-Z0-9]{22}\b"),
        }

    def detect_chinese_pii(self, text: str) -> List[PIIResult]:
        """检测中文特有的PII信息"""
        chinese_results = []

        for entity_type, pattern in self.chinese_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                result = PIIResult(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.9,  # 正则匹配给予较高置信度
                    text=match.group(),
                )
                chinese_results.append(result)

        return chinese_results

    def detect_pii(self, text: str, entities: Optional[List[str]] = None) -> List[PIIResult]:
        """重写检测方法，结合presidio和中文模式"""
        # 使用父类方法检测标准PII
        presidio_results = super().detect_pii(text)

        # 检测中文特有PII
        chinese_results = self.detect_chinese_pii(text)

        # 合并结果并去重
        all_results = presidio_results + chinese_results
        all_results = self._remove_duplicates(all_results)

        logger.info(f"检测到 {len(presidio_results)} 个标准PII和 {len(chinese_results)} 个中文PII")
        return all_results

    def _remove_duplicates(self, results: List[PIIResult]) -> List[PIIResult]:
        """去除重叠的检测结果"""
        if not results:
            return results

        # 按位置排序
        results.sort(key=lambda x: (x.start, x.end))

        # 去重
        unique_results = [results[0]]
        for result in results[1:]:
            last_result = unique_results[-1]
            # 如果当前结果与上一个结果不重叠，则保留
            if result.start >= last_result.end:
                unique_results.append(result)
            # 如果重叠，保留置信度更高的
            elif result.score > last_result.score:
                unique_results[-1] = result

        return unique_results
