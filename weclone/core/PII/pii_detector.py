import re
from dataclasses import dataclass
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine
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

    def __init__(self, language: str = "en", threshold: float = 0.5, use_ai_model: bool = False):
        self.language = language
        self.threshold = threshold
        self.use_ai_model = use_ai_model

        self._init_engines()
        self.supported_entities = self.get_supported_entities()
        logger.info(f"支持的实体类型: {self.supported_entities}")

    def _init_engines(self):
        """初始化presidio分析和匿名化引擎"""
        if not self.use_ai_model:
            # 不使用AI模型，仅使用presidio内置的基于规则的检测器
            self.analyzer = AnalyzerEngine()
            # self.anonymizer = AnonymizerEngine()
            logger.info("Presidio引擎初始化成功，仅使用基于规则的检测器")
            return

        try:
            # 根据语言配置相应的NLP模型
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
            self.anonymizer = AnonymizerEngine()

            logger.info(f"Presidio引擎初始化成功，使用语言: {self.language}, 模型: {model_name}")
        except Exception as e:
            logger.warning(f"Presidio引擎初始化失败: {e}, 将使用默认配置")
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

    def has_pii(self, text: str, entities: Optional[List[str]] = None) -> bool:
        """
        检测文本中是否包含PII信息

        Args:
            text: 待检测的文本
            entities: 指定检测的实体类型，默认检测所有支持的类型

        Returns:
            是否包含PII信息
        """
        pii_results = self.detect_pii(text, entities)
        return len(pii_results) > 0

    def detect_pii(self, text: str, entities: Optional[List[str]] = None) -> List[PIIResult]:
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
            text=text, language=self.language, entities=entities, score_threshold=self.threshold
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

        logger.info(f"检测到 {len(pii_results)} 个PII实体")
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

    def batch_detect_pii(
        self, texts: List[str], entities: Optional[List[str]] = None
    ) -> List[List[PIIResult]]:
        """
        批量检测PII信息

        Args:
            texts: 待检测的文本列表
            entities: 指定检测的实体类型

        Returns:
            每个文本对应的PII检测结果列表
        """
        results = []
        for text in texts:
            pii_results = self.detect_pii(text, entities)
            results.append(pii_results)

        total_pii_count = sum(len(result) for result in results)
        logger.info(f"批量检测完成，共处理 {len(texts)} 个文本，发现 {total_pii_count} 个PII实体")

        return results

    def batch_anonymize_texts(self, texts: List[str], entities: Optional[List[str]] = None) -> List[str]:
        """
        批量匿名化文本

        Args:
            texts: 待匿名化的文本列表
            entities: 指定匿名化的实体类型

        Returns:
            匿名化后的文本列表
        """
        anonymized_texts = []
        for text in texts:
            anonymized_text = self.anonymize_text(text, entities)
            anonymized_texts.append(anonymized_text)

        logger.info(f"批量匿名化完成，共处理 {len(texts)} 个文本")
        return anonymized_texts


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
        presidio_results = super().detect_pii(text, entities)

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


# 便捷函数
def detect_pii_in_text(text: str, language: str = "en", threshold: float = 0.5) -> List[PIIResult]:
    """检测文本中的PII信息的便捷函数"""
    if language == "zh":
        detector = ChinesePIIDetector(threshold=threshold)
    else:
        detector = PIIDetector(language=language, threshold=threshold)

    return detector.detect_pii(text)


def anonymize_pii_in_text(text: str, language: str = "en", threshold: float = 0.5) -> str:
    """匿名化文本中PII信息的便捷函数"""
    if language == "zh":
        detector = ChinesePIIDetector(threshold=threshold)
    else:
        detector = PIIDetector(language=language, threshold=threshold)

    return detector.anonymize_text(text)


def has_pii_in_text(text: str, language: str = "en", threshold: float = 0.5) -> bool:
    """检测文本中是否包含PII信息的便捷函数"""
    if language == "zh":
        detector = ChinesePIIDetector(threshold=threshold)
    else:
        detector = PIIDetector(language=language, threshold=threshold)

    return detector.has_pii(text)


# 不使用AI模型的便捷函数（基于presidio内置规则）
def detect_pii_no_ai(text: str, language: str = "en", threshold: float = 0.5) -> List[PIIResult]:
    """使用presidio基于规则的检测器检测PII信息"""
    detector = PIIDetector(language=language, threshold=threshold, use_ai_model=False)
    return detector.detect_pii(text)


def has_pii_no_ai(text: str, language: str = "en", threshold: float = 0.5) -> bool:
    """使用presidio基于规则的检测器检测文本是否包含PII信息"""
    detector = PIIDetector(language=language, threshold=threshold, use_ai_model=False)
    return detector.has_pii(text)


def anonymize_pii_no_ai(text: str, language: str = "en", threshold: float = 0.5) -> str:
    """使用presidio基于规则的检测器匿名化PII信息"""
    detector = PIIDetector(language=language, threshold=threshold, use_ai_model=False)
    return detector.anonymize_text(text)
