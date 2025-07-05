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
    entity_type: str
    start: int
    end: int
    score: float
    text: str


class PIIDetector:
    """PII detector based on presidio library"""

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
        logger.info(f"Privacy filtered entity types: {self.filtered_entities}")

    def _init_engines(self):
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

        logger.info(
            f"Presidio engine initialized successfully, using language: {self.language}, model: {model_name}"
        )

    def _add_custom_recognizers(self):
        # Create numeric ID recognizer - matches 5+ digit numbers or numbers with - separators
        numeric_id_patterns = [
            Pattern(name="numeric_id", regex=r"\b(?:\d{5,}|\d+-\d+(?:-\d+)*)\b", score=0.8),
        ]

        numeric_id_recognizer = PatternRecognizer(
            supported_entity="NUMERIC_ID",
            patterns=numeric_id_patterns,
            name="numeric_id_recognizer",
            context=["id", "编号", "号码", "代码", "code", "number", "序号", "sequence", "identifier"],
        )

        self.analyzer.registry.add_recognizer(numeric_id_recognizer)

        logger.info("Custom numeric ID recognizer added")

    def has_pii(self, text: str, entities: Optional[List[str]] = None) -> bool:
        pii_results = self.detect_pii(text)
        return len(pii_results) > 0

    def detect_pii(self, text: str) -> List[PIIResult]:
        """
        Detect PII information in text

        Args:
            text: Text to be detected
            entities: Specified entity types to detect, defaults to all supported types

        Returns:
            List of detected PII information
        """
        if not text or not isinstance(text, str):
            return []

        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.filtered_entities,
            score_threshold=self.threshold,
        )

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
            logger.debug(f"Detected {len(pii_results)} PII entities")

        return pii_results

    def anonymize_text(self, text: str, entities: Optional[List[str]] = None) -> str:
        """
        Anonymize PII information in text

        Args:
            text: Text to be anonymized
            entities: Specified entity types to anonymize, defaults to all detected types

        Returns:
            Anonymized text
        """
        if not text or not isinstance(text, str):
            return text

        try:
            analyzer_results = self.analyzer.analyze(
                text=text, language=self.language, entities=entities, score_threshold=self.threshold
            )

            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=analyzer_results)

            logger.info(f"Successfully anonymized {len(analyzer_results)} PII entities")
            return anonymized_result.text

        except Exception as e:
            logger.error(f"Text anonymization failed: {e}")
            return text

    def get_supported_entities(self) -> List[str]:
        return self.analyzer.get_supported_entities(language=self.language)


class ChinesePIIDetector(PIIDetector):
    """Chinese PII detector, extended to recognize Chinese-specific PII"""

    def __init__(self, threshold: float = 0.5):
        super().__init__(language="zh", threshold=threshold)
        self._init_chinese_patterns()

    def _init_chinese_patterns(self):
        self.chinese_patterns = {
            "CHINESE_ID_CARD": re.compile(r"\b\d{15}|\d{18}|\d{17}[Xx]\b"),
            "CHINESE_PHONE": re.compile(r"\b1[3-9]\d{9}\b"),
            "CHINESE_NAME": re.compile(r"[\u4e00-\u9fff]{2,4}"),
            "QQ_NUMBER": re.compile(r"\b[1-9]\d{4,10}\b"),
            "WECHAT_ID": re.compile(r"\bwxid_[a-zA-Z0-9]{22}\b"),
        }

    def detect_chinese_pii(self, text: str) -> List[PIIResult]:
        chinese_results = []

        for entity_type, pattern in self.chinese_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                result = PIIResult(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.9,
                    text=match.group(),
                )
                chinese_results.append(result)

        return chinese_results

    def detect_pii(self, text: str, entities: Optional[List[str]] = None) -> List[PIIResult]:
        """Override detection method, combining presidio and Chinese patterns"""
        presidio_results = super().detect_pii(text)

        chinese_results = self.detect_chinese_pii(text)

        all_results = presidio_results + chinese_results
        all_results = self._remove_duplicates(all_results)

        logger.info(f"检测到 {len(presidio_results)} 个标准PII和 {len(chinese_results)} 个中文PII")
        return all_results

    def _remove_duplicates(self, results: List[PIIResult]) -> List[PIIResult]:
        """Remove overlapping detection results"""
        if not results:
            return results

        results.sort(key=lambda x: (x.start, x.end))

        unique_results = [results[0]]
        for result in results[1:]:
            last_result = unique_results[-1]
            # Keep if current result doesn't overlap with previous result
            if result.start >= last_result.end:
                unique_results.append(result)
            # If overlapping, keep the one with higher confidence
            elif result.score > last_result.score:
                unique_results[-1] = result

        return unique_results
