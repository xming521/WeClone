from dataclasses import dataclass
from typing import List, Optional, cast

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine.recognizer_result import (
    RecognizerResult as AnonymizerRecognizerResult,  # type: ignore
)

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
        self.supported_entities = self.get_all_entities()
        self.filtered_entities = [
            entity for entity in self.supported_entities if entity not in self.not_filtered_entities
        ]
        if self.language == "en":
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

        self._add_custom_recognizers(language=self.language)

        self.batch_analyzer = BatchAnalyzerEngine(analyzer_engine=self.analyzer)

        # self.anonymizer = AnonymizerEngine()

        logger.info(
            f"Presidio engine initialized successfully, using language: {self.language}, model: {model_name}"
        )

    def _add_custom_recognizers(self, language: str):
        # Create numeric ID recognizer - matches 5+ digit numbers or numbers with - separators
        numeric_id_patterns = [
            Pattern(name="numeric_id", regex=r"\b(?:[A-Za-z]*\d{5,}[A-Za-z]*|\d+-\d+(?:-\d+)*)\b", score=0.8),
            Pattern(name="unicode_escape_id", regex=r"\\u[0-9a-fA-F]{4}", score=0.8),
            Pattern(name="hex_escape_id", regex=r"\\xa0", score=0.8),
        ]

        numeric_id_recognizer = PatternRecognizer(
            supported_entity="NUMERIC_ID",
            patterns=numeric_id_patterns,
            supported_language=language,
            name="numeric_id_recognizer",
            context=["id", "编号", "号码", "代码", "code", "number", "序号", "sequence", "identifier"],
        )

        self.analyzer.registry.add_recognizer(numeric_id_recognizer)

        logger.info("Custom numeric ID recognizer added")

    def has_pii(self, text: str, entities: Optional[List[str]] = None) -> bool:
        pii_results = self.detect_pii(text)
        return len(pii_results) > 0

    def batch_has_pii(self, texts: List[str]) -> List[bool]:
        """
        Check if multiple texts contain PII information using batch processing

        Args:
            texts: List of texts to be checked

        Returns:
            List of boolean values indicating whether each text contains PII
        """
        if not texts or not isinstance(texts, list):
            return []

        batch_results = self.batch_detect_pii(texts)
        return [len(results) > 0 for results in batch_results]

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

    def batch_detect_pii(self, texts: List[str]) -> List[List[PIIResult]]:
        """
        Detect PII information in multiple texts using batch processing

        Args:
            texts: List of texts to be detected

        Returns:
            List of lists containing detected PII information for each text
        """
        if not texts or not isinstance(texts, list):
            return []

        # Filter out empty or non-string texts
        valid_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str):
                valid_texts.append(text)
                text_indices.append(i)

        if not valid_texts:
            return [[] for _ in texts]

        # Use batch analyzer for multiple texts
        results_iterator = self.batch_analyzer.analyze_iterator(
            texts=valid_texts,
            language=self.language,
            entities=self.filtered_entities,
            score_threshold=self.threshold,
            n_process=12,
            batch_size=16,
        )

        # Process results
        all_pii_results = [[] for _ in texts]

        for batch_idx, results in enumerate(results_iterator):
            original_idx = text_indices[batch_idx]
            text = valid_texts[batch_idx]

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

            all_pii_results[original_idx] = pii_results

        total_entities = sum(len(results) for results in all_pii_results)
        if total_entities > 0:
            logger.debug(f"Batch detected {total_entities} PII entities across {len(valid_texts)} texts")

        return all_pii_results

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

            anonymized_result = self.anonymizer.anonymize(
                text=text, analyzer_results=cast(List[AnonymizerRecognizerResult], analyzer_results)
            )

            logger.info(f"Successfully anonymized {len(analyzer_results)} PII entities")
            return anonymized_result.text

        except Exception as e:
            logger.error(f"Text anonymization failed: {e}")
            return text

    def get_supported_entities(self) -> List[str]:
        return self.analyzer.get_supported_entities(language=self.language)

    def get_all_entities(self) -> List[str]:
        """Get all entities including custom ones from the registry"""
        predefined_entities = self.get_supported_entities()
        custom_entities = []

        # Get custom entities from registry
        for recognizer in self.analyzer.registry.recognizers:
            for entity in recognizer.supported_entities:
                if entity not in predefined_entities and entity not in custom_entities:
                    custom_entities.append(entity)

        return predefined_entities + custom_entities


class ChinesePIIDetector(PIIDetector):
    """Chinese PII detector, extended to recognize Chinese-specific PII"""

    def __init__(self, threshold: float = 0.5):
        super().__init__(language="zh", threshold=threshold)

        # Filter out country-specific entities that are not relevant for Chinese context
        country_prefixes = ["US_", "UK_", "SG_", "AU_", "IN_"]
        # Get entities that are actually supported by the analyzer
        all_entities = self.get_all_entities()
        supported_entities = self.get_supported_entities()

        self.filtered_entities = [
            entity
            for entity in all_entities
            if entity not in self.not_filtered_entities
            and not any(entity.startswith(prefix) for prefix in country_prefixes)
            and (entity in supported_entities or entity in ["NUMERIC_ID", "CHINESE_PII"])
        ]
        logger.info(f"Chinese PII filtered entity types: {self.filtered_entities}")

    def _add_custom_recognizers(self, language: str):
        # Add parent class recognizers first
        super()._add_custom_recognizers(language="zh")

        # Add Chinese-specific recognizers that are not covered by NUMERIC_ID
        chinese_patterns = [
            Pattern(name="chinese_id_with_x", regex=r"\b\d{17}[Xx]\b", score=0.9),
            Pattern(
                name="chinese_email", regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", score=0.9
            ),
            Pattern(
                name="chinese_email_with_plus",
                regex=r"\b[A-Za-z0-9._%+-]+\+[A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                score=0.95,
            ),
        ]

        chinese_recognizer = PatternRecognizer(
            supported_entity="CHINESE_PII",
            supported_language="zh",
            patterns=chinese_patterns,
            name="chinese_pii_recognizer",
            context=["中文PII"],
        )
        self.analyzer.registry.add_recognizer(chinese_recognizer)

        logger.info("Chinese PII recognizer added")
