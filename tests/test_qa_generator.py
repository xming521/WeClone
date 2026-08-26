import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

from pandas import Timestamp

from weclone.data.models import Message, QaPair


def _load_data_processor(monkeypatch):
    pii_module = ModuleType("weclone.core.PII.pii_detector")
    pii_module.ChinesePIIDetector = object
    pii_module.PIIDetector = object
    monkeypatch.setitem(sys.modules, "weclone.core.PII.pii_detector", pii_module)

    cleaning_module = ModuleType("weclone.data.clean.strategies")
    cleaning_module.LLMCleaningStrategy = object
    cleaning_module.OlineLLMCleaningStrategy = object
    monkeypatch.setitem(sys.modules, "weclone.data.clean.strategies", cleaning_module)
    monkeypatch.delitem(sys.modules, "weclone.data.qa_generator", raising=False)

    return importlib.import_module("weclone.data.qa_generator").DataProcessor


def test_save_result_updates_configured_dataset_file(tmp_path, monkeypatch):
    DataProcessor = _load_data_processor(monkeypatch)
    dataset_dir = tmp_path / "custom-dataset"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "current-chat.json"
    dataset_path.write_text('[{"messages": [{"content": "stale data"}]}]', encoding="utf-8")
    (dataset_dir / "dataset_info.json").write_text(
        json.dumps({"custom-chat": {"file_name": "current-chat.json"}}),
        encoding="utf-8",
    )

    processor = DataProcessor.__new__(DataProcessor)
    processor.c = SimpleNamespace(dataset="custom-chat", dataset_dir=str(dataset_dir))
    qa_pair = QaPair(
        id=1,
        time=Timestamp("2026-08-26T00:00:00"),
        score=0,
        messages=[Message(role="user", content="fresh data")],
        images=[],
        system="",
    )
    monkeypatch.chdir(tmp_path)

    output_path = processor.save_result([qa_pair])

    saved_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert saved_data[0]["messages"][0]["content"] == "fresh data"
    assert output_path == str(dataset_path)
    assert not (tmp_path / "dataset/res_csv/sft/sft-my.json").exists()
