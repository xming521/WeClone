"""Synthetic security checks; never reads the user's chats or changes their state."""

import json
import os
import sqlite3
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from weclone.cli import cli
from weclone.utils import secure_storage as secure


class SecureStorageChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="weclone-storage-check-")
        self.root = Path(self.directory.name)
        self.config = self.root / "settings.jsonc"
        self.document = {
            "version": "0.4.01",
            "security_args": {
                "storage_mode": "encrypted",
                "state_dir": str(self.root / "state"),
            },
        }
        self.config.write_text(json.dumps(self.document))
        self.old_config = os.environ.get("WECLONE_CONFIG_PATH")
        secure.configure(self.config)
        secure.lock()
        self.password = "synthetic-password-123"
        secure.ensure_initialized(self.password)

    def tearDown(self):
        secure.lock()
        if self.old_config is None:
            os.environ.pop("WECLONE_CONFIG_PATH", None)
        else:
            os.environ["WECLONE_CONFIG_PATH"] = self.old_config
        self.directory.cleanup()

    def test_import_preserves_source_and_requires_password(self):
        original = self.root / "chat.csv"
        content = b"synthetic-private-chat,hello\n"
        original.write_bytes(content)
        imported = secure.import_file(original)
        self.assertNotEqual(original, imported)
        self.assertEqual(original.read_bytes(), content)
        self.assertNotIn(content, imported.read_bytes())
        self.assertEqual(secure.read_bytes(imported), content)
        with self.assertRaises(secure.SecurityError):
            secure.read_bytes(original)
        secure.lock()
        with self.assertRaises(secure.InvalidPassword):
            secure.unlock("wrong-password")
        with self.assertRaises(secure.StorageLocked):
            secure.get_unlocked_key()
        secure.unlock(self.password)
        self.assertEqual(secure.read_bytes(imported), content)

    def test_mode_only_config_uses_project_state_and_system_temporary_directory(self):
        self.config.write_text(json.dumps({"security_args": {"storage_mode": "encrypted"}}))
        secure.lock()
        with patch.object(secure, "ROOT", self.root):
            settings = secure.configure(self.config)
            self.assertEqual(settings["state_dir"], self.root / ".weclone")
            self.assertEqual(settings["runtime_dir"], Path(tempfile.gettempdir()).resolve())
            secure.ensure_initialized(self.password)
            with secure.runtime_directory() as runtime:
                self.assertEqual(runtime.parent, settings["runtime_dir"])
                (runtime / "chat.json").write_text("synthetic-private-fact")
            self.assertFalse(runtime.exists())

    def test_temporary_plaintext_is_encrypted_and_removed_even_on_error(self):
        output = self.root / "result"
        previous_tempdir = tempfile.tempdir
        previous_environment = {name: os.environ.get(name) for name in ("TMPDIR", "TMP", "TEMP")}
        with self.assertRaisesRegex(ValueError, "synthetic failure"):
            with secure.sensitive_runtime(
                {"output_dir": str(output)}, dataset_keys=(), model_keys=()
            ) as config:
                runtime_output = Path(config["output_dir"])
                runtime_root = runtime_output.parents[1]
                runtime_output.mkdir(parents=True, exist_ok=True)
                (runtime_output / "result.json").write_text("synthetic-private-fact")
                raise ValueError("synthetic failure")
        self.assertFalse(runtime_root.exists())
        self.assertTrue(secure.is_encrypted_file(output / "result.json.enc"))
        self.assertNotIn(b"synthetic-private-fact", (output / "result.json.enc").read_bytes())
        self.assertEqual(secure.read_text(output / "result.json"), "synthetic-private-fact")
        self.assertEqual(tempfile.tempdir, previous_tempdir)
        self.assertEqual({name: os.environ.get(name) for name in previous_environment}, previous_environment)

    def test_file_locks_serialize_encrypted_sqlite_across_processes(self):
        database = self.root / "shared.sqlite3"
        with secure.encrypted_sqlite(database) as db:
            db.execute("CREATE TABLE facts (value TEXT)")
        script = self.root / "insert.py"
        script.write_text(
            "import sys\n"
            "from weclone.utils.secure_storage import encrypted_sqlite\n"
            "for index in range(6):\n"
            "    with encrypted_sqlite(sys.argv[1]) as db:\n"
            "        db.execute('INSERT INTO facts VALUES (?)', (sys.argv[2] + str(index),))\n"
        )

        def insert(index):
            secure.run_python(script, [str(database), str(index)], check=True, capture_output=True)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(insert, range(2)))
        with secure.encrypted_sqlite(database) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 12)

    def test_chunk_authentication_truncation_and_trailing_data(self):
        output = self.root / "records.json"
        content = b"synthetic-private-chat--" * (secure.CHUNK_SIZE // 10)
        output = secure.write_bytes(output, content)
        original = output.read_bytes()
        self.assertEqual(secure.read_bytes(output), content)
        for damaged in (
            original[:-1],
            original + b"injected",
            original[: secure.HEADER_SIZE + 9]
            + bytes([original[secure.HEADER_SIZE + 9] ^ 1])
            + original[secure.HEADER_SIZE + 10 :],
        ):
            output.write_bytes(damaged)
            with self.assertRaises(secure.SecurityError):
                secure.read_bytes(output)
        output.write_bytes(original)
        secure.write_bytes(self.root / "empty", b"")
        self.assertEqual(secure.read_bytes(self.root / "empty"), b"")

    def test_sqlite_concurrent_updates_and_rollback(self):
        database = self.root / "review.sqlite3"
        with secure.encrypted_sqlite(database) as db:
            db.execute("CREATE TABLE facts (value TEXT)")

        def insert(index):
            with secure.encrypted_sqlite(database) as db:
                db.execute("INSERT INTO facts VALUES (?)", (f"synthetic-private-fact-{index}",))

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(insert, range(12)))
        physical_database = secure.resolve_path(database)
        self.assertNotIn(b"synthetic-private-fact", physical_database.read_bytes())
        before = physical_database.read_bytes()
        with self.assertRaises(ValueError):
            with secure.encrypted_sqlite(database) as db:
                db.execute("INSERT INTO facts VALUES ('uncommitted')")
                raise ValueError("synthetic cancellation")
        self.assertEqual(physical_database.read_bytes(), before)
        with secure.encrypted_sqlite(database) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM facts").fetchone()[0], 12)
        with self.assertRaises(sqlite3.DatabaseError):
            with sqlite3.connect(physical_database) as db:
                db.execute("SELECT * FROM facts")
        self.assertFalse(physical_database.with_name(physical_database.name + "-journal").exists())

    def test_encrypted_stream_rejects_removed_or_reordered_chunks(self):
        output = self.root / "chunked.bin"
        output = secure.write_bytes(output, b"A" * secure.CHUNK_SIZE + b"B" * secure.CHUNK_SIZE + b"C")
        original = output.read_bytes()
        frames = []
        offset = secure.HEADER_SIZE
        while offset < len(original):
            size = struct.unpack(">I", original[offset : offset + 4])[0]
            frames.append(original[offset : offset + 4 + size])
            offset += 4 + size
        header = original[: secure.HEADER_SIZE]
        for altered in ([frames[1], frames[0], *frames[2:]], [frames[0], *frames[2:]]):
            output.write_bytes(header + b"".join(altered))
            with self.assertRaises(secure.SecurityError):
                secure.read_bytes(output)

    def test_password_change_preserves_data_reset_invalidates_old_keys(self):
        output = self.root / "records.json"
        secure.write_json(output, {"content": "synthetic-private-fact"})
        generation, auth_generation = secure.get_generation(), secure.get_auth_generation()
        new_password = "new-synthetic-password-123"
        secure.change_password(self.password, new_password)
        self.assertEqual(secure.get_generation(), generation)
        self.assertNotEqual(secure.get_auth_generation(), auth_generation)
        with self.assertRaises(secure.InvalidPassword):
            secure.unlock(self.password)
        secure.unlock(new_password)
        self.assertEqual(secure.read_json(output)["content"], "synthetic-private-fact")
        old_key = secure.get_unlocked_key()
        secure.reset_password("reset-synthetic-password-123")
        with self.assertRaises(secure.StorageLocked):
            with secure.use_key(old_key):
                pass
        with self.assertRaises(secure.StorageLocked):
            secure._set_run_key(old_key)
        with self.assertRaises(secure.SecurityError):
            secure.read_json(output)

    def test_mode_is_fixed_and_internal_plaintext_cannot_be_overwritten(self):
        original = self.root / "original.json"
        original.write_text('{"content":"synthetic-private-fact"}')
        with self.assertRaises(secure.SecurityError):
            secure.write_json(original, {"content": "replacement"})
        self.document["security_args"]["storage_mode"] = "plaintext"
        self.config.write_text(json.dumps(self.document))
        with self.assertRaises(secure.SecurityError):
            secure.is_encrypted_mode()

    def test_export_never_overwrites_and_failure_leaves_no_plaintext_file(self):
        source, target = self.root / "secret.json", self.root / "export.json"
        source = secure.write_json(source, {"content": "synthetic-private-fact"})
        secure.decrypt_file(source, target)
        self.assertEqual(json.loads(target.read_bytes())["content"], "synthetic-private-fact")
        with self.assertRaises(secure.SecurityError):
            secure.decrypt_file(source, target)
        target.unlink()
        ciphertext = source.read_bytes()
        source.write_bytes(ciphertext[:-1])
        with self.assertRaises(secure.SecurityError):
            secure.decrypt_file(source, target)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".wc-write-*")))
        source.write_bytes(ciphertext)
        decrypt = secure._decrypt_stream

        def competing_writer(input_file, output):
            decrypt(input_file, output)
            target.write_bytes(b"other-writer")

        with patch.object(secure, "_decrypt_stream", competing_writer):
            with self.assertRaises(FileExistsError):
                secure.decrypt_file(source, target)
        self.assertEqual(target.read_bytes(), b"other-writer")

    def test_password_reset_during_write_cancels_commit(self):
        output = self.root / "result.json"
        encrypt = secure._encrypt_stream

        def reset_during_write(input_file, destination):
            encrypt(input_file, destination)
            secure.reset_password("reset-synthetic-password-123")

        with patch.object(secure, "_encrypt_stream", reset_during_write):
            with self.assertRaises(secure.SecurityError):
                secure.write_json(output, {"content": "synthetic-private-fact"})
        self.assertFalse(secure.file_exists(output))
        self.assertFalse(list(self.root.glob(".wc-write-*")))

    def test_read_does_not_downgrade_if_file_changes_between_header_checks(self):
        original = self.root / "plain.json"
        original.write_text('{"content":"synthetic-private-fact"}')
        with patch.object(secure, "is_encrypted_file", return_value=True):
            with self.assertRaises(secure.SecurityError):
                secure.read_bytes(original)

    def test_subprocess_uses_inherited_pipe_and_cli_exports_a_new_file(self):
        source = self.root / "source.json"
        source = secure.write_json(source, {"content": "synthetic-private-fact"})
        script = self.root / "child.py"
        script.write_text(
            "from weclone.utils.secure_storage import read_json\nimport sys\nassert read_json(sys.argv[1])['content'] == 'synthetic-private-fact'\nprint('child-ok')\n"
        )
        result = secure.run_python(script, [str(source)], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "child-ok")
        secure.lock()
        output = self.root / "export.json"
        result = CliRunner().invoke(
            cli,
            ["--config-path", str(self.config), "decrypt", "--input", str(source), "--output", str(output)],
            input=self.password + "\n",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn(self.password, result.output)
        self.assertEqual(json.loads(output.read_bytes())["content"], "synthetic-private-fact")

    def test_suffixed_ciphertext_is_required_and_scans_do_not_duplicate(self):
        logical = self.root / "chat.json"
        payload = {"content": "synthetic-private-fact"}
        physical = secure.write_json(logical, payload)
        self.assertEqual(physical, self.root / "chat.json.enc")
        self.assertFalse(logical.exists())
        self.assertEqual(secure.read_json(logical), payload)
        self.assertEqual(secure.read_json(physical), payload)
        logical.write_bytes(physical.read_bytes())
        legacy_bytes = logical.read_bytes()
        self.assertEqual(list(secure.iter_files(self.root)), [physical])
        physical.unlink()
        self.assertEqual(list(secure.iter_files(self.root)), [])
        legacy_tree = self.root / "old-input"
        legacy_tree.mkdir()
        (legacy_tree / "chat.json").write_bytes(legacy_bytes)
        for operation in (
            lambda: secure.read_json(logical),
            lambda: secure.read_json(logical, import_plaintext=True),
            lambda: secure.import_file(logical),
            lambda: secure.decrypt_file(logical, self.root / "export.json"),
            lambda: secure.materialize_tree(logical, self.root / "materialized.json"),
            lambda: secure.materialize_tree(legacy_tree, self.root / "materialized"),
        ):
            with self.assertRaisesRegex(secure.SecurityError, r"\.enc"):
                operation()
        self.assertFalse((self.root / "export.json").exists())
        self.assertFalse((self.root / "materialized.json").exists())
        secure.write_json(logical, {"content": "new synthetic fact"})
        self.assertEqual(logical.read_bytes(), legacy_bytes)
        self.assertEqual(secure.read_json(logical)["content"], "new synthetic fact")
        self.assertEqual(list(secure.iter_files(self.root)), [physical])
        self.assertFalse((self.root / "chat.json.enc.enc").exists())

    def test_temporary_inputs_restore_original_extensions_and_dataset_index(self):
        source, runtime = self.root / "dataset", self.root / "materialized"
        source.mkdir()
        info = {"synthetic": {"file_name": "chat.json"}}
        (source / "dataset_info.json").write_text(json.dumps(info))
        secure.write_json(source / "chat.json", [{"content": "synthetic-private-fact"}])
        secure.write_bytes(source / "image.jpg", b"synthetic-image")
        secure.materialize_tree(source, runtime)
        self.assertEqual(json.loads((runtime / "dataset_info.json").read_text()), info)
        self.assertEqual(
            json.loads((runtime / "chat.json").read_text())[0]["content"], "synthetic-private-fact"
        )
        self.assertEqual((runtime / "image.jpg").read_bytes(), b"synthetic-image")
        self.assertFalse(list(runtime.glob("*.enc")))
        self.assertTrue((source / "dataset_info.json").exists())
        self.assertTrue((source / "chat.json.enc").exists())

    def test_stage2_and_checkpoints_use_suffixed_files_on_rerun(self):
        from weclone.data import make_dataset_stage2 as stage2
        from weclone.data.agent import distill_profile

        source, people = self.root / "chat.json", self.root / "people"
        secure.write_json(
            source,
            [
                {
                    "id": "1",
                    "chat_with": "synthetic-person",
                    "messages": [
                        {"role": "assistant", "content": "a sufficiently long synthetic message"},
                    ],
                }
            ],
        )
        with patch.object(stage2, "_load_ltp", return_value=None):
            groups = stage2.build_stage2_outputs(input_path=source, output_dir=people)
            self.assertTrue(groups[0]["file_name"].endswith(".json.enc"))
            self.assertTrue(Path(groups[0]["output_path"]).is_file())
            self.assertEqual(len(list(distill_profile.iter_chat_files(people))), 1)
            stage2.build_stage2_outputs(input_path=source, output_dir=people)
            secure.write_json(people / "unmanaged.json", [])
            with self.assertRaisesRegex(ValueError, "not listed"):
                stage2.build_stage2_outputs(input_path=source, output_dir=people)
        checkpoint = self.root / "checkpoint.json"
        state = {"version": 1, "entries": {"sample": {"status": "done"}}}
        secure.write_json(checkpoint, state)
        loaded = distill_profile.load_state(
            checkpoint, input_dir=people, output_dir=self.root, overwrite=False
        )
        self.assertEqual(loaded["entries"], state["entries"])

    def test_training_checkpoint_suffixes_and_materialization(self):
        from weclone.train.security import _encrypted_checkpoint

        source, target, runtime = self.root / "training", self.root / "durable", self.root / "restored"
        checkpoint = source / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
        (checkpoint / "optimizer.pt").write_bytes(b"synthetic-training-state")
        secure.persist_tree(source, target)
        durable = target / "checkpoint-1"
        self.assertTrue(_encrypted_checkpoint(durable))
        self.assertTrue((durable / "trainer_state.json.enc").is_file())
        self.assertTrue((durable / "optimizer.pt.enc").is_file())
        secure.persist_tree(source, target)
        secure.materialize_tree(target, runtime)
        self.assertEqual(
            json.loads((runtime / "checkpoint-1/trainer_state.json").read_text()), {"global_step": 1}
        )
        self.assertEqual((runtime / "checkpoint-1/optimizer.pt").read_bytes(), b"synthetic-training-state")

    def test_training_entrypoints_accept_encrypted_datasets(self):
        from weclone.train import train_pt, train_sft

        dataset = self.root / "dataset"
        dataset.mkdir()
        (dataset / "dataset_info.json").write_text(
            json.dumps(
                {
                    "synthetic": {"file_name": "chat.json", "columns": {"prompt": "text"}},
                }
            )
        )
        payload = [{"text": "synthetic-private-chat"}]
        secure.write_json(dataset / "chat.json", payload)
        for module, stage in ((train_sft, "sft"), (train_pt, "pt")):
            with self.subTest(stage=stage):
                output = self.root / stage
                config = {
                    "dataset_dir": str(dataset),
                    "dataset": "synthetic",
                    "output_dir": str(output),
                    "stage": stage,
                }
                training = SimpleNamespace(**config, model_dump=lambda **kwargs: dict(config))
                preparation = SimpleNamespace(
                    dataset_dir=str(dataset), clean_dataset=SimpleNamespace(enable_clean=False)
                )
                called = []

                def fake_trainer(runtime, callbacks):
                    restored = Path(runtime["dataset_dir"]) / "chat.json"
                    self.assertEqual(json.loads(restored.read_bytes()), payload)
                    self.assertTrue(callbacks)
                    target = Path(runtime["output_dir"])
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "adapter.safetensors").write_bytes(b"synthetic-model")
                    called.append(restored)

                def load_config(arg_type):
                    return preparation if arg_type == "make_dataset" else training

                with (
                    patch.object(module, "load_config", side_effect=load_config),
                    patch.object(module, "get_current_device", return_value="cpu"),
                    patch.object(module, "run_exp", side_effect=fake_trainer),
                ):
                    module.main()
                self.assertEqual(len(called), 1)
                self.assertFalse(called[0].exists())
                self.assertFalse((output / "adapter.safetensors").exists())
                self.assertEqual(secure.read_bytes(output / "adapter.safetensors.enc"), b"synthetic-model")

    def test_csv_scan_deduplicates_manual_plaintext_exports(self):
        from weclone.data.qa_generator import DataProcessor

        folder = self.root / "csv" / "person"
        physical = secure.write_text(folder / "chat_0_1.csv", "synthetic,chat\n")
        secure.decrypt_file(physical, folder / "chat_0_1.csv")
        processor = DataProcessor.__new__(DataProcessor)
        processor.csv_folder = str(folder.parent)
        processor._parsed_csv_files = None
        self.assertEqual(processor.get_csv_files(), [str(physical)])

    def test_encrypted_memory_and_grouping_scans_preserve_logical_names(self):
        from weclone.data.agent import canonicalize_state_memories, group_state_memories, organize_memories

        source = self.root / "distill/state_people/person.json"
        physical = secure.write_json(
            source,
            [
                {
                    "id": "1",
                    "state_memories": [
                        {
                            "type": "stable_fact",
                            "content": "synthetic-private-fact",
                            "importance": 2,
                            "confidence": 3,
                        }
                    ],
                }
            ],
        )
        records = organize_memories.read_memories(self.root / "distill")
        self.assertEqual(records[0]["content"], "synthetic-private-fact")
        self.assertEqual(records[0]["origins"][0]["file"], str(source))
        self.assertEqual(group_state_memories.input_paths_for(source.parent), [source])
        self.assertEqual(group_state_memories.input_paths_for(physical), [source])
        grouping = self.root / "grouping/person.prefilter_candidates.json"
        secure.write_json(grouping, {})
        self.assertEqual(canonicalize_state_memories.grouping_paths_for(grouping.parent), [grouping])
        self.assertEqual(canonicalize_state_memories.person_stem(secure.resolve_path(grouping)), "person")

    def test_plaintext_keeps_names_and_rejects_existing_encrypted_sqlite(self):
        database = self.root / "review.sqlite3"
        with secure.encrypted_sqlite(database) as db:
            db.execute("CREATE TABLE facts (value TEXT)")
        physical = secure.resolve_path(database)
        ciphertext = physical.read_bytes()
        self.document["security_args"].update(
            storage_mode="plaintext", state_dir=str(self.root / "plain-state")
        )
        self.config.write_text(json.dumps(self.document))
        secure.lock()
        secure.configure(self.config)
        source = self.root / "plain-chat.json"
        self.assertEqual(secure.write_json(source, {"content": "synthetic-private-fact"}), source)
        self.assertFalse((self.root / "plain-chat.json.enc").exists())
        self.assertEqual(json.loads(source.read_text())["content"], "synthetic-private-fact")
        with self.assertRaises(secure.SecurityError):
            with secure.encrypted_sqlite(database):
                pass
        self.assertFalse(database.exists())
        self.assertEqual(physical.read_bytes(), ciphertext)


if __name__ == "__main__":
    unittest.main(verbosity=2)
