"""Unified CLI for checklist chapters of personal style extraction."""

import argparse
import hashlib
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from weclone.data.agent.distill_style import (
    affective_style,
    information_density,
    interactional_style,
    lexical_choice,
    morphosyntax,
    orthographic_style,
    paralinguistic_style,
    pragmatic_style,
    register_style,
    rhetorical_devices,
    syntactic_style,
)
from weclone.data.agent.distill_style.data import (
    DEFAULT_WINDOW_DAYS,
    PRIMARY_CORPUS,
    UNITS,
    all_speaker_replies,
    apply_time_window,
    load_profile_chats,
    digest,
    load_training,
    read_rows,
    validate_records,
    write_json,
)
from weclone.data.agent.distill_style.extraction import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    Judge,
    Parser,
    distinct_records,
)
from weclone.data.agent.distill_style.grouping import (
    DEFAULT_MIN_GROUP_MESSAGES,
    assign_groups,
    build_corpora,
    corpus_of,
    corpus_scope,
    load_group_map,
    public,
)
from weclone.data.agent.distill_style.lexical_choice import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DISCOVERY_BATCH,
    DEFAULT_MIN_CONVERSATIONS,
    discover,
    validate_frozen,
)
from weclone.data.agent.distill_style.taxonomy import CATEGORIES, selected_categories, extractors_for
from weclone.data.agent.distill_style.stats import LENGTH_STRATA, RELIABILITY
from weclone.data.agent.distill_style.reporting import write_report
from weclone.data.agent.distill_style.wire import WIRE_VERSION
from weclone.data.agent.distill_style.sampling import (
    DEFAULT_GROUP_FLOOR,
    DEFAULT_LLM_SAMPLES,
    DEFAULT_SAMPLE_SEED,
    sample_records,
)
from weclone.data.agent.distill_style.batching import (
    BATCH_PROTOCOL,
    DEFAULT_LLM_BATCH_SIZE,
    DEFAULT_LLM_BATCH_INPUT_CHARS,
    DEFAULT_LLM_BATCH_OUTPUT_CHARS,
)
from weclone.utils import secure_storage


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = ROOT / "dataset/res_csv/agent/distill/style"
EXTRACTORS = {
    module.__name__.rsplit(".", 1)[-1]: module
    for module in (
        lexical_choice,
        morphosyntax,
        syntactic_style,
        orthographic_style,
        paralinguistic_style,
        pragmatic_style,
        affective_style,
        rhetorical_devices,
        register_style,
        information_density,
        interactional_style,
    )
}
# Input each extraction group reads: LTP words and POS ("words", the default), LTP dependencies ("syntax"),
# dependencies cut into sentence units ("syntax_units"), or raw text only (None).
PARSES = {
    "syntactic_style": "syntax_units",
    "paralinguistic_style": "syntax",
    "affective_style": "syntax",
    "rhetorical_devices": None,
    "interactional_style": None,
    "information_density": None,
}
# Every orthographic measure is local, so this group never calls the judge.
LOCAL_ONLY = {"orthographic_style"}
# Groups run concurrently, but a group whose requests another group also sends starts after that group and
# reads them from the cache. Register sends lexical and rhetorical requests in its joint batches, and
# affective re-annotates laughter with the paralinguistic prompt.
AFTER = {
    "lexical_choice": "register_style",
    "rhetorical_devices": "register_style",
    "affective_style": "paralinguistic_style",
}
# One lazily loaded LTP model serves every group, so groups parse one at a time.
PARSE_LOCK = threading.Lock()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("extract", "discover"), default="extract")
    parser.add_argument(
        "--chapter",
        choices=(*CATEGORIES, "all"),
        default="all",
    )
    parser.add_argument("--format", choices=("canonical", "chat"), default="chat")
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Chat snapshot, such as the people directory of a run"
    )
    parser.add_argument("--output", "--output-dir", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=ROOT / "dataset/res_csv/agent/distill/style_cache")
    parser.add_argument("--ltp-model", type=Path, default=ROOT / "models/LTPbase")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.jsonc")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--llm-provider",
        choices=("codex_exec", "api"),
        default=None,
        help="Default: settings.jsonc agent_distill_args.llm_provider (codex_exec if absent)",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--llm-samples",
        type=int,
        default=DEFAULT_LLM_SAMPLES,
        help="Semantic sample size; 0 uses all replies",
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Keep chat windows from the last N days before the latest record; 0 keeps all history",
    )
    parser.add_argument(
        "--groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report one corpus per contact (or --contact-groups label) with enough messages",
    )
    parser.add_argument(
        "--contact-groups",
        type=Path,
        default=None,
        help="Optional JSON mapping contact_id to a relationship label; unmapped contacts stay separate",
    )
    parser.add_argument("--min-group-messages", type=int, default=DEFAULT_MIN_GROUP_MESSAGES)
    parser.add_argument(
        "--group-floor",
        type=int,
        default=DEFAULT_GROUP_FLOOR,
        help="Minimum semantic windows per group, drawn inside --llm-samples",
    )
    parser.add_argument("--llm-batch-size", type=int, default=DEFAULT_LLM_BATCH_SIZE)
    parser.add_argument("--llm-batch-input-chars", type=int, default=DEFAULT_LLM_BATCH_INPUT_CHARS)
    parser.add_argument("--llm-batch-output-chars", type=int, default=DEFAULT_LLM_BATCH_OUTPUT_CHARS)
    parser.add_argument(
        "--limit", type=int, default=None, help="Engineering smoke test: first N chat segments"
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--short-words", type=int, default=syntactic_style.DEFAULT_SHORT_WORDS)
    parser.add_argument("--long-characters", type=int, default=interactional_style.DEFAULT_LONG_CHARACTERS)
    parser.add_argument(
        "--training",
        type=Path,
        default=None,
        help="Optional separate corpus for discover; default is the input",
    )
    parser.add_argument("--person-id", default="p0")
    parser.add_argument("--frozen", type=Path, default=None)
    parser.add_argument("--min-conversations", type=int, default=DEFAULT_MIN_CONVERSATIONS)
    parser.add_argument("--discovery-batch", type=int, default=DEFAULT_DISCOVERY_BATCH)
    parser.add_argument(
        "--qa-samples", type=int, default=0, help="Repeat annotations on seeded random samples"
    )
    parser.add_argument(
        "--review-model", default=None, help="Optional different model for the same random QA subset"
    )
    args = parser.parse_args(argv)
    for key in (
        "short_words",
        "long_characters",
        "batch_size",
        "workers",
        "block_size",
        "min_conversations",
        "discovery_batch",
        "llm_batch_size",
        "llm_batch_input_chars",
        "llm_batch_output_chars",
        "min_group_messages",
    ):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.block_size < 42:
        parser.error("--block-size must be >=42 for HD-D")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.qa_samples < 0:
        parser.error("--qa-samples must be nonnegative")
    if args.llm_samples < 0:
        parser.error("--llm-samples must be nonnegative")
    if args.window_days < 0:
        parser.error("--window-days must be nonnegative")
    if args.group_floor < 0:
        parser.error("--group-floor must be nonnegative")
    if args.command == "discover" and args.chapter not in ("lexical_choice", "all"):
        parser.error("discover belongs to lexical_choice")
    if args.frozen and args.chapter not in ("lexical_choice", "all"):
        parser.error("--frozen requires lexical_choice or all")
    return args


def source_identity(path, records):
    path = Path(path)
    if secure_storage.file_exists(path):
        return {"path": str(path), "sha256": hashlib.sha256(secure_storage.read_bytes(path)).hexdigest()}
    files = sorted({r["source_file"] for r in records})
    return {"path": str(path), "files": [source_identity(file, []) for file in files]}


def load_records(args):
    """Input windows inside the time window; --limit then keeps the first N for smoke tests."""
    if args.format == "chat":
        records = load_profile_chats(args.input_dir, args.person_id)
    else:
        records = validate_records(read_rows(args.input_dir))
    records, window = apply_time_window(records, args.window_days)
    return all_speaker_replies(records[: args.limit]), window


def prepare_inputs(args):
    """Load, window and group the input once per run."""
    records, window = load_records(args)
    if not records:
        raise ValueError("Empty input corpus")
    mapping = load_group_map(args.contact_groups)
    grouping = assign_groups(records, mapping, args.min_group_messages, args.groups)
    return {"records": records, "window": window, "grouping": grouping, "selection": None}


def timestamped(suffix):
    return OUTPUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + suffix)


def make_judge(args, protocol, model=None):
    return Judge(
        args.settings,
        args.cache / "llm",
        model or args.judge_model,
        args.base_url,
        protocol=protocol,
        provider=args.llm_provider,
    )


def qa(records, parsed, annotations, judge, args, *, annotate_fn, items, protocol):
    chosen = random.Random(42).sample(list(range(len(records))), min(args.qa_samples, len(records)))
    review = make_judge(args, protocol, args.review_model) if args.review_model else None
    try:
        samples = []
        for i in chosen:
            repeat = annotate_fn(records[i], parsed, judge, refresh=True)
            checked = annotate_fn(records[i], parsed, review) if review else None
            samples.append(
                {
                    "sample_id": records[i]["sample_id"],
                    "original": annotations[i],
                    "repeat": repeat,
                    "review": checked,
                    "exact_item_agreement": {
                        k: annotations[i]["items"][k] == repeat["items"][k] for k in items
                    },
                }
            )
        return {"seed": 42, "kind": "random_repeatability_not_accuracy", "samples": samples}

    finally:
        if review is not None:
            review.close()


def discover_expressions(args):
    output = args.output or timestamped(args.command)
    output.mkdir(parents=True, exist_ok=False)
    judge = make_judge(args, lexical_choice.PROTOCOL_VERSION)
    try:
        if args.training:
            records = load_training(args.training, args.person_id)[: args.limit]
            source = args.training
        else:
            records, _ = load_records(args)
            source = args.input_dir
        if not records or {r["person_id"] for r in records} != {args.person_id}:
            raise ValueError("Discovery needs nonempty single-person corpus")
        print(f"Discovering expressions from {len(records)} records", flush=True)
        artifact = discover(records, judge, args.workers, args.min_conversations, args.discovery_batch)
        artifact["input_path"] = str(source.resolve())
        artifact["artifact_digest"] = digest({k: v for k, v in artifact.items() if k != "artifact_digest"})
        write_json(output / "frozen_expressions.json", artifact)
        print(
            f"{artifact['state']}: {sum(c['accepted'] for c in artifact['candidates'])} recurring candidates -> {output}",
            flush=True,
        )
    finally:
        judge.close()


def parse_messages(kind, records, args, parsers):
    """Parse every target message for one extraction group; `parsers` keeps the LTP models across groups."""
    if kind is None:
        return {}
    if "words" not in parsers:
        parsers["words"] = Parser(args.ltp_model, args.cache / "ltp", args.device, args.batch_size)
    texts = [m["text"] for r in records for m in r["messages"]]
    if kind == "words":
        return parsers["words"].parse(texts)
    if "syntax" not in parsers:
        parsers["syntax"] = syntactic_style.SyntaxParser(parsers["words"])
    if kind == "syntax_units":
        return syntactic_style.prepare(records, parsers["syntax"])
    return parsers["syntax"].parse(texts)


def module_options(name, args, frozen):
    """CLI options for an extraction group's extract(), and for its protocol() where it has one."""
    if name == "lexical_choice":
        return {"block_size": args.block_size, "frozen": frozen}
    if name == "syntactic_style":
        return {"short_words": args.short_words}
    if name == "interactional_style":
        return {"long_characters": args.long_characters}
    return {}


def borrowed_prompt_digests(name):
    """Prompts a group sends besides its own annotation prompt."""
    if name == "affective_style":
        return {
            "context_prompt_digest": digest(affective_style.CONTEXT_PROMPT),
            "laughter_prompt_digest": digest(paralinguistic_style.ANNOTATION_PROMPT),
        }
    if name == "register_style":
        return {
            "reused_prompt_digests": {
                kind: digest(module.ANNOTATION_PROMPT + str(module.ITEMS))
                for kind, (module, _) in register_style.REUSE.items()
            }
        }
    return {}


def run_module(name, args, output, inputs, parsers, *, shared_judge=None, frozen=None):
    """Run one extraction group over the person and group corpora and write its manifest.

    Without `shared_judge` the group computes local statistics only.
    """
    module = EXTRACTORS[name]
    lexical = name == "lexical_choice"
    parse = PARSES.get(name, "words")
    output.mkdir(parents=True, exist_ok=False)
    judge = (
        shared_judge.for_protocol(module.PROTOCOL_VERSION)
        if shared_judge is not None and name not in LOCAL_ONLY
        else None
    )
    try:
        if judge is not None:
            judge.configure_batches(
                output / "llm",
                args.workers,
                args.llm_batch_size,
                args.llm_batch_input_chars,
                args.llm_batch_output_chars,
            )
        records = inputs["records"]
        # Only LLM annotation is sampled; without a judge the group covers every window.
        selection = inputs["selection"] if judge is not None else None
        if selection is not None:
            records = [r for r in records if r["sample_id"] in selection["selected_ids"]]
        corpora = build_corpora(records, inputs["grouping"], selection)
        if frozen:
            validate_frozen(frozen, records)
        print(
            f"{name}: {'Loading' if parse is None else 'Parsing'} {distinct_records(corpora)} windows "
            f"across {len(corpora)} corpora",
            flush=True,
        )
        with PARSE_LOCK:
            parsed = parse_messages(parse, records, args, parsers)
        qa_runner = None
        if args.qa_samples and judge is not None:
            annotate_fn = module.annotate
            if name == "syntactic_style":
                annotate_fn = partial(module.annotate, short_words=args.short_words)

            def qa_runner(rows, annotations):
                return qa(
                    rows,
                    parsed,
                    annotations,
                    judge,
                    args,
                    annotate_fn=annotate_fn,
                    items=module.ITEMS,
                    protocol=module.PROTOCOL_VERSION,
                )

        options = module_options(name, args, frozen)
        semantic = module.extract(corpora, parsed, judge, output, args.workers, qa_runner, **options)
        manifest = {
            "protocol": module.PROTOCOL_VERSION,
            "implementation_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "scope": module.SCOPE,
            "engineering_smoke_test": args.limit is not None,
            "limit": args.limit,
            "coverage": module.extraction_coverage(semantic, frozen, judge)
            if lexical
            else module.extraction_coverage(semantic, judge),
            "sources": {PRIMARY_CORPUS: source_identity(args.input_dir.resolve(), records)},
            "units": UNITS,
            "length_buckets": LENGTH_STRATA,
            "reliability": RELIABILITY,
            "window": inputs["window"],
            "grouping": public(inputs["grouping"]),
            "corpus_digests": {corpus: digest(rows) for corpus, rows in corpora.items()},
            "corpus_scope": corpus_scope(corpora),
        }
        if parse is not None:
            manifest.update(ltp_model=str(args.ltp_model), ltp_identity=parsers["words"].identity)
        manifest.update(
            judge_model=judge.model if judge else None,
            llm_wire_protocol=WIRE_VERSION if judge else None,
            llm_batch_protocol=BATCH_PROTOCOL if judge and judge.batcher else None,
            judge_provider=judge.provider if judge else None,
            judge_backend_options=judge.backend_options if judge else None,
            annotation_prompt_digest=None
            if name in LOCAL_ONLY
            else digest(module.ANNOTATION_PROMPT + str(module.ITEMS))
            if lexical
            else digest(module.ANNOTATION_PROMPT),
        )
        if lexical:
            manifest.update(
                topic_prompt_digest=digest(lexical_choice.TOPIC_PROMPT),
                block_size=args.block_size,
                frozen_artifact_digest=frozen["artifact_digest"] if frozen else None,
            )
        else:
            manifest[name + "_protocol"] = module.protocol(**options)
        manifest.update(
            batch_size=args.batch_size,
            workers=args.workers,
            llm_batch_size=args.llm_batch_size,
            llm_batch_input_chars=args.llm_batch_input_chars,
            llm_batch_output_chars=args.llm_batch_output_chars,
            llm_usage_path="llm/progress.json" if judge and judge.batcher is not None else None,
            context_policy="Visible chat only; completion annotations excluded; source inference conditioning retained in input artifact",
        )
        if parse in ("syntax", "syntax_units"):
            manifest["syntax_parser_identity"] = parsers["syntax"].identity
        manifest.update(borrowed_prompt_digests(name))
        write_json(output / "manifest.json", manifest)
        print(f"Completed -> {output}", flush=True)
        return manifest

    finally:
        if judge is not None:
            judge.close()


def run_jobs(jobs, after):
    """Run jobs concurrently; a job named in `after` starts once the job it names has finished."""

    def chained(job, first):
        if first is not None:
            first.result()
        return job()

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = {}
        for key, job in jobs.items():
            futures[key] = pool.submit(chained, job, futures.get(after.get(key)))
        return {key: future.result() for key, future in futures.items()}


def main(argv=None):
    args = parse_args(argv)
    # Inputs, caches and outputs all go through secure storage, so unlock before any worker thread needs the key.
    secure_storage.configure(args.settings)
    secure_storage.ensure_unlocked()
    if args.command == "discover":
        return discover_expressions(args)
    categories = selected_categories(args.chapter)
    output = args.output or timestamped(args.chapter)
    output.mkdir(parents=True, exist_ok=False)
    inputs = prepare_inputs(args)
    records, grouping = inputs["records"], inputs["grouping"]
    print(
        f"Input: {len(records)} windows, time window {inputs['window']['state']}; "
        f"{len(grouping['corpora'])} group corpora with >= {args.min_group_messages} messages",
        flush=True,
    )
    frozen = secure_storage.read_json(args.frozen) if args.frozen else None
    modules = extractors_for(categories)
    parsers, sampling, surface_dirs, directories, jobs = {}, None, {}, {}, {}
    if args.llm and args.llm_samples:
        groups = {r["sample_id"]: name for r in records if (name := corpus_of(r, grouping))}
        selected, sampling = sample_records(
            records, args.llm_samples, args.sample_seed, groups, args.group_floor
        )
        inputs["selection"] = {
            "selected_ids": {r["sample_id"] for r in selected},
            "pooled_ids": sampling["pooled_ids"],
            "group_ids": sampling["group_ids"],
        }
        write_json(output / "sampling.json", sampling)
        print(
            f"Semantic sample: {len(selected)}/{len(records)} windows "
            f"({sampling['pooled_selected']} person, group floor {sampling['group_floor']}); seed={args.sample_seed}",
            flush=True,
        )
    shared_judge = (
        make_judge(args, None) if args.llm and any(m not in LOCAL_ONLY for m in modules) else None
    )
    # Producers are submitted before the groups that wait for them.
    for name in sorted(modules, key=lambda m: m in AFTER):
        directories[name] = output / "extractors" / name
        jobs[name] = partial(
            run_module,
            name,
            args,
            directories[name],
            inputs,
            parsers,
            shared_judge=shared_judge,
            frozen=frozen if name == "lexical_choice" else None,
        )
    if sampling is not None and len(selected) < len(records):
        # Local statistics still cover every window; the report reads them from surface/.
        for name in modules:
            surface_dirs[name] = output / "surface" / name
            jobs["surface", name] = partial(
                run_module,
                name,
                args,
                surface_dirs[name],
                inputs,
                parsers,
                frozen=frozen if name == "lexical_choice" else None,
            )
    try:
        run_jobs(jobs, AFTER if shared_judge is not None else {})
    finally:
        if shared_judge is not None:
            shared_judge.close()
    manifest = write_report(
        output,
        categories,
        directories,
        surface_dirs=surface_dirs,
        sampling=sampling,
        window=inputs["window"],
        grouping=public(grouping),
    )
    print(f"Category report -> {output}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
