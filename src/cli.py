"""CLI. Resumable by default."""

from __future__ import annotations

import json
from typing import Annotated, Optional

import typer

from .analysis import analyse_dataset, combine_datasets
from .config import load_config
from .datasets import load_dataset_bundle
from .embedding import build_encoders
from .indexing import build_collection, token_vector_count
from .manifest import base_manifest, merge, record_dataset
from .memory import read_collection_memory, verify_expected_vectors, write_memory_record
from .qdrant_control import QdrantController
from .runner import run_dataset

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


def _echo(message: str) -> None:
    typer.echo(message)


def _parse_arms(arms: str, cfg) -> Optional[list[str]]:
    if arms.strip().lower() == "all":
        return None
    return [name.strip() for name in arms.split(",") if name.strip()]


def _parse_depths(depths: Optional[str]) -> Optional[list[int]]:
    if not depths:
        return None
    return [int(part.strip()) for part in depths.split(",") if part.strip()]


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="Dataset name, or 'all' for every enabled dataset")] = "nfcorpus",
    arms: Annotated[str, typer.Option(help="Comma-separated arm names, or 'all'")] = "all",
    depths: Annotated[Optional[str], typer.Option(help="Override the depth sweep, e.g. 10,50,100")] = None,
    resume: Annotated[bool, typer.Option(help="Skip records already on disk")] = True,
    overwrite: Annotated[
        bool, typer.Option(help="With --no-resume, first delete this dataset's existing raw records")
    ] = False,
    measure_memory: Annotated[
        bool,
        typer.Option(help="Read Qdrant's per-vector memory accounting off the built collection"),
    ] = True,
    resume_ingestion: Annotated[
        bool,
        typer.Option(
            help="Reuse a matching existing collection and re-send only missing points, "
            "instead of wiping Qdrant storage and rebuilding. Use after a crash mid-upload."
        ),
    ] = False,
    allow_unverified_baseline: Annotated[
        bool,
        typer.Option(help="Proceed even though the default dense model's Gate 1 reference is unverified"),
    ] = False,
    dense_model: Annotated[
        Optional[str],
        typer.Option(help="Override the baseline dense model every arm reranks (default: config/models.yaml)"),
    ] = None,
    dense_model_dim: Annotated[
        Optional[int],
        typer.Option(help="Dimension of --dense-model, if it can't be auto-resolved from FastEmbed's registry"),
    ] = None,
    analyse: Annotated[bool, typer.Option(help="Aggregate after the run completes")] = True,
    allow_partial: Annotated[
        bool,
        typer.Option(help="Aggregate even if some (arm, depth) units end up with incomplete records. Never use this for a result you intend to trust."),
    ] = False,
) -> None:
    """Index, measure memory, execute the interleaved plan, and aggregate."""
    cfg = load_config()
    if depths:
        cfg = cfg.with_depths_override(_parse_depths(depths))
    used_default_dense_model = dense_model is None
    if dense_model:
        from .embedding import resolve_dense_model_dim

        default_dense_model = cfg.models.dense.name
        dim = dense_model_dim if dense_model_dim is not None else resolve_dense_model_dim(dense_model)
        cfg = cfg.model_copy(
            update={
                "models": cfg.models.model_copy(
                    update={"dense": cfg.models.dense.model_copy(update={"name": dense_model, "dim": dim})}
                )
            }
        )
        _echo(f"baseline dense model overridden: {dense_model} (dim={dim})")
        if dense_model != default_dense_model:
            _echo(
                f"note: Gate 1's reference NDCG@10 is pinned to {default_dense_model!r}, so it "
                "does not apply to this run's baseline; Gate 1 will fail or be meaningless "
                "until it is re-pinned in config/models.yaml for the new model."
            )

    ref = cfg.models.dense.reference
    if used_default_dense_model and not ref.verified:
        if not allow_unverified_baseline:
            raise typer.BadParameter(
                f"Gate 1's reference {ref.metric}={ref.value} on {ref.dataset} is marked "
                "verified: false in config/models.yaml. Every quality number in this run "
                "would rest on an unconfirmed baseline. Confirm it against the MTEB "
                "leaderboard and flip the flag, or pass --allow-unverified-baseline to "
                "proceed anyway."
            )
        _echo("WARNING: proceeding with an unverified Gate 1 baseline (--allow-unverified-baseline).")

    names = (
        [spec.name for spec in cfg.datasets.enabled_ordered]
        if dataset.strip().lower() == "all"
        else [dataset]
    )
    requested_arms = _parse_arms(arms, cfg)

    if not resume and not overwrite:
        existing = list(cfg.paths["raw"].glob(f"{'*' if dataset.strip().lower() == 'all' else dataset}__*.jsonl"))
        if existing:
            raise typer.BadParameter(
                f"--no-resume with existing raw records for {dataset!r} would append into "
                f"the same files and double-count them. Pass --overwrite "
                "to delete the existing records first, or use --resume."
            )

    controller = QdrantController(cfg)
    controller.up()
    manifest_path = cfg.paths["manifest"]
    merge(
        manifest_path,
        base_manifest(cfg, image_digest=controller.image_digest(), launcher=controller._launcher),
    )

    encoders = build_encoders(cfg)
    merge(manifest_path, {"model_file_digests": encoders.model_file_digests()})
    from .muvera import matrices_digest as _muvera_matrices_digest

    merge(
        manifest_path,
        {
            "muvera_matrices_digest": _muvera_matrices_digest(
                encoders.muvera, cfg.models.muvera, cfg.models.colbert.dim
            )
        },
    )

    for name in names:
        spec = cfg.datasets.get(name)
        if not spec.enabled:
            _echo(f"[skip] {name}: {spec.disabled_reason or 'disabled in config'}")
            merge(manifest_path, {"exclusions": {name: {"__dataset__": spec.disabled_reason or "disabled"}}})
            continue

        _echo(f"\n=== {name} ===")
        bundle = load_dataset_bundle(cfg, name)
        _echo(
            f"  {len(bundle.documents)} docs, {len(bundle.queries)} sampled queries "
            f"({len(bundle.eval_ids)} evaluated / {len(bundle.warmup_ids)} warmup)"
        )

        if not resume and overwrite:
            removed = 0
            for f in cfg.paths["raw"].glob(f"{name}__*.jsonl"):
                f.unlink()
                removed += 1
            if removed:
                _echo(f"  --overwrite: deleted {removed} existing raw file(s) for {name}")

        _echo("  building the collection every arm queries")
        collection = build_collection(
            cfg, bundle, encoders, controller,
            restart=not resume_ingestion, resume=resume_ingestion,
        )

        # After the build, not before: Qdrant reports memory per named vector for the
        # real collection the arms query, so there is nothing to measure until it exists.
        memory_record: dict = {}
        if measure_memory and spec.measure_memory:
            memory = read_collection_memory(
                cfg,
                collection,
                dataset=name,
                n_documents=len(bundle.documents),
                n_token_vectors=token_vector_count(cfg, bundle),
            )
            verify_expected_vectors(memory, bundle.spec.representations)
            memory_record = memory.as_dict()
            write_memory_record(cfg.paths["agg"] / f"{name}__memory.json", memory)
            for vector_name, vec in sorted(memory.vectors.items()):
                _echo(
                    f"    {vector_name:8s} disk={_human(vec.total.disk_bytes)} "
                    f"working_set={_human(vec.total.working_set_bytes)}"
                )
        else:
            _echo("  memory read skipped")

        summary = run_dataset(
            cfg, bundle, encoders, collection,
            include_arms=requested_arms, resume=resume, log=_echo,
        )
        _echo(
            f"  wrote {summary.n_records_written} records "
            f"(skipped {summary.n_records_skipped}) in {summary.wall_seconds:.1f}s"
        )
        for arm, reason in sorted(summary.exclusions.items()):
            _echo(f"  [excluded] {arm}: {reason}")

        record_dataset(
            manifest_path,
            name,
            corpus_digest=bundle.corpus_digest,
            id_map_digest=bundle.id_map.digest,
            n_documents=len(bundle.documents),
            n_subsample=len(bundle.queries),
            n_warmup=len(bundle.warmup_ids),
            n_evaluated=len(bundle.eval_ids),
            representations=list(spec.representations),
            exclusions=summary.exclusions,
            memory=memory_record,
            extra={
                "plan_digest": summary.plan_digest,
                "units": summary.units,
                "collection_queried": collection,
                "shard_number": 1,
                # Without the pinned revisions a later rerun can silently retrieve a
                # different corpus, queries, or qrels from the same repo name.
                "hf_corpus_revision": spec.hf_corpus_revision,
                "hf_qrels_revision": spec.hf_qrels_revision,
            },
        )

        if analyse:
            written = analyse_dataset(cfg, bundle, allow_partial=allow_partial)
            for label, path in written.items():
                _echo(f"  [{label}] {path}")

    if analyse:
        combined = combine_datasets(cfg)
        if combined:
            _echo(f"\n[combined] {combined}")


@app.command()
def aggregate(
    dataset: Annotated[str, typer.Option(help="Dataset name, or 'all'")] = "nfcorpus",
    allow_partial: Annotated[
        bool,
        typer.Option(help="Aggregate even if some (arm, depth) units have incomplete records. Never use this for a result you intend to trust."),
    ] = False,
) -> None:
    """Re-aggregate from existing raw records without re-running any query."""
    cfg = load_config()
    names = (
        [spec.name for spec in cfg.datasets.enabled_ordered]
        if dataset.strip().lower() == "all"
        else [dataset]
    )
    for name in names:
        bundle = load_dataset_bundle(cfg, name)
        for label, path in analyse_dataset(cfg, bundle, allow_partial=allow_partial).items():
            _echo(f"[{label}] {path}")
    combined = combine_datasets(cfg)
    if combined:
        _echo(f"[combined] {combined}")


@app.command()
def memory(
    dataset: Annotated[str, typer.Option()] = "nfcorpus",
) -> None:
    """Read Qdrant's per-vector memory accounting for this dataset's collection.

    Reuses the collection if it is already there with the right vector config, since the
    numbers describe that collection and rebuilding it changes nothing about them.
    """
    cfg = load_config()
    controller = QdrantController(cfg)
    controller.up()
    encoders = build_encoders(cfg)
    bundle = load_dataset_bundle(cfg, dataset)
    collection = build_collection(
        cfg, bundle, encoders, controller, restart=False, resume=True
    )
    memory_read = read_collection_memory(
        cfg,
        collection,
        dataset=dataset,
        n_documents=len(bundle.documents),
        n_token_vectors=token_vector_count(cfg, bundle),
    )
    verify_expected_vectors(memory_read, bundle.spec.representations)
    write_memory_record(cfg.paths["agg"] / f"{dataset}__memory.json", memory_read)
    _echo(json.dumps(memory_read.as_dict(), indent=2, sort_keys=True))


@app.command("calibrate-muvera")
def calibrate_muvera(
    dataset: Annotated[str, typer.Option()] = "nfcorpus",
    n_docs: Annotated[int, typer.Option(help="Corpus slice to calibrate against")] = 2000,
    n_queries: Annotated[int, typer.Option()] = 25,
) -> None:
    """Measure FDE prefetch recall against exact max_sim on REAL ColBERT embeddings.

    Required before publishing any muvera number. Synthetic vectors cannot validate the
    encoder: on isotropic random tokens the FDE cannot beat a plain summed-token dot
    product at any parameter setting.
    """
    from src.muvera import calibrate

    cfg = load_config()
    encoders = build_encoders(cfg)
    bundle = load_dataset_bundle(cfg, dataset)

    docs = [doc.text for doc in bundle.documents[:n_docs]]
    query_ids = sorted(bundle.eval_queries())[:n_queries]
    _echo(f"embedding {len(docs)} documents and {len(query_ids)} queries with ColBERT...")
    doc_tokens = encoders.embed_docs_colbert(docs)
    query_tokens = [encoders.embed_query_colbert(bundle.queries[qid]) for qid in query_ids]

    result = calibrate(
        encoders.muvera,
        doc_tokens,
        query_tokens,
        depths=tuple(cfg.depths_for(dataset)),
        truth_k=cfg.run.execution.final_limit,
    )
    payload = result.as_dict() | {
        "config": cfg.models.muvera.model_dump(),
        "corpus": f"{dataset} first {n_docs} documents",
    }
    out = cfg.paths["agg"] / f"{dataset}__muvera_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    _echo(f"\nFDE dim            : {result.dim}")
    _echo(f"spearman vs max_sim: {result.spearman_vs_chamfer:.3f}")
    for depth in sorted(result.recall_at):
        _echo(
            f"  recall@{depth:<4d} FDE={result.recall_at[depth]:.3f}  "
            f"bag-of-tokens={result.bag_recall_at[depth]:.3f}"
        )
    if not result.beats_bag_of_tokens:
        _echo(
            "\nFAIL: the FDE does not beat a plain summed-token dot product. The muvera arm "
            f"would store {result.dim} dimensions per document for nothing. Retune "
            "k_sim / n_reps / d_proj, or exclude the arm explicitly."
        )
    else:
        _echo(f"\n[calibration] {out}")
        _echo("Set models.yaml muvera.calibrated: true only if the recall above is usable.")


@app.command()
def plan(
    dataset: Annotated[str, typer.Option()] = "nfcorpus",
    arms: Annotated[str, typer.Option()] = "all",
) -> None:
    """Print the shuffled execution plan and the arm exclusions. Touches nothing."""
    from .arms import resolve_arms
    from .runner import build_plan

    cfg = load_config()
    resolved, exclusions = resolve_arms(cfg, dataset, include=_parse_arms(arms, cfg))
    units = build_plan(cfg, resolved, dataset)
    _echo(f"{len(units)} units for {dataset}, shuffled with seed {cfg.run.seeds.plan_shuffle}:")
    for unit in units:
        _echo(f"  {unit}")
    for arm, reason in sorted(exclusions.items()):
        _echo(f"[excluded] {arm}: {reason}")


@app.command()
def doctor() -> None:
    """Check the environment before committing to a long run."""
    from .platform_info import current_platform

    cfg = load_config()
    info = current_platform()
    _echo(f"platform tag      : {info.tag}")
    _echo(f"cpu               : {info.cpu_model} ({info.logical_cpus} logical)")
    _echo(f"ram               : {info.total_ram_gb} GB")
    _echo(f"onnx providers    : {', '.join(info.onnx_providers) or 'none'}")

    controller = QdrantController(cfg)
    try:
        controller.wait_ready(timeout_s=5)
        _echo(f"qdrant            : ready at {cfg.run.qdrant.url}")
    except Exception as exc:
        _echo(f"qdrant            : NOT ready ({exc}). Run: docker compose up -d")

    ref = cfg.models.dense.reference
    if not ref.verified:
        _echo(
            f"WARNING: Gate 1 reference {ref.metric}={ref.value} on {ref.dataset} is marked "
            "verified: false. Confirm it against the MTEB leaderboard before publishing."
        )
    if not cfg.models.muvera.calibrated:
        _echo(
            f"WARNING: muvera is uncalibrated ({cfg.models.muvera.dim} dims from the paper's "
            "parameters, not a tuned setting). Run `benchmark calibrate-muvera` before "
            "publishing any muvera number."
        )


def _human(value: int | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


if __name__ == "__main__":
    app()
