"""CLI. Resumable by default."""

from __future__ import annotations

import json
from typing import Annotated

import numpy as np
import typer

from .analysis import analyse_dataset, combine_datasets
from .config import load_config
from .datasets import load_dataset_bundle
from .embedding import build_encoders
from .indexing import build_collection, token_vector_count
from .manifest import base_manifest, merge, record_dataset
from .memory import (
    check_fits_in_memory,
    read_collection_memory,
    verify_expected_representations,
    write_memory_record,
)
from .qdrant_control import QdrantController
from .runner import run_dataset

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)


def _echo(message: str) -> None:
    typer.echo(message)


def _parse_arms(arms: str, cfg) -> list[str] | None:
    if arms.strip().lower() == "all":
        return None
    return [name.strip() for name in arms.split(",") if name.strip()]


def _parse_depths(depths: str | None) -> list[int] | None:
    if not depths:
        return None
    return [int(part.strip()) for part in depths.split(",") if part.strip()]


@app.command()
def run(
    dataset: Annotated[
        str,
        typer.Option(
            help="Dataset name, or 'all' for the default set. Naming a dataset explicitly runs it even if it is out of that set."
        ),
    ] = "nfcorpus",
    arms: Annotated[str, typer.Option(help="Comma-separated arm names, or 'all'")] = "all",
    depths: Annotated[
        str | None, typer.Option(help="Override the depth sweep, e.g. 10,50,100")
    ] = None,
    resume: Annotated[bool, typer.Option(help="Skip records already on disk")] = True,
    overwrite: Annotated[
        bool,
        typer.Option(help="With --no-resume, first delete this dataset's existing raw records"),
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
        typer.Option(
            help="Proceed even though the default dense model's Gate 1 reference is unverified"
        ),
    ] = False,
    dense_model: Annotated[
        str | None,
        typer.Option(
            help="Override the dense first-stage model the dense-family arms rerank (default: config/models.yaml)"
        ),
    ] = None,
    dense_model_dim: Annotated[
        int | None,
        typer.Option(
            help="Dimension of --dense-model, if it can't be auto-resolved from FastEmbed's registry"
        ),
    ] = None,
    analyse: Annotated[bool, typer.Option(help="Aggregate after the run completes")] = True,
    allow_partial: Annotated[
        bool,
        typer.Option(
            help="Aggregate even if some (arm, depth) units end up with incomplete records. Never use this for a result you intend to trust."
        ),
    ] = False,
    allow_disk_bound: Annotated[
        bool,
        typer.Option(
            help="Run even when a collection's working set does not fit in Qdrant's memory. Its latency is then disk-bound for the large-vector arms only."
        ),
    ] = False,
    results_dir: Annotated[
        str | None,
        typer.Option(
            help="Write raw records, aggregates, query samples and the manifest here instead of results/ (relative to the repo root). The embedding cache is shared."
        ),
    ] = None,
    cross_encoder_model: Annotated[
        str | None,
        typer.Option(
            help="Override the cross-encoder the *_cross_encoder arms rerank with. Either a FastEmbed model name, or HF_REPO:onnx/file.onnx to register one FastEmbed does not ship."
        ),
    ] = None,
    muvera_setting: Annotated[
        str | None,
        typer.Option(help="Override MUVERA's k_sim,d_proj,n_reps for this run, e.g. 4,16,40"),
    ] = None,
    allow_uncalibrated_muvera: Annotated[
        bool,
        typer.Option(
            help="Run muvera_late_interaction_rescore without a passing calibration, to measure it end to end. Recorded in the manifest."
        ),
    ] = False,
) -> None:
    """Index, measure memory, execute the interleaved plan, and aggregate."""
    cfg = load_config()
    if results_dir:
        cfg = cfg.with_results_dir(results_dir)
        _echo(f"writing results under {cfg.paths['results']}")
    if depths:
        cfg = cfg.with_depths_override(_parse_depths(depths))
    if cross_encoder_model:
        from .config import RerankerOnnxSource

        if ":" in cross_encoder_model:
            hf, _, file = cross_encoder_model.partition(":")
            update = {"name": hf, "onnx": RerankerOnnxSource(hf=hf, file=file)}
        else:
            update = {"name": cross_encoder_model, "onnx": None}
        cross = cfg.models.cross_encoder.model_copy(update=update)
        cfg = cfg.model_copy(
            update={"models": cfg.models.model_copy(update={"cross_encoder": cross})}
        )
        _echo(f"cross-encoder overridden: {cross.model_id}")

    if muvera_setting or allow_uncalibrated_muvera:
        update: dict = {"allow_uncalibrated": allow_uncalibrated_muvera}
        if muvera_setting:
            k_sim, d_proj, n_reps = (int(part) for part in muvera_setting.split(","))
            update |= {"k_sim": k_sim, "d_proj": d_proj, "n_reps": n_reps}
        update.setdefault("center", cfg.models.muvera.center)
        muvera = cfg.models.muvera.model_copy(update=update)
        cfg = cfg.model_copy(update={"models": cfg.models.model_copy(update={"muvera": muvera})})
        _echo(
            f"muvera: k_sim={muvera.k_sim} d_proj={muvera.d_proj} n_reps={muvera.n_reps} "
            f"(dim {muvera.dim}){', uncalibrated allowed' if allow_uncalibrated_muvera else ''}"
        )
    used_default_dense_model = dense_model is None
    if dense_model:
        from .embedding import resolve_dense_model_dim

        default_dense_model = cfg.models.dense.name
        dim = (
            dense_model_dim if dense_model_dim is not None else resolve_dense_model_dim(dense_model)
        )
        cfg = cfg.model_copy(
            update={
                "models": cfg.models.model_copy(
                    update={
                        "dense": cfg.models.dense.model_copy(
                            update={"name": dense_model, "dim": dim, "onnx": None}
                        )
                    }
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
        _echo(
            "WARNING: proceeding with an unverified Gate 1 baseline (--allow-unverified-baseline)."
        )

    run_all = dataset.strip().lower() == "all"
    names = [spec.name for spec in cfg.datasets.enabled_ordered] if run_all else [dataset]
    requested_arms = _parse_arms(arms, cfg)

    if not resume and not overwrite:
        existing = list(
            cfg.paths["raw"].glob(
                f"{'*' if dataset.strip().lower() == 'all' else dataset}__*.jsonl"
            )
        )
        if existing:
            raise typer.BadParameter(
                f"--no-resume with existing raw records for {dataset!r} would append into "
                f"the same files and double-count them. Pass --overwrite "
                "to delete the existing records first, or use --resume."
            )

    controller = QdrantController(cfg)
    controller.up()
    server_resources = controller.server_resources()
    if server_resources is None:
        raise typer.BadParameter(
            "could not read the Docker host's CPU and memory from `docker info`; the run "
            "would not record what Qdrant ran on."
        )
    _echo(
        f"qdrant server: {server_resources['cpus']} CPUs, "
        f"{_human(server_resources['memory_bytes'])} memory"
    )
    manifest_path = cfg.paths["manifest"]
    merge(
        manifest_path,
        base_manifest(
            cfg,
            image_digest=controller.image_digest(),
            launcher=controller._launcher,
            server_resources=server_resources,
        ),
    )

    encoders = build_encoders(cfg)
    merge(manifest_path, {"model_file_digests": encoders.model_file_digests()})
    from .muvera import matrices_digest as _muvera_matrices_digest

    merge(
        manifest_path,
        {
            "muvera_matrices_digest": _muvera_matrices_digest(
                encoders.muvera, cfg.models.muvera, cfg.models.late_interaction.dim
            )
        },
    )

    for name in names:
        spec = cfg.datasets.get(name)
        # `enabled` governs the default set only. Naming a dataset explicitly runs it,
        # which is how the ones kept out of `--dataset all` for cost are run on demand.
        if not spec.enabled and run_all:
            _echo(f"[skip] {name}: {spec.disabled_reason or 'not in the default set'}")
            merge(
                manifest_path,
                {
                    "exclusions": {
                        name: {"__dataset__": spec.disabled_reason or "not in the default set"}
                    }
                },
            )
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

        # Build only what the requested arms read: a pilot without the MUVERA arm should
        # pay neither its encoding nor its 10,240-dim HNSW build. The official run
        # requests every arm, so it still builds everything.
        from .arms import required_representations, resolve_arms
        from .runner import validate_arm_depth_caps

        validate_arm_depth_caps(cfg)
        resolved_arms, excluded = resolve_arms(cfg, name, include=requested_arms)
        if not resolved_arms:
            raise RuntimeError(f"{name}: no runnable arms; exclusions={excluded}")
        needed = required_representations(resolved_arms)
        representations = [r for r in bundle.spec.representations if r in needed]
        _echo(f"  building the collection every arm queries ({', '.join(representations)})")
        collection = build_collection(
            cfg,
            bundle,
            encoders,
            controller,
            representations=representations,
            restart=not resume_ingestion,
            resume=resume_ingestion,
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
                n_token_vectors=token_vector_count(cfg, bundle)
                if "late_interaction" in needed or "muvera" in needed
                else None,
            )
            verify_expected_representations(cfg, memory, representations)
            if not allow_disk_bound:
                check_fits_in_memory(memory, server_resources["memory_bytes"])
            memory_record = memory.as_dict()
            write_memory_record(cfg, cfg.paths["agg"] / f"{name}__memory.json", memory)
            for vector_name, vec in sorted(memory.vectors.items()):
                _echo(
                    f"    {vector_name:8s} disk={_human(vec.total.disk_bytes)} "
                    f"working_set={_human(vec.total.working_set_bytes)}"
                )
        else:
            _echo("  memory read skipped")

        muvera_calibration: dict | None = None
        if any(arm.name == "muvera_late_interaction_rescore" for arm in resolved_arms):
            from .arms import muvera_calibration_status

            passed, reason = muvera_calibration_status(cfg, name)
            muvera_calibration = {"passed": passed, "reason": reason}
            _echo(f"  muvera calibration: {'PASS' if passed else 'FAIL'} — {reason}")

        summary = run_dataset(
            cfg,
            bundle,
            encoders,
            collection,
            include_arms=requested_arms,
            resume=resume,
            log=_echo,
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
                # Recorded whether it passed or not: with require_calibration off the arm
                # still runs, so the verdict is the only thing marking its FDE as
                # under-performing on this dataset.
                "muvera_calibration": muvera_calibration,
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
        typer.Option(
            help="Aggregate even if some (arm, depth) units have incomplete records. Never use this for a result you intend to trust."
        ),
    ] = False,
    results_dir: Annotated[
        str | None,
        typer.Option(help="Read and write under this results directory instead of results/"),
    ] = None,
) -> None:
    """Re-aggregate from existing raw records without re-running any query."""
    cfg = load_config()
    if results_dir:
        cfg = cfg.with_results_dir(results_dir)
    run_all = dataset.strip().lower() == "all"
    names = [spec.name for spec in cfg.datasets.enabled_ordered] if run_all else [dataset]
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
    collection = build_collection(cfg, bundle, encoders, controller, restart=False, resume=True)
    memory_read = read_collection_memory(
        cfg,
        collection,
        dataset=dataset,
        n_documents=len(bundle.documents),
        n_token_vectors=token_vector_count(cfg, bundle),
    )
    verify_expected_representations(cfg, memory_read, bundle.spec.representations)
    write_memory_record(cfg, cfg.paths["agg"] / f"{dataset}__memory.json", memory_read)
    _echo(json.dumps(memory_read.as_dict(), indent=2, sort_keys=True))


@app.command("tune-muvera")
def tune_muvera(
    results_dir: Annotated[
        str | None,
        typer.Option(
            help="Write the tuning record and query splits under this directory instead of results/"
        ),
    ] = None,
) -> None:
    """Sweep models.yaml's pre-declared MUVERA grid on held-out warmup queries.

    Each setting is scored by how much of the exact max_sim top-k its FDE recovers at
    every candidate depth, and its stored size per document is compared with the
    late-interaction multivectors it prefetches for. The selected setting is the
    smallest FDE that reaches muvera.min_recall at muvera.calibration_depth on every
    tuning dataset. Evaluated queries are never used.
    """
    from itertools import product

    from src.muvera import calibrate, calibration_inputs

    cfg = load_config()
    if results_dir:
        cfg = cfg.with_results_dir(results_dir)
    muvera_cfg = cfg.models.muvera
    grid = muvera_cfg.tuning_grid
    if grid is None:
        raise typer.BadParameter("models.yaml muvera.tuning_grid is not set")
    token_dim = cfg.models.late_interaction.dim
    settings = sorted(
        (
            muvera_cfg.model_copy(update={"k_sim": k_sim, "d_proj": d_proj, "n_reps": n_reps})
            for k_sim, d_proj, n_reps in product(grid.k_sim, grid.d_proj, grid.n_reps)
        ),
        key=lambda m: (m.dim, m.k_sim, m.d_proj, m.n_reps),
    )
    encoders = build_encoders(cfg)

    rows: list[dict] = []
    for dataset in grid.datasets:
        bundle = load_dataset_bundle(cfg, dataset)
        doc_tokens, query_tokens, mean = calibration_inputs(cfg, bundle, encoders)
        multivector_bytes = float(np.mean([len(d) for d in doc_tokens])) * token_dim * 4
        depths = tuple(sorted({*cfg.depths_for(dataset), cfg.models.muvera.calibration_depth}))
        _echo(
            f"\n{dataset}: {len(doc_tokens)} documents, {len(query_tokens)} held-out warmup "
            f"queries; multivectors {multivector_bytes / 1024:.0f} KB per document"
        )
        for setting in settings:
            result = calibrate(
                setting,
                token_dim,
                cfg.run.seeds.muvera,
                doc_tokens,
                query_tokens,
                depths=depths,
                truth_k=cfg.run.execution.final_limit,
                mean=mean,
                workers=cfg.run.ingestion.muvera_workers or None,
            )
            fde_bytes = setting.dim * 4
            rows.append(
                {
                    "dataset": dataset,
                    "k_sim": setting.k_sim,
                    "d_proj": setting.d_proj,
                    "n_reps": setting.n_reps,
                    "dim": setting.dim,
                    "fde_bytes_per_doc": fde_bytes,
                    "multivector_bytes_per_doc": multivector_bytes,
                    "larger_than_multivectors": fde_bytes > multivector_bytes,
                    "recall_at_calibration_depth": result.recall_at[muvera_cfg.calibration_depth],
                    "passes": result.beats_bag_of_tokens
                    and result.recall_at[muvera_cfg.calibration_depth] >= muvera_cfg.min_recall,
                    **result.as_dict(),
                }
            )
            row = rows[-1]
            _echo(
                f"  k_sim={setting.k_sim} d_proj={setting.d_proj:<2d} n_reps={setting.n_reps:<2d} "
                f"dim={setting.dim:<6d} {fde_bytes / 1024:4.0f} KB"
                f"{' > multivectors' if row['larger_than_multivectors'] else '               '} "
                + " ".join(f"r@{d}={result.recall_at[d]:.2f}" for d in depths)
                + ("  PASS" if row["passes"] else "")
            )

    def passes_everywhere(setting) -> bool:
        return all(
            row["passes"]
            for row in rows
            if (row["k_sim"], row["d_proj"], row["n_reps"])
            == (setting.k_sim, setting.d_proj, setting.n_reps)
        )

    selected = next((s for s in settings if passes_everywhere(s)), None)
    record = {
        "late_interaction_model": cfg.models.late_interaction.name,
        "grid": grid.model_dump(),
        "min_recall": muvera_cfg.min_recall,
        "calibration_depth": muvera_cfg.calibration_depth,
        "rule": f"smallest FDE whose recall@{muvera_cfg.calibration_depth} reaches min_recall "
        "and beats bag-of-tokens on every tuning dataset, on held-out warmup queries",
        "selected": None
        if selected is None
        else {
            "k_sim": selected.k_sim,
            "d_proj": selected.d_proj,
            "n_reps": selected.n_reps,
            "dim": selected.dim,
        },
        "rows": rows,
    }
    out = cfg.paths["agg"] / "muvera_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True))
    if selected is None:
        _echo(
            f"\nno setting in the grid reaches recall {muvera_cfg.min_recall:.2f} on every tuning dataset"
        )
    else:
        _echo(
            f"\nselected: k_sim={selected.k_sim} d_proj={selected.d_proj} "
            f"n_reps={selected.n_reps} (dim {selected.dim})"
        )
    _echo(f"[tuning] {out}")


@app.command("calibrate-muvera")
def calibrate_muvera(
    dataset: Annotated[str, typer.Option()] = "nfcorpus",
    results_dir: Annotated[
        str | None,
        typer.Option(
            help="Write the calibration artifact and query split under this directory instead of results/"
        ),
    ] = None,
) -> None:
    """Measure FDE prefetch recall against exact max_sim on REAL late-interaction embeddings.

    Runs over the whole corpus and the dataset's held-out warmup queries. Required before
    publishing any muvera number. Synthetic vectors cannot validate the encoder: on
    isotropic random tokens the FDE cannot beat a plain summed-token dot product at any
    parameter setting.
    """
    from src.muvera import calibrate, calibration_inputs

    cfg = load_config()
    if results_dir:
        cfg = cfg.with_results_dir(results_dir)
    encoders = build_encoders(cfg)
    bundle = load_dataset_bundle(cfg, dataset)

    doc_tokens, query_tokens, mean = calibration_inputs(cfg, bundle, encoders)
    _echo(
        f"{len(doc_tokens)} documents, {len(query_tokens)} held-out warmup queries"
        f"{', mean-centered' if mean is not None else ''}"
    )

    result = calibrate(
        cfg.models.muvera,
        cfg.models.late_interaction.dim,
        cfg.run.seeds.muvera,
        doc_tokens,
        query_tokens,
        depths=tuple(sorted({*cfg.depths_for(dataset), cfg.models.muvera.calibration_depth})),
        truth_k=cfg.run.execution.final_limit,
        mean=mean,
        workers=cfg.run.ingestion.muvera_workers or None,
    )
    payload = result.as_dict() | {
        "config": cfg.models.muvera.model_dump(),
        "late_interaction_model": cfg.models.late_interaction.name,
        "corpus": f"{dataset}, all {len(doc_tokens)} documents",
        "queries": f"{len(query_tokens)} held-out warmup queries",
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
        resources = controller.server_resources()
        _echo(
            "qdrant server     : "
            + (
                f"{resources['cpus']} CPUs, {_human(resources['memory_bytes'])} memory"
                if resources
                else "unknown (docker info failed)"
            )
        )
    except Exception as exc:  # noqa: BLE001  # doctor reports any failure
        _echo(f"qdrant            : NOT ready ({exc}). Run: docker compose up -d")

    ref = cfg.models.dense.reference
    if not ref.verified:
        _echo(
            f"WARNING: Gate 1 reference {ref.metric}={ref.value} on {ref.dataset} is marked "
            "verified: false. Confirm it against the MTEB leaderboard before publishing."
        )
    if not cfg.models.muvera_calibrated:
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
