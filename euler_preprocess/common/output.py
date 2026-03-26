from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ds_crawler import DatasetWriter, ZipDatasetWriter
from euler_loading import MultiModalDataset
from euler_loading.dataset import create_dataset_writer_from_index
from euler_loading.loaders._writer_utils import supports_stream_target

from euler_preprocess.common.io import OutputWriter


PIPELINE_OUTPUTS_MANIFEST_VERSION = 1
_PIPELINE_OUTPUT_STORAGE_KINDS = {"directory", "zip", "file"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class PipelineOutputTargetConfig:
    """Runtime-resolved pipeline output target for a single transform output."""

    slot: str
    model_modality_id: int
    dataset_type: str
    relative_path: str
    path: str
    storage: str

    def __post_init__(self) -> None:
        if not self.slot:
            raise ValueError("pipeline.output_targets[].slot is required")
        if self.model_modality_id <= 0:
            raise ValueError(
                "pipeline.output_targets[].modelModalityId must be positive"
            )
        if not self.dataset_type:
            raise ValueError("pipeline.output_targets[].datasetType is required")
        if not self.relative_path:
            raise ValueError("pipeline.output_targets[].relativePath is required")
        if not self.path:
            raise ValueError("pipeline.output_targets[].path is required")
        if self.storage not in _PIPELINE_OUTPUT_STORAGE_KINDS:
            raise ValueError(
                "pipeline.output_targets[].storage must be one of: "
                "directory, zip, file"
            )
        if self.storage == "zip" and not self.path.lower().endswith(".zip"):
            raise ValueError(
                "pipeline.output_targets[].path must end with .zip when storage='zip'"
            )
        if self.storage == "zip" and not self.relative_path.lower().endswith(".zip"):
            raise ValueError(
                "pipeline.output_targets[].relativePath must end with .zip "
                "when storage='zip'"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineOutputTargetConfig":
        return cls(
            slot=data.get("slot", ""),
            model_modality_id=int(data.get("modelModalityId", 0)),
            dataset_type=data.get("datasetType", ""),
            relative_path=data.get("relativePath", ""),
            path=data.get("path", ""),
            storage=data.get("storage", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "modelModalityId": self.model_modality_id,
            "datasetType": self.dataset_type,
            "relativePath": self.relative_path,
            "storage": self.storage,
        }


@dataclass
class PipelineRuntimeConfig:
    """Pipeline-provided runtime context for transform outputs."""

    output_root: str | None = None
    outputs_manifest_path: str | None = None
    output_targets: list[PipelineOutputTargetConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen_slots: set[str] = set()
        for target in self.output_targets:
            if target.slot in seen_slots:
                raise ValueError(
                    f"pipeline.output_targets contains duplicate slot '{target.slot}'"
                )
            seen_slots.add(target.slot)

    def get_output_target(self, slot: str) -> PipelineOutputTargetConfig | None:
        for target in self.output_targets:
            if target.slot == slot:
                return target
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineRuntimeConfig":
        output_targets_data = data.get("output_targets")
        if output_targets_data is None:
            parsed_targets: list[PipelineOutputTargetConfig] = []
        else:
            if not isinstance(output_targets_data, list):
                raise ValueError("pipeline.output_targets must be a list")
            parsed_targets = [
                PipelineOutputTargetConfig.from_dict(entry)
                for entry in output_targets_data
            ]

        output_root = data.get("output_root")
        outputs_manifest_path = data.get("outputs_manifest_path")
        return cls(
            output_root=output_root if isinstance(output_root, str) else None,
            outputs_manifest_path=(
                outputs_manifest_path
                if isinstance(outputs_manifest_path, str)
                else None
            ),
            output_targets=parsed_targets,
        )


def parse_pipeline_config(config: dict[str, Any]) -> PipelineRuntimeConfig | None:
    pipeline_data = config.get("pipeline")
    if pipeline_data is None:
        return None
    if not isinstance(pipeline_data, dict):
        raise ValueError("pipeline must be a JSON object")
    return PipelineRuntimeConfig.from_dict(pipeline_data)


def _write_pipeline_outputs_manifest(
    manifest_path: Path,
    output_targets: list[PipelineOutputTargetConfig],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": PIPELINE_OUTPUTS_MANIFEST_VERSION,
                "outputs": [target.to_dict() for target in output_targets],
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def _set_stream_name(stream: Any, basename: str) -> None:
    try:
        stream.name = basename
    except Exception:
        pass


class LegacyOutputBackend:
    """Legacy output backend used by direct transform invocation."""

    is_source_backed = False

    def __init__(self, root: str | Path) -> None:
        self._writer = OutputWriter(root)
        self.root = self._writer.root

    def write(
        self,
        sample: dict[str, Any],
        value: Any,
        *,
        default_path: Path | None = None,
    ) -> Path:
        if default_path is None:
            raise ValueError("default_path is required for legacy outputs")

        self._writer.mkdir(default_path.parent)
        suffix = default_path.suffix.lower()
        if suffix == ".npy":
            self._writer.save_depth_npy(default_path, value)
            return default_path
        if suffix in _IMAGE_EXTENSIONS:
            self._writer.save_image(default_path, value)
            return default_path
        raise ValueError(f"Unsupported legacy output suffix: {suffix}")

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        self._writer.mkdir(path.parent)
        self._writer.write_json(path, data)

    def finalize(self) -> None:
        self._writer.close()


class SourceBackedOutputBackend:
    """Source-aware output backend mirroring an input modality's writer metadata."""

    is_source_backed = True

    def __init__(
        self,
        *,
        source_modality: str,
        root: Path,
        dataset_writer: DatasetWriter | ZipDatasetWriter,
        modality_writer: Any,
        modality_meta: dict[str, Any] | None,
        pipeline_manifest_path: Path | None = None,
        pipeline_manifest_targets: list[PipelineOutputTargetConfig] | None = None,
    ) -> None:
        self.source_modality = source_modality
        self.root = root
        self.dataset_writer = dataset_writer
        self.modality_writer = modality_writer
        self.modality_meta = modality_meta
        self.pipeline_manifest_path = pipeline_manifest_path
        self.pipeline_manifest_targets = pipeline_manifest_targets or []

    def write(
        self,
        sample: dict[str, Any],
        value: Any,
        *,
        default_path: Path | None = None,
    ) -> Path:
        del default_path

        sample_id = sample.get("id", "?")
        full_id = str(sample.get("full_id") or f"/{sample_id}")
        meta = sample.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(
                "Source-backed outputs require samples from MultiModalDataset "
                "so sample['meta'] is available."
            )

        source_meta = meta.get(self.source_modality)
        if not isinstance(source_meta, dict) or "path" not in source_meta:
            raise ValueError(
                f"Source-backed output for modality '{self.source_modality}' "
                "requires sample['meta'][source_modality]['path']."
            )

        basename = Path(str(source_meta["path"])).name
        relative_path = str(source_meta["path"])
        source_meta_copy = dict(source_meta)

        if isinstance(self.dataset_writer, ZipDatasetWriter):
            if supports_stream_target(self.modality_writer):
                with self.dataset_writer.open(
                    full_id,
                    basename,
                    source_meta=source_meta_copy,
                ) as stream:
                    _set_stream_name(stream, basename)
                    self.modality_writer(stream, value, self.modality_meta)
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_path = Path(tmpdir) / basename
                    self.modality_writer(str(temp_path), value, self.modality_meta)
                    self.dataset_writer.write(
                        full_id,
                        basename,
                        temp_path.read_bytes(),
                        source_meta=source_meta_copy,
                    )
            return Path(f"{self.dataset_writer.root}::{relative_path}")

        target_path = self.dataset_writer.get_path(
            full_id,
            basename,
            source_meta=source_meta_copy,
        )
        self.modality_writer(str(target_path), value, self.modality_meta)
        return target_path

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        raise RuntimeError(
            "Source-backed outputs do not support auxiliary JSON sidecars."
        )

    def finalize(self) -> None:
        self.dataset_writer.save_index()
        if self.pipeline_manifest_path and self.pipeline_manifest_targets:
            _write_pipeline_outputs_manifest(
                self.pipeline_manifest_path,
                self.pipeline_manifest_targets,
            )


def _resolve_output_slot(
    config: dict[str, Any],
    transform_class: type,
) -> str | None:
    explicit_slot = config.get("output_slot")
    if explicit_slot is not None:
        if not isinstance(explicit_slot, str) or not explicit_slot:
            raise ValueError("output_slot must be a non-empty string")
        return explicit_slot

    output_slot = getattr(transform_class, "OUTPUT_SLOT", None)
    if isinstance(output_slot, str) and output_slot:
        return output_slot

    source_modality = getattr(transform_class, "SOURCE_MODALITY", None)
    if isinstance(source_modality, str) and source_modality:
        return source_modality

    return None


def _select_pipeline_target(
    config: dict[str, Any],
    pipeline: PipelineRuntimeConfig | None,
    transform_class: type,
) -> PipelineOutputTargetConfig | None:
    if pipeline is None or not pipeline.output_targets:
        return None

    slot = _resolve_output_slot(config, transform_class)
    if slot is not None:
        target = pipeline.get_output_target(slot)
        if target is not None:
            return target
        if config.get("output_slot") is not None:
            raise ValueError(
                f"pipeline.output_targets does not contain slot '{slot}'"
            )

    if len(pipeline.output_targets) == 1:
        return pipeline.output_targets[0]

    raise ValueError(
        "pipeline.output_targets contains multiple entries; set top-level "
        "'output_slot' to select the target for this transform."
    )


def _resolve_output_root(
    config: dict[str, Any],
    pipeline: PipelineRuntimeConfig | None,
    pipeline_target: PipelineOutputTargetConfig | None,
) -> Path:
    if pipeline_target is not None:
        if pipeline_target.storage == "file":
            raise ValueError(
                f"Pipeline output target '{pipeline_target.slot}' uses "
                "unsupported storage='file'"
            )
        return Path(pipeline_target.path)

    output_path = config.get("output_path")
    if not output_path and pipeline and pipeline.output_root:
        output_path = pipeline.output_root

    if not output_path:
        raise ValueError(
            "output_path is required unless provided via pipeline.output_root "
            "or pipeline.output_targets[].path"
        )

    return Path(str(output_path))


def prepare_output_backend(
    config: dict[str, Any],
    dataset: MultiModalDataset,
    transform_class: type,
) -> SourceBackedOutputBackend:
    """Create a source-backed output backend for a transform run."""

    source_modality = getattr(transform_class, "SOURCE_MODALITY", None)
    if not isinstance(source_modality, str) or not source_modality:
        raise ValueError(
            f"{transform_class.__name__} does not declare SOURCE_MODALITY"
        )

    pipeline = parse_pipeline_config(config)
    pipeline_target = _select_pipeline_target(config, pipeline, transform_class)
    root = _resolve_output_root(config, pipeline, pipeline_target)
    zip_mode = (
        pipeline_target.storage == "zip"
        if pipeline_target is not None
        else root.suffix.lower() == ".zip"
    )

    try:
        modality_writer = dataset.get_writer(source_modality)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"No euler-loading writer is available for source modality "
            f"'{source_modality}': {exc}"
        ) from exc

    index_output = dataset.get_modality_index(source_modality)
    meta_overrides = getattr(transform_class, "OUTPUT_INDEX_META_OVERRIDES", None)
    if isinstance(meta_overrides, dict) and meta_overrides:
        merged_meta = dict(index_output.get("meta") or {})
        merged_meta.update(meta_overrides)
        index_output = dict(index_output)
        index_output["meta"] = merged_meta

    dataset_writer = create_dataset_writer_from_index(
        index_output=index_output,
        root=root,
        zip=zip_mode,
    )

    pipeline_manifest_path = None
    if (
        pipeline is not None
        and pipeline.outputs_manifest_path
        and pipeline_target is not None
    ):
        pipeline_manifest_path = Path(pipeline.outputs_manifest_path)

    return SourceBackedOutputBackend(
        source_modality=source_modality,
        root=root,
        dataset_writer=dataset_writer,
        modality_writer=modality_writer,
        modality_meta=index_output.get("meta"),
        pipeline_manifest_path=pipeline_manifest_path,
        pipeline_manifest_targets=[pipeline_target] if pipeline_target else [],
    )
