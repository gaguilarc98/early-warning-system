"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines.pipeline import create_coldspells_pipeline, create_rainfall_pipeline, update_coldspell_pipeline, update_rainfall_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    #pipelines = find_pipelines(raise_errors=True)
    pipelines['coldspell'] = create_coldspells_pipeline()
    pipelines['rainfall'] = create_rainfall_pipeline()
    pipelines['update_coldspell'] = update_coldspell_pipeline()
    pipelines['update_rainfall'] = update_rainfall_pipeline()
    pipelines["__default__"] = pipelines['coldspell']
    return pipelines
