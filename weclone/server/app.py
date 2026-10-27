"""Unified WeClone web and optional inference server."""

from pathlib import Path

from weclone.server.profile_review import create_app as create_profile_app


def create_app(
    *,
    run: Path | None = None,
    static_dir: Path | None = None,
    inference: bool = False,
):
    """Serve one build-profile run, the newest one when run is None."""
    inference_app = None
    if inference:
        from weclone.server.api_service import create_inference_app

        inference_app = create_inference_app()
    app = create_profile_app(
        None,
        run / "profile_hierarchy" / "profile_hierarchy.json" if run is not None else None,
        static_dir,
        inference_router=inference_app.router if inference_app is not None else None,
    )
    if inference_app is not None:
        app.root_path = inference_app.root_path
    return app


def serve(*, host="127.0.0.1", port=5175, run=None, inference=False):
    import uvicorn

    app = create_app(run=run, inference=inference)
    uvicorn.run(app, host=host, port=port)
