"""Point generated documentation pages at their committed sources."""

from pathlib import Path, PurePosixPath
from urllib.parse import urljoin


def on_page_context(context, *, page, config, nav):
    if not page.edit_url:
        return context

    source = PurePosixPath(page.file.src_uri)
    notebook = Path(page.file.abs_src_path).with_suffix(".ipynb")
    if notebook.is_file():
        target = source.with_suffix(".ipynb").as_posix()
    elif source.parts[0] in {"api_subpackages", "api_modules"}:
        # API pages collect docstrings from the corresponding Python package.
        package = (
            source.stem if source.parts[0] == "api_subpackages" else source.parts[1]
        )
        target = "../../" + package.replace(".", "/")
    else:
        return context

    page.edit_url = urljoin(config.edit_uri.rstrip("/") + "/", target)
    return context
