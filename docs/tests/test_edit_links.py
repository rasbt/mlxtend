from pathlib import Path

import pytest

build = pytest.importorskip("mkdocs.commands.build").build
load_config = pytest.importorskip("mkdocs.config").load_config
yaml = pytest.importorskip("yaml")


@pytest.mark.parametrize(
    "source, notebook, target",
    [
        ("index.md", False, "docs/sources/index.md"),
        ("user_guide/example.md", True, "docs/sources/user_guide/example.ipynb"),
        ("user_guide/handwritten.md", False, "docs/sources/user_guide/handwritten.md"),
        ("api_subpackages/mlxtend.evaluate.md", False, "mlxtend/evaluate"),
        ("api_modules/mlxtend.evaluate/mcnemar.md", False, "mlxtend/evaluate"),
    ],
)
def test_generated_page_edit_links(tmp_path, source, notebook, target):
    docs = tmp_path / "sources"
    markdown = docs / source
    markdown.parent.mkdir(parents=True)
    markdown.write_text("# Example\n", encoding="utf-8")
    if notebook:
        markdown.with_suffix(".ipynb").write_text("{}", encoding="utf-8")
    project_docs = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((project_docs / "mkdocs.yml").read_text())
    hook = project_docs / settings["hooks"][0]
    config_file = tmp_path / "mkdocs.yml"
    config_file.write_text(
        "site_name: test\n"
        "docs_dir: sources\n"
        "repo_url: https://github.com/rasbt/mlxtend\n"
        "edit_uri: https://github.com/rasbt/mlxtend/tree/master/docs/sources\n"
        f"hooks: ['{hook.as_posix()}']\n",
        encoding="utf-8",
    )
    config = load_config(str(config_file))
    build(config)
    pages = list(Path(config.site_dir).rglob("*.html"))
    assert any(
        f'href="https://github.com/rasbt/mlxtend/tree/master/{target}"'
        in page.read_text(encoding="utf-8")
        for page in pages
    )
