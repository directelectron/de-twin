# Development and releases

## Setting up

```
git clone https://github.com/CSSFrancis/de-twin
cd de-twin
uv sync --group dev          # the package (editable), deapi, pytest, towncrier, ruff
uv run pytest                # ~340 tests, a few minutes
```

On Windows, `cpp\build.bat` builds the stand-alone check of DE-Server's
`ExternalFrameSource` (needs Visual Studio); the layout and C++ ↔ Python interop tests run
when it has been built and are skipped otherwise.

Build the documentation with:

```
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html -W --keep-going
```

## Changelog entries

Every pull request that changes something a user would notice adds a news fragment to
`upcoming_changes/`, named `{PR number}.{type}.rst` (types: `api_change`, `new_feature`,
`bugfix`, `deprecation`, `removal`, `doc`, `maintenance`). See
`upcoming_changes/README.rst`. Preview the next release notes with:

```
uvx towncrier build --draft --version NEXT
```

## Making a release

Releases are prepared by a workflow and published by hand on GitHub.

1. **Prepare.** In GitHub, *Actions → Prepare Release → Run workflow*. Pick the bump
   (`minor`, `bugfix`, `major`; `pre-release` / `beta` for betas, `finalize` to turn a beta
   into the stable release). The workflow bumps the version in `pyproject.toml`, builds
   `CHANGELOG.rst` from the fragments (and deletes them), adds the version to the docs
   switcher, and opens a `release/vX.Y.Z` pull request.
2. **Review and merge** that pull request once CI passes. Edit `CHANGELOG.rst` in the PR if
   the notes need work.
3. **Publish on GitHub.** *Releases → Draft a new release*: tag `vX.Y.Z` on `main` (the tag
   must match `pyproject.toml`), title `vX.Y.Z`, paste the version's `CHANGELOG.rst` section as
   the notes, tick *pre-release* for a beta, and **Publish release**.
4. Publishing runs the **Release** workflow: it checks the tag matches the package version,
   builds the wheel and sdist, uploads them to PyPI by trusted publishing, and attaches them
   to the GitHub Release. Pushing the tag also builds that version's documentation.

### One-time setup

- **PyPI**: on pypi.org, *Your projects → Publishing → Add a pending publisher*: project
  `de-twin`, owner `CSSFrancis`, repository `de-twin`, workflow `release.yml`,
  environment `pypi`.
- **GitHub**: create an environment named `pypi` (*Settings → Environments*); optionally
  require a reviewer for it. Enable GitHub Pages from the `gh-pages` branch once the Docs
  workflow has pushed it.
