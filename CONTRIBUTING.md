# Contributing to de-twin

## Development setup

```
git clone https://github.com/directelectron/de-twin
cd de-twin
uv sync --group dev
uv run pytest
```

On Windows, `cpp\build.bat` builds the stand-alone check of DE-Server's frame source so
the C++ layout and interop tests run too.

## Pull requests

- Add or update tests with every behaviour change; the full suite must pass.
- Add a changelog fragment to `upcoming_changes/` named `{PR number}.{type}.rst`
  (see `upcoming_changes/README.rst` for the types).
- Keep the twin's scope: it simulates the instrument and produces data plus ground truth.
  Analysis and reconstruction methods belong in the tools that consume the data.
- If you change the shared-memory layout, change `cpp/ExternalFrameSource.h` and
  `src/de_twin/transport/shm_layout.py` together and bump the protocol version.

## Releases

Maintainers release with the **Prepare Release** workflow, then publish a GitHub Release by
hand, which uploads the package to PyPI. The steps are in `docs/dev/index.md`.
