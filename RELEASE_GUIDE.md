# Release Guide

This package follows the GenesisAeon ecosystem release process.

## Versioning

We use [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- **MAJOR** — breaking changes to the public API.
- **MINOR** — new features, backwards-compatible.
- **PATCH** — bug fixes, documentation, dependency bumps.

## Release types

| Tag pattern | Channel | Where it publishes |
|---|---|---|
| `vX.Y.Z` | Production | PyPI, GitHub Release, Zenodo (if integration enabled) |
| `vX.Y.Z-rc.N`, `-alpha.N`, `-beta.N` | Canary | TestPyPI, GitHub pre-release |

## How to cut a release

1. Ensure `CHANGELOG.md` has an entry for the new version under `## [X.Y.Z]`.
2. Ensure `pyproject.toml`'s `[project].version` matches.
3. Ensure `.zenodo.json`'s `"version"` field matches.
4. Commit these changes (if any) to `main`.
5. Tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
6. `.github/workflows/release.yml` builds, tests, and publishes automatically.
7. If Zenodo-GitHub integration is enabled for this repo, a new Zenodo DOI
   version is minted automatically from the GitHub Release using
   `.zenodo.json` metadata.

## Licensing of new contributions

This repository is dual-licensed: source code under GPL-3.0-or-later
(see [`LICENSE-CODE`](./LICENSE-CODE)), documentation under CC BY 4.0
(see [`LICENSE-DOCS`](./LICENSE-DOCS)). New files should carry the
license that matches their content type, and contributors must agree
their contributions are made under these terms.

## Dependency pins within the GenesisAeon ecosystem

If this package depends on other `GenesisAeon/*` packages, pin them with
`>=` lower bounds matching the minimum version that provides the API this
package relies on. Do not pin exact versions (`==`) for ecosystem
dependencies.
