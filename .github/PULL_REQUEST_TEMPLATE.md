### Your checklist for this pull request

Please review the [guidelines for contributing](../CONTRIBUTING.md) to this repository.

- [ ] Make sure you are requesting to **pull a feature/fix branch** (right side). Don't request your master!
- [ ] Make sure tests pass (`docker-compose run --rm test`) and coverage has not fallen (`tox -e coverage`, which CI does not run).
- [ ] Update the [CHANGELOG.md](../CHANGELOG.md) to describe your changes in a bulleted list under the "Development" section at the top of the changelog. If this section does not exist, create it.
- [ ] Make sure code style follows PEP 008 using `docker-compose run --rm blacken`.
- [ ] Make sure new public functions, classes, and methods have [google-style docstrings](https://mkdocstrings.github.io/python/usage/docstrings/google/). The API reference at https://opendsm.energy auto-generates from them via mkdocstrings (in the [opendsm-website](https://github.com/opendsm/opendsm-website) repo); there are no reference files to edit in this repo.
- [ ] If this pull request changes an existing test or snapshot, the description says why.
- [ ] Make sure that all git commits have the "Signed-off-by" message for
  the Developer Certificate of Origin. When you're making a commit, just add
  the `-s/--signoff` flag (e.g., `git commit -s`).

### Description

\<description of pull request (suggestion: start by copying from your changelog update and elaborate if necessary.)>
