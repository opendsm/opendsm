Contributing
============

Guidelines
----------

* Make sure you follow PEP 008 style guide conventions. You can check PEP 008
  compliance with the included script: `docker-compose run --rm blacken`
* Commit messages should start with a capital letter ("Updated models", not "updated models").
* Write new tests and run old tests! Make sure that % test coverage does not decrease.
* Contributions are reviewed by a maintainer before acceptance. To facilitate
  review, please make a [pull request](https://github.com/opendsm/opendsm/pulls/new)
  and provide a description and follow the checklist in the pull request template.
  Tests will be automatically run using GitHub Actions after a pull request is created.
* Prefix new feature branches with `feature/` and bug fix branches with `fix/`
  and make pull requests directly against `master`.
* Contributions that add new required dependencies to the library will be
  given a more thorough review to ensure that those dependency additions
  1) do not pose a security risk and 2) are absolutely necessary.
* Contributions that allow for data exfiltration by making external HTTP or TCP
  requests will not be accepted.

Contributor maintenance responsibility
--------------------------------------

Contributions of all kinds are encouraged, and we do not require contributors
to be responsible for ongoing support of patches they make. However, because
we accept "toss over the wall" contributions, contributions deemed by the
maintainers to be too difficult to maintain will not be accepted.

Testing
-------

Please write unit tests for all new features. At time of writing, this
package has 100% test coverage. We would like to maintain that coverage level,
because 100% coverage is easier than 99% coverage. This does not necessarily
mean that all line are tested. For lines that are sufficiently inconvenient to
test, we maintain 100% test coverage by adding `# pragma: no cover` comments
after the difficult to test lines or blocks.

This helps us stay on top of un-covered sections without sacrificing the
convenience of 100% coverage and without being too overbearing about tests.

Tests are run using the following commands (flags are passed to the py.test
executable):

```
docker-compose run --rm test                         # run all tests

# cheat sheet of variations
docker-compose run --rm test --no-cov                # no coverage
docker-compose run --rm test tests/test_features.py  # run a specific suite
docker-compose run --rm test -k compute              # filter for tests with `compute` in the name
```

Test configuration can be found in `tox.ini`, `pytest.ini`, and `tests/conftest.py`.

When writing tests using py.test fixtures, place fixtures as close as possible
to the test functions or classes that call them.

General Discussion
------------------

Discussions for this project take place on the
[opendsm@lists.lfenergy.org](https://lists.lfenergy.org/g/opendsm/)
mailing list.

License
-------

This project is licensed under [Apache 2.0](LICENSE).

All source files must apply the following SPDX header:

``` python

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
```

Developer Certificate of Origin
-------------------------------

This project uses the
[Developer Certificate of Origin](https://developercertificate.org/),
the text of which is copied below:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Charter
-------

The charter for the open source project can be find in [CHARTER](CHARTER) and
contains the following sections:

1. Mission and Scope of the Project
2. Techincal Steering Committee
3. TSC Voting
4. Compliance with Policies
5. Community Assets
6. General Rules and Operations
7. Intellectual Property Policy
8. Amendments

Release process
---------------

Pre-release

1. create branch off of master named `feature/<examplefeature>` or `fix/<>` and make desired changes.
2. edit CHANGELOG.md with changes under a new section called `Development`
3. create, review, and merge PR for feature/examplefeature
4. repeat steps 1-3 if desired for other features as convenient, though preference is for frequent version bumps

Releasing

5. bump version - edit `__version__.py` with the new version
6. Rename `Development` section with the new version in CHANGELOG.md
7. commit changes
8. push branch and create PR
9. merge release branch to master after tests pass and an approved review
10. create GitHub release describing changes, creating tag `vX.Y.Z` to match new `__version__.py`
11. approve the resulting pypi-publish action

You can use `bump_version.sh` to print out commands that rotate the changelog and bump the package version:

```
./bump_version.sh X.X.X Y.Y.Y
```

Other resources
---------------

- [README](README.rst): basic project information written in RST for PyPI preview.
  Copied and lightly modified for formatting from [docs/index.rst](docs/index.rst)
- [MAINTAINERS](MAINTAINERS.md): an ordered list of project maintainers.
- [LICENSE](LICENSE): Apache 2.0.
- [CHARTER](CHARTER): open source project charter.
- [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md): code of conduct for contributors.
