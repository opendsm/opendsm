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
  against `master` from a branch prefixed `feature/` (new features) or `fix/`
  (bug fixes), describe the change, and follow the checklist in the pull
  request template. CI runs the tests on every pull request.
* Contributions that add new required dependencies to the library will be
  given a more thorough review to ensure that those dependency additions
  1) do not pose a security risk and 2) are absolutely necessary.
* Contributions that allow for data exfiltration will not be accepted. The
  library sends no user data over the network, and its only request downloads
  the test data published in the opendsm repository.

Author responsibility
---------------------

AI tools (e.g., coding assistants and large language models) may be used for
any part of a contribution, including code, tests, documentation, and pull
request descriptions. Their use does not need to be disclosed, and the project
does not ask which tools were used. The author is responsible for all of the
contribution, however it was produced, and the rules below apply to every
contribution in the same way.

* You are responsible for everything you submit. Understand each change well
  enough to explain it in review. Maintainers may close a pull request whose
  author cannot, because review depends on the author answering questions
  about the change.
* Your `Signed-off-by` line certifies, under the Developer Certificate of
  Origin, that you have the right to submit every part of the contribution,
  including any part you did not write by hand.
* Statements in a pull request description are yours. Claim only what you
  have checked (e.g., that the suite passes on a platform or that model
  results are unchanged), and say what you left unchecked.
* Do not change, skip, or loosen an existing test or snapshot to make a
  failing test pass. When a pull request changes an existing test or
  snapshot, its description explains why.
* Bug reports include a short, self-contained snippet that reproduces the
  problem, as the bug report template asks. Report security problems
  privately as [SECURITY.md](SECURITY.md) describes, never in a public issue.

Contributor maintenance responsibility
--------------------------------------

Contributions of all kinds are encouraged, and we do not require contributors
to be responsible for ongoing support of patches they make. However, because
we accept "toss over the wall" contributions, contributions deemed by the
maintainers to be too difficult to maintain will not be accepted.

Testing
-------

Please write tests for all new features, and pair each bug fix with a
regression test that fails without the fix.

Test coverage must not fall. The floor is `fail_under` in `pyproject.toml`,
measured over the full suite with numba JIT disabled so that JIT-compiled
functions are visible to the tracer. CI does not measure coverage, so run
`tox -e coverage` locally before opening a pull request (the run is serial
and memory-intensive). Lines that are impractical to test may carry a
`# pragma: no cover` comment, with a nearby comment giving the reason.

Tests are run using the following commands (flags are passed to the py.test
executable):

```
docker-compose run --rm test                            # run all tests

# cheat sheet of variations
docker-compose run --rm test tests/eemeter/daily_model  # run a specific suite
docker-compose run --rm test -k compute                 # filter for tests with `compute` in the name
```

Test configuration can be found in `pyproject.toml` (`[tool.pytest.ini_options]`), `tox.ini`, and `tests/conftest.py`.

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

All source files must apply the following license header:

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
[Developer Certificate of Origin](https://developercertificate.org/). Sign off
every commit with `git commit -s`, which adds a `Signed-off-by` line with your
name and email. The DCO check on each pull request fails when a commit lacks
it. The text of the certificate is copied below:

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

The charter for the open source project can be found in [CHARTER](CHARTER.md) and
contains the following sections:

1. Mission and Scope of the Project
2. Technical Steering Committee
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

5. bump version - edit `version` in `pyproject.toml` with the new version
6. Rename `Development` section with the new version in CHANGELOG.md
7. commit changes
8. push branch and create PR
9. merge release branch to master after tests pass and an approved review
10. create GitHub release describing changes, creating tag `vX.Y.Z` to match the new `pyproject.toml` version
11. approve the resulting pypi-publish action

Other resources
---------------

- [README](README.md): basic project information shown on the PyPI project page.
- [MAINTAINERS](MAINTAINERS.md): an ordered list of project maintainers.
- [LICENSE](LICENSE): Apache 2.0.
- [CHARTER](CHARTER.md): open source project charter.
- [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md): code of conduct for contributors.
- [SECURITY](SECURITY.md): how to report a vulnerability.
