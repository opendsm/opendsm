---
name: Bug report
about: A report of or request to fix an issue in the opendsm library
---

A complete bug report will help opendsm contributors to reproduce the bug and provide
insight into fixing.

Bug reports must:

1. Include a short, self-contained Python snippet reproducing the problem. You can
  format the code nicely by using GitHub Flavored Markdown:

    ```python
    >>> import opendsm
    >>> import pandas as pd
    >>> meter_data = pd.DataFrame({"start": ..., "value": ...})
    >>> ...  # the call that fails
    ```

2. Include the full version string of opendsm, pandas, and their dependencies. You can
  print them with:

    ```python
    >>> import opendsm
    >>> import pandas as pd
    >>> opendsm.__version__
    >>> pd.show_versions()
    ```
3. Explain why the current behavior is wrong/not desired and what you expect instead.


### Template

**Report installed package versions**
```
opendsm==X.X.X
pandas==X.X.X
scipy==X.X.X
numpy==X.X.X
```

**Describe the bug**
A clear and concise description of what the bug is, including code samples and
tracebacks.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Additional context**
Add any other context about the problem here.
