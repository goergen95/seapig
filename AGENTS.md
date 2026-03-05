# Quick commands

Review the Makefile targets to understand how to install dependencies, 
formatting, linting and running tests. 

When done implementing changes ALWAYS run the check target: `make check`

# Rules & expectations

- Typing is MANDATORY for all source code. Tests are source code! Add explicit 
  annotations for functions, methods, and tests. Do not silence missing types 
  with ignores in PRs.
- Testing: TDD recommended — write tests first, then implement.
- Tests must be deterministic (fixed seeds when needed) and use pytest.
