#!/usr/bin/env python3
"""Pre-commit hook to validate conventional commit messages.

This hook ensures that commit messages follow the Conventional Commits
specification with an approved list of types.
"""

import re
import sys
from pathlib import Path

# Allowed conventional commit types
ALLOWED_TYPES = [
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "chore",
    "build",
    "ci",
    "revert",
]

# Regex pattern for conventional commit format
# Matches: type: message OR type(scope): message
PATTERN = re.compile(rf"^({'|'.join(ALLOWED_TYPES)})(\(.+?\))?:\s.+")


def validate_commit_message(commit_msg_file: Path) -> bool:
    """Validate the first non-empty line of a commit message.

    Parameters
    ----------
    commit_msg_file : Path
        Path to the commit message file.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    try:
        content = commit_msg_file.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading commit message file: {e}", file=sys.stderr)
        return False

    # Get first non-empty line
    first_line = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    if not first_line:
        print("Error: Commit message is empty.", file=sys.stderr)
        return False

    # Validate against pattern
    if not PATTERN.match(first_line):
        print("\n❌ Invalid commit message format!\n", file=sys.stderr)
        print(
            "Commit messages must follow Conventional Commits format:\n",
            file=sys.stderr,
        )
        print("  type: description", file=sys.stderr)
        print("  type(scope): description\n", file=sys.stderr)
        print("Allowed types:", file=sys.stderr)
        for commit_type in ALLOWED_TYPES:
            print(f"  - {commit_type}", file=sys.stderr)
        print(f"\nYour commit message:\n  {first_line}\n", file=sys.stderr)
        return False

    return True


def main() -> int:
    """Run the commit message validation hook.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    if len(sys.argv) < 2:
        print("Error: No commit message file provided.", file=sys.stderr)
        return 1

    commit_msg_file = Path(sys.argv[1])

    if not commit_msg_file.exists():
        print(
            f"Error: Commit message file not found: {commit_msg_file}",
            file=sys.stderr,
        )
        return 1

    if validate_commit_message(commit_msg_file):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
