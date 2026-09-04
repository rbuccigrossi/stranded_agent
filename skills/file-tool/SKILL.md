---
name: file-tool
description: Inspect and edit UTF-8 text files with line-aware operations. Use when reviewing, searching, or making precise text-file changes.
---

# File tool

Use `python skills/file-tool/scripts/file_tool.py` for simple cross-platform text-file inspection and editing.

Commands:
- `info PATH` — size, lines, characters, and MIME type.
- `view PATH [START END]` — display numbered lines.
- `search PATH PATTERN` — find matching numbered lines.
- `replace PATH OLD NEW [COUNT]` — replace text; add `--backup` to create `.bak`.
- `edit-lines PATH START END TEXT` — replace an inclusive line range.
- `insert-lines PATH LINE TEXT` — insert at a one-based line.
- `delete-lines PATH START END` — delete an inclusive line range.
- `copy PATH DESTINATION` — copy a file with metadata.

Text is read and written as UTF-8. Editing is direct; use `--backup` or Git when a backup is wanted.
