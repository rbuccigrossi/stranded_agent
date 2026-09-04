# Project skills

Put `SKILL.md` files in this directory, one skill per subdirectory. STRANDed hands this
directory to the Strands `AgentSkills` plugin, which advertises each skill's name and
description in the system prompt and loads the full instructions on demand. The agent
can create or update project skills here through its shell tool. Framework-provided
skills live separately under `_builtin/`.
