# Skills

[`ddstore/`](ddstore/SKILL.md) is a skill for AI coding agents: how to add DDStore to a PyTorch training code (method, communicator, loader, Slurm job script, common errors). It loads only when a task matches its description.

Install for Claude Code, for all your projects or for one application:

```bash
ln -s /path/to/DDStore/skills/ddstore ~/.claude/skills/ddstore
ln -s /path/to/DDStore/skills/ddstore /path/to/app/.claude/skills/ddstore
```

Other agents can read `ddstore/SKILL.md` and the files it links. `ddstore/scripts/check_env.py` reports the build, methods and settings of the current environment.
