Execute a skill — load specialized instructions and capabilities into the conversation.

Skills provide domain-specific knowledge and workflows. Available skills are listed in the system prompt. When a task matches a skill's description, invoke it to load full instructions before proceeding.

Usage:
- `Skill(name="summon")` — load the summon skill's instructions
- `Skill(name="compose", args="review-loop")` — load with arguments
- `Skill(name="browse")` — load the browser automation skill

When users reference a "slash command" or "/<something>" (e.g., "/summon", "/compose"), they are referring to a skill. Use this tool to invoke it.

Important:
- Available skills are listed in your system prompt under `<available-skills>`. Only invoke skills that appear there.
- When a skill matches the user's request, invoke it BEFORE generating any other response about the task.
- Do not invoke a skill that is already loaded in the current conversation. If you see `<skill-content>` tags from a previous activation, follow those instructions directly.
- Skill instructions include a directory path. Resolve any relative paths in the instructions against that directory.
