Performs exact string replacements in files.

Usage:
- Before using this tool, you should use the `Read` tool first to read the file's contents. This tool might fail if you did not read the file first.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears *after* the line number prefix. The line number prefix format is: `spaces + line number + tab`. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- The edit will fail if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
