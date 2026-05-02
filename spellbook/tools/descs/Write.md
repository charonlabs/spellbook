Writes a file to the local filesystem.

Usage:
- This tool will overwrite the existing file if there is one at the provided path.
- This tool will create parent dirs as needed
- If this is an existing file, you should use the `Read` tool first to read the file's contents. This tool might fail if you did not read the file first.
- Prefer the `Edit` tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.
