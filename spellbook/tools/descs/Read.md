Reads a file from the local filesystem. You can access any text file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- Relative paths resolve from your *original* cwd. Even if you've `cd`'d somewhere, it always resolves from the directory mentioned in your orientation.
- Results are returned using cat -n format, with line numbers starting at 1
- This tool can read images (PNG, JPG, GIF, WEBP). When reading an image file the contents are presented visually.
- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.
