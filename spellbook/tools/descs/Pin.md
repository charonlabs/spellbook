Pin a semantic block or a summary facet to protect it from compaction.

Block pins preserve the entire block at full fidelity and prevent automatic compaction.
Facet pins preserve one summary facet at full fidelity while still allowing the rest of the block to compact.

Usage:
- **Pinning a semantic block:** `Pin(block_idx=<idx>, reason="...")`
- **Pinning a facet:** `Pin(block_idx=<idx>, facet_id="<facet-id>", reason="...")`
- A block pin protects the whole block and prevents automatic compaction.
- A facet pin lets the block compact to summary, but preserves that facet as original conversation turns inside the summary rendering.
- The `reason` param is for you, for later - it will be shown to future-you as a reminder of why you found it sacred in the first place.
- Use `Reflect` first to get the `block_idx` values associated with the blocks in your context.
