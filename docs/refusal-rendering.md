# Refusal rendering

Spellbook records provider refusals as canonical `IRRefusalBlock` events. The
event retains ordered partial text, thinking summaries, partial tool JSON, and
provider metadata. Rendering is a derived request-surface policy and never
changes that transcript truth.

## Production default

The default policy is `partial-note`:

1. Preserve model-visible partial text as an assistant response.
2. Omit thinking summaries, partial tool JSON, the refusal envelope, and
   provider stop metadata from subsequent model requests.
3. Follow the partial response with a user-role system note:

```xml
<spellbook>
A system interruption occurred. Your previous response was not delivered.
</spellbook>
```

When no partial text exists, only the system note is rendered.

This projection prevents synthetic refusal records from recursively loading
future generations while preserving the model-authored portion of the turn and
honestly recording that delivery was interrupted.

## Legacy transcripts

Older transcripts store a flattened assistant string ending in `<refusal>`.
Spellbook strictly parses these strings into a derived canonical refusal during
request construction. Historical transcript bytes and semantic-block
coordinates remain unchanged.

Malformed or ambiguous refusal envelopes fail loudly. They are not silently
reinterpreted.

## Policy persistence and precedence

New sessions record their effective policy as an `IRRuntimeConfigRecord` in the
`refusal_rendering` namespace. On resume, policy selection is:

1. Explicit runtime composition override
2. Latest transcript `refusal_rendering` record
3. Production `partial-note` default

## Existing-transcript amendment

Preflight is read-only:

```bash
python -m scripts.amend_refusals path/to/transcript.jsonl
```

Create a copied rehearsal transcript:

```bash
python -m scripts.amend_refusals SOURCE \
  --output /tmp/refusal-rehearsal/transcript.jsonl
```

An in-place append requires a locked source hash and a new backup path:

```bash
python -m scripts.amend_refusals SOURCE --apply \
  --expect-sha256 SHA256 \
  --backup path/to/transcript.pre-refusal-amendment.jsonl
```

This operation appends one operator-owned policy record. It does not rewrite
historical turns. A refusal turn that has only lifecycle stop state and no
assistant refusal event is reported and left unchanged rather than being given
synthetic model output.
