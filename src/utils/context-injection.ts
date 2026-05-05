
import path from 'path';
import { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync, statSync, chmodSync, unlinkSync } from 'fs';
import { randomBytes } from 'crypto';

export const CONTEXT_TAG_OPEN = '<claude-mem-context>';
export const CONTEXT_TAG_CLOSE = '</claude-mem-context>';

// Atomic write: tmp + rename, so a crash mid-write cannot leave the target
// file truncated. POSIX rename(2) is atomic on the same filesystem.
//
// Hardenings:
//  - tmp suffix mixes pid + random bytes so concurrent calls in the same
//    process don't collide on the tmp filename.
//  - on writeFileSync failure (disk full, permission), unlink the (possibly
//    half-written) tmp file before propagating the error so we don't leak
//    `*.tmp.*` siblings.
//  - on rename failure, unlink the staged tmp file too.
//  - if the target already exists, chmod the new inode to match the prior
//    mode so atomic-replace doesn't silently change file permissions.
function atomicWrite(filePath: string, content: string): void {
  const tmpSuffix = `${process.pid}.${randomBytes(6).toString('hex')}`;
  const tmpPath = `${filePath}.tmp.${tmpSuffix}`;
  let priorMode: number | undefined;
  try {
    priorMode = statSync(filePath).mode;
  } catch {
    // File doesn't exist yet — no mode to preserve.
  }

  try {
    writeFileSync(tmpPath, content, 'utf-8');
  } catch (writeError) {
    try { unlinkSync(tmpPath); } catch { /* tmp may not exist */ }
    throw writeError;
  }

  try {
    renameSync(tmpPath, filePath);
  } catch (renameError) {
    try { unlinkSync(tmpPath); } catch { /* best effort */ }
    throw renameError;
  }

  if (priorMode !== undefined) {
    try {
      chmodSync(filePath, priorMode & 0o7777);
    } catch {
      // Non-fatal: write succeeded; mode preservation is best-effort.
    }
  }
}

// Strip any literal tag markers that appear inside the incoming context so
// that an unlucky or malicious upstream string can't reshape the existing
// AGENTS.md block boundaries on the next inject. Idempotent.
function sanitizeContextPayload(content: string): string {
  return content
    .split(CONTEXT_TAG_OPEN).join('')
    .split(CONTEXT_TAG_CLOSE).join('');
}

// Strip every claude-mem context tag (open and close, in any order/count)
// from existing markdown content. Used when the file has a malformed block
// — inverted pair, doubled tags from concurrent writers — so we can append
// a fresh well-formed block without leaving the broken markers behind.
// Without this, fall-through to the append branch would just keep growing
// the file on every inject.
function stripAllContextTags(content: string): string {
  return content
    .split(CONTEXT_TAG_OPEN).join('')
    .split(CONTEXT_TAG_CLOSE).join('');
}

export function injectContextIntoMarkdownFile(
  filePath: string,
  contextContent: string,
  headerLine?: string,
): void {
  const parentDirectory = path.dirname(filePath);
  mkdirSync(parentDirectory, { recursive: true });

  const safeContent = sanitizeContextPayload(contextContent);
  const wrappedContent = `${CONTEXT_TAG_OPEN}\n${safeContent}\n${CONTEXT_TAG_CLOSE}`;

  if (existsSync(filePath)) {
    let existingContent = readFileSync(filePath, 'utf-8');

    const tagStartIndex = existingContent.indexOf(CONTEXT_TAG_OPEN);
    const tagEndIndex = existingContent.indexOf(CONTEXT_TAG_CLOSE);
    const wellFormed =
      tagStartIndex !== -1 && tagEndIndex !== -1 && tagEndIndex > tagStartIndex;

    if (wellFormed) {
      existingContent =
        existingContent.slice(0, tagStartIndex) +
        wrappedContent +
        existingContent.slice(tagEndIndex + CONTEXT_TAG_CLOSE.length);
    } else if (tagStartIndex !== -1 || tagEndIndex !== -1) {
      // Malformed (inverted pair, orphaned tag, or duplicates from a prior
      // race) — strip ALL stray markers before appending the new block, so
      // the file doesn't accumulate broken markers across inject cycles.
      existingContent = stripAllContextTags(existingContent).trimEnd() +
        '\n\n' + wrappedContent + '\n';
    } else {
      existingContent = existingContent.trimEnd() + '\n\n' + wrappedContent + '\n';
    }

    atomicWrite(filePath, existingContent);
  } else {
    if (headerLine) {
      atomicWrite(filePath, `${headerLine}\n\n${wrappedContent}\n`);
    } else {
      atomicWrite(filePath, wrappedContent + '\n');
    }
  }
}
