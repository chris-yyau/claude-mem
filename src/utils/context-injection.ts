
import path from 'path';
import { existsSync, readFileSync, writeFileSync, renameSync, mkdirSync } from 'fs';

export const CONTEXT_TAG_OPEN = '<claude-mem-context>';
export const CONTEXT_TAG_CLOSE = '</claude-mem-context>';

// Atomic write: tmp + rename, so a crash mid-write cannot leave the user's
// AGENTS.md (or any target file) truncated. POSIX rename(2) is atomic on the
// same filesystem.
function atomicWrite(filePath: string, content: string): void {
  const tmpPath = `${filePath}.tmp.${process.pid}`;
  writeFileSync(tmpPath, content, 'utf-8');
  renameSync(tmpPath, filePath);
}

// Strip any literal tag markers that appear inside the incoming context so
// that an unlucky or malicious upstream string can't reshape the existing
// AGENTS.md block boundaries on the next inject. Idempotent.
function sanitizeContextPayload(content: string): string {
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

    // Splice only when both tags exist AND end follows start. An inverted
    // pair (close before open, e.g. from manual edit) would otherwise produce
    // a reordered, content-losing splice — fall through to the append branch.
    if (tagStartIndex !== -1 && tagEndIndex !== -1 && tagEndIndex > tagStartIndex) {
      existingContent =
        existingContent.slice(0, tagStartIndex) +
        wrappedContent +
        existingContent.slice(tagEndIndex + CONTEXT_TAG_CLOSE.length);
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
