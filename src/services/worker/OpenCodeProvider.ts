import { execFileSync } from 'child_process';
import { spawnHidden } from '../../shared/spawn.js';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import { logger } from '../../utils/logger.js';
import { buildInitPrompt, buildObservationPrompt, buildSummaryPrompt, buildContinuationPrompt } from '../../sdk/prompts.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../shared/paths.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { ModeManager } from '../domain/ModeManager.js';
import type { ModeConfig } from '../domain/types.js';
import {
  processAgentResponse,
  isAbortError,
  type WorkerRef
} from './agents/index.js';
import { ClassifiedProviderError } from './provider-errors.js';
import { withRetry } from './retry.js';

const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

/**
 * Classify an OpenCode subprocess failure into ClassifiedProviderError.
 */
export function classifyOpenCodeError(input: {
  exitCode?: number | null;
  stderr?: string;
  cause: unknown;
}): ClassifiedProviderError {
  const exitCode = input.exitCode;
  const stderr = input.stderr ?? '';
  const lower = stderr.toLowerCase();

  // Check cause for spawn errors (ENOENT = binary not found)
  const causeCode = (input.cause as any)?.code;
  if (causeCode === 'ENOENT') {
    return new ClassifiedProviderError(
      `OpenCode executable not found. Ensure 'opencode' is installed and in PATH.`,
      { kind: 'unrecoverable', cause: input.cause },
    );
  }

  // Auth failures
  if (
    lower.includes('authentication') ||
    lower.includes('unauthorized') ||
    lower.includes('invalid api key') ||
    lower.includes('api_key_invalid') ||
    lower.includes('api key not valid') ||
    lower.includes('api key expired')
  ) {
    return new ClassifiedProviderError(
      `OpenCode auth error${exitCode !== undefined ? ` (exit ${exitCode})` : ''}`,
      { kind: 'auth_invalid', cause: input.cause },
    );
  }

  // Rate limit
  if (lower.includes('rate limit') || lower.includes('rate_limit') || lower.includes('429')) {
    return new ClassifiedProviderError(
      `OpenCode rate limit error${exitCode !== undefined ? ` (exit ${exitCode})` : ''}`,
      { kind: 'rate_limit', cause: input.cause },
    );
  }

  // Quota
  if (lower.includes('quota exceeded') || lower.includes('insufficient') || lower.includes('capacity')) {
    return new ClassifiedProviderError(
      `OpenCode quota exhausted${exitCode !== undefined ? ` (exit ${exitCode})` : ''}`,
      { kind: 'quota_exhausted', cause: input.cause },
    );
  }

  // Context overflow
  if (lower.includes('prompt is too long') || lower.includes('context window') || lower.includes('max context')) {
    return new ClassifiedProviderError(
      `OpenCode context overflow${exitCode !== undefined ? ` (exit ${exitCode})` : ''}`,
      { kind: 'unrecoverable', cause: input.cause },
    );
  }

  // Executable not found
  if (lower.includes('enoent') || lower.includes('not found') || lower.includes('command not found')) {
    return new ClassifiedProviderError(
      `OpenCode executable not found. Ensure 'opencode' is installed and in PATH.`,
      { kind: 'unrecoverable', cause: input.cause },
    );
  }

  // Non-zero exit — treat as transient for retry
  if (exitCode !== undefined && exitCode !== null && exitCode !== 0) {
    return new ClassifiedProviderError(
      `OpenCode process exited with code ${exitCode}`,
      { kind: 'transient', cause: input.cause },
    );
  }

  // Default: transient
  return new ClassifiedProviderError(
    `OpenCode error: ${input.cause instanceof Error ? input.cause.message : String(input.cause)}`,
    { kind: 'transient', cause: input.cause },
  );
}

export class OpenCodeProvider {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    const { model, maxTokens, skipPermissions } = this.getOpenCodeConfig();

    if (!session.memorySessionId) {
      const syntheticMemorySessionId = `opencode-${session.contentSessionId}-${Date.now()}`;
      session.memorySessionId = syntheticMemorySessionId;
      this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
      logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=OpenCode`);
    }

    const mode = ModeManager.getInstance().getActiveMode();

    const initPrompt = session.lastPromptNumber === 1
      ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
      : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

    session.conversationHistory.push({ role: 'user', content: initPrompt });

    try {
      const initResponse = await this.queryOpenCode(initPrompt, model, skipPermissions, maxTokens);
      await this.handleInitResponse(initResponse, session, worker, model);
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'OpenCode init failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'OpenCode init failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      await this.handleSessionError(error, session, worker);
      return;
    }

    let lastCwd: string | undefined;

    try {
      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        lastCwd = await this.processOneMessage(session, message, lastCwd, model, worker, mode, skipPermissions, maxTokens);
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        logger.error('SDK', 'OpenCode message processing failed', { sessionId: session.sessionDbId, model }, error);
      } else {
        logger.error('SDK', 'OpenCode message processing failed with non-Error', { sessionId: session.sessionDbId, model }, new Error(String(error)));
      }
      await this.handleSessionError(error, session, worker);
      return;
    }

    const sessionDuration = Date.now() - session.startTime;
    logger.success('SDK', 'OpenCode agent completed', {
      sessionId: session.sessionDbId,
      duration: `${(sessionDuration / 1000).toFixed(1)}s`,
      historyLength: session.conversationHistory.length,
      model
    });
  }

  private prepareMessageMetadata(session: ActiveSession, message: { agentId?: string | null; agentType?: string | null }): void {
    session.pendingAgentId = message.agentId ?? null;
    session.pendingAgentType = message.agentType ?? null;
  }

  private async handleInitResponse(
    initResponse: { content: string; tokensUsed?: number },
    session: ActiveSession,
    worker: WorkerRef | undefined,
    model: string
  ): Promise<void> {
    if (initResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: initResponse.content });
      const tokensUsed = initResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

      await processAgentResponse(
        initResponse.content, session, this.dbManager, this.sessionManager,
        worker, tokensUsed, null, 'OpenCode', undefined, model
      );
    } else {
      logger.error('SDK', 'Empty OpenCode init response - session may lack context', {
        sessionId: session.sessionDbId, model
      });
    }
  }

  private async processOneMessage(
    session: ActiveSession,
    message: { _persistentId: number; agentId?: string | null; agentType?: string | null; type: 'observation' | 'summarize'; cwd?: string; prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; last_assistant_message?: string },
    lastCwd: string | undefined,
    model: string,
    worker: WorkerRef | undefined,
    mode: ModeConfig,
    skipPermissions: boolean,
    maxTokens: number
  ): Promise<string | undefined> {
    this.prepareMessageMetadata(session, message);

    if (message.cwd) {
      lastCwd = message.cwd;
    }
    const originalTimestamp = session.earliestPendingTimestamp;

    if (message.type === 'observation') {
      await this.processObservationMessage(
        session, message, originalTimestamp, lastCwd, model, worker, mode, skipPermissions, maxTokens
      );
    } else if (message.type === 'summarize') {
      await this.processSummaryMessage(
        session, message, originalTimestamp, lastCwd, model, worker, mode, skipPermissions, maxTokens
      );
    }

    return lastCwd;
  }

  private async processObservationMessage(
    session: ActiveSession,
    message: { prompt_number?: number; tool_name?: string; tool_input?: unknown; tool_response?: unknown; cwd?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    model: string,
    worker: WorkerRef | undefined,
    _mode: ModeConfig,
    skipPermissions: boolean,
    maxTokens: number
  ): Promise<void> {
    if (message.prompt_number !== undefined) {
      session.lastPromptNumber = message.prompt_number;
    }

    if (!session.memorySessionId) {
      throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
    }

    const obsPrompt = buildObservationPrompt({
      id: 0,
      tool_name: message.tool_name!,
      tool_input: JSON.stringify(message.tool_input),
      tool_output: JSON.stringify(message.tool_response),
      created_at_epoch: originalTimestamp ?? Date.now(),
      cwd: message.cwd
    });

    session.conversationHistory.push({ role: 'user', content: obsPrompt });
    const obsResponse = await this.queryOpenCode(obsPrompt, model, skipPermissions, maxTokens);

    let tokensUsed = 0;
    if (obsResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: obsResponse.content });
      tokensUsed = obsResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    await processAgentResponse(
      obsResponse.content || '', session, this.dbManager, this.sessionManager,
      worker, tokensUsed, originalTimestamp, 'OpenCode', lastCwd, model
    );
  }

  private async processSummaryMessage(
    session: ActiveSession,
    message: { last_assistant_message?: string },
    originalTimestamp: number | null,
    lastCwd: string | undefined,
    model: string,
    worker: WorkerRef | undefined,
    mode: ModeConfig,
    skipPermissions: boolean,
    maxTokens: number
  ): Promise<void> {
    if (!session.memorySessionId) {
      throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
    }

    const summaryPrompt = buildSummaryPrompt({
      id: session.sessionDbId,
      memory_session_id: session.memorySessionId,
      project: session.project,
      user_prompt: session.userPrompt,
      last_assistant_message: message.last_assistant_message || ''
    }, mode);

    session.conversationHistory.push({ role: 'user', content: summaryPrompt });
    const summaryResponse = await this.queryOpenCode(summaryPrompt, model, skipPermissions, maxTokens);

    let tokensUsed = 0;
    if (summaryResponse.content) {
      session.conversationHistory.push({ role: 'assistant', content: summaryResponse.content });
      tokensUsed = summaryResponse.tokensUsed || 0;
      session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
      session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
    }

    await processAgentResponse(
      summaryResponse.content || '', session, this.dbManager, this.sessionManager,
      worker, tokensUsed, originalTimestamp, 'OpenCode', lastCwd, model
    );
  }

  private async handleSessionError(error: unknown, session: ActiveSession, _worker?: WorkerRef): Promise<never> {
    if (isAbortError(error)) {
      logger.warn('SDK', 'OpenCode agent aborted', { sessionId: session.sessionDbId });
      throw error;
    }

    logger.failure('SDK', 'OpenCode agent error', { sessionDbId: session.sessionDbId }, error instanceof Error ? error : new Error(String(error)));
    throw error;
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  /**
   * Query the OpenCode CLI with a prompt and return the response.
   *
   * Uses `opencode run --format json` with the prompt sent via stdin.
   * The --format json flag outputs JSON events, from which we extract the
   * assistant's text content.
   */
  private async queryOpenCode(
    prompt: string,
    model: string,
    skipPermissions: boolean,
    maxTokens: number
  ): Promise<{ content: string; tokensUsed?: number }> {
    const estimatedTokens = this.estimateTokens(prompt);
    logger.debug('SDK', `Querying OpenCode (${model || 'default'})`, {
      promptLength: prompt.length,
      estimatedTokens,
      maxTokens,
    });

    // Warn if prompt exceeds configured token limit — the opencode CLI
    // will handle its own context but this helps identify runaway prompts.
    if (estimatedTokens > maxTokens) {
      logger.warn('SDK', `Prompt tokens (${estimatedTokens}) exceed CLAUDE_MEM_OPENCODE_MAX_TOKENS (${maxTokens})`, {
        promptLength: prompt.length,
      });
    }

    const result = await withRetry<{ content: string; tokensUsed?: number }>(async (attemptSignal) => {
      return new Promise<{ content: string; tokensUsed?: number }>((resolve, reject) => {
        const args: string[] = ['run', '--format', 'json'];
        if (model) {
          args.push('-m', model);
        }

        if (skipPermissions) {
          args.push('--dangerously-skip-permissions');
        }

        const child = spawnHidden('opencode', args, {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env },
        });

        // Stdio is explicitly 'pipe' for all three streams above, so the
        // narrowed `non-null` assertions hold; spawnHidden's loose return
        // type doesn't carry that constraint.
        const childStdin = child.stdin!;
        const childStdout = child.stdout!;
        const childStderr = child.stderr!;

        // Send prompt via stdin to avoid ARG_MAX issues
        childStdin.on('error', (err) => {
          // EPIPE or similar — child process may have exited before
          // consuming stdin (auth failure, bad args, etc.). Stdin errors
          // are expected in error paths; the close handler will reject.
          logger.debug('SDK', 'OpenCode stdin error (handled)', {
            error: err.message,
            code: (err as any).code,
          });
        });
        childStdin.write(prompt, (writeErr) => {
          if (writeErr && (writeErr as any).code === 'EPIPE') {
            logger.debug('SDK', 'OpenCode stdin write EPIPE (handled)', {
              sessionId: undefined,
            });
          }
        });
        childStdin.end();

        let stdout = '';
        let stderr = '';

        childStdout.on('data', (data: Buffer) => {
          stdout += data.toString();
        });

        childStderr.on('data', (data: Buffer) => {
          stderr += data.toString();
        });

        // Handle abort signal
        const onAbort = () => {
          child.kill('SIGTERM');
          const abortErr = new Error('OpenCode query aborted');
          (abortErr as any).name = 'AbortError';
          reject(abortErr);
        };
        attemptSignal.addEventListener('abort', onAbort);

        child.on('close', (code, signal) => {
          attemptSignal.removeEventListener('abort', onAbort);

          if (code !== 0 && code !== null) {
            const classified = classifyOpenCodeError({
              exitCode: code,
              stderr,
              cause: new Error(`OpenCode process exited with code ${code}: ${stderr.substring(0, 500)}`),
            });
            reject(classified);
            return;
          }

          if (signal) {
            reject(classifyOpenCodeError({
              exitCode: code,
              stderr,
              cause: new Error(`OpenCode process killed by signal ${signal}`),
            }));
            return;
          }

          try {
            const parsed = this.parseOpenCodeJsonOutput(stdout);
            resolve(parsed);
          } catch (parseError) {
            logger.error('SDK', 'Failed to parse OpenCode JSON output', {
              stdoutLength: stdout.length,
              stderrLength: stderr.length,
            }, parseError instanceof Error ? parseError : new Error(String(parseError)));
            reject(classifyOpenCodeError({
              exitCode: code,
              stderr,
              cause: new Error(`Failed to parse OpenCode output: ${parseError instanceof Error ? parseError.message : String(parseError)}`),
            }));
          }
        });

        child.on('error', (err) => {
          attemptSignal.removeEventListener('abort', onAbort);
          reject(classifyOpenCodeError({ cause: err }));
        });
      });
    }, { label: `OpenCode ${model || 'default'}` });

    return result;
  }

  /**
   * Parse the JSON event stream output from `opencode run --format json`.
   *
   * The output is a stream of JSON objects (one per line / JSONL).
   * Known event types from opencode:
   *   - { type: "text", part: { text: "...", tokens: { input, output, ... } } }
   *   - { type: "step_start", part: { ... } }
   *   - { type: "step_finish", part: { tokens: { total, input, output, ... }, cost: ... } }
   *   - { type: "error", error: { ... } }
   *
   * We also handle Claude-style events as a fallback:
   *   - { type: "assistant", message: { content: [...] } }
   *   - { type: "result", result: "..." }
   */
  private parseOpenCodeJsonOutput(stdout: string): { content: string; tokensUsed?: number } {
    if (!stdout || !stdout.trim()) {
      logger.warn('SDK', 'Empty stdout from OpenCode');
      return { content: '' };
    }

    const lines = stdout.trim().split('\n').filter(line => line.trim());
    let content = '';
    let inputTokens = 0;
    let outputTokens = 0;
    let hasStepFinishTokens = false;
    // Track whether the native opencode shape already contributed content
    // (text / assistant / message events). Some adapters ALSO emit a final
    // `result` event with the same payload as a tail summary — appending it
    // would double the output. The result fallback only fires when nothing
    // earlier in the stream produced content.
    let hasNativeContent = false;

    for (const line of lines) {
      try {
        const event = JSON.parse(line);

        // OpenCode native JSON format: { type: "text", part: { text: "..." } }
        if (event.type === 'text' && event.part?.text) {
          content += event.part.text;
          hasNativeContent = true;
          // Extract tokens from text event part (only if no step_finish seen yet)
          if (event.part.tokens && !hasStepFinishTokens) {
            inputTokens += event.part.tokens.input || event.part.tokens.prompt_tokens || 0;
            outputTokens += event.part.tokens.output || event.part.tokens.completion_tokens || 0;
          }
        }

        // OpenCode step_finish: { type: "step_finish", part: { tokens: { total, input, output, ... } } }
        if (event.type === 'step_finish' && event.part?.tokens) {
          hasStepFinishTokens = true;
          inputTokens = event.part.tokens.input || event.part.tokens.prompt_tokens || 0;
          outputTokens = event.part.tokens.output || event.part.tokens.completion_tokens || 0;
        }

        // OpenCode error event
        if (event.type === 'error' && event.error) {
          const errorName = event.error.name || 'UnknownError';
          const errorMessage = event.error.data?.message || event.error.message || 'Unknown error';
          throw new Error(`OpenCode error event: ${errorName} - ${errorMessage}`);
        }

        // Fallback: Claude-style assistant messages.
        // Pick the FIRST non-empty source so we don't double-count when the
        // upstream emits both `event.message.content` and `event.content`
        // for the same payload (which some adapter shapes do).
        if (event.type === 'assistant' || event.type === 'message') {
          let extracted = '';
          if (event.message?.content) {
            if (typeof event.message.content === 'string') {
              extracted = event.message.content;
            } else if (Array.isArray(event.message.content)) {
              for (const block of event.message.content) {
                if (block.type === 'text' && block.text) {
                  extracted += block.text;
                }
              }
            }
          }
          if (!extracted && event.content && typeof event.content === 'string') {
            extracted = event.content;
          }
          if (!extracted && event.text && typeof event.text === 'string') {
            extracted = event.text;
          }
          if (extracted) {
            content += extracted;
            hasNativeContent = true;
          }
        }

        // Fallback: `result` event content. Only contributes when nothing
        // earlier in the stream produced content — adapters that emit a
        // tail-summary `result` after `text` events would otherwise double
        // the output.
        if (event.type === 'result' && !hasNativeContent) {
          if (event.result && typeof event.result === 'string') {
            content += event.result;
          } else if (event.content && typeof event.content === 'string') {
            content += event.content;
          }
        }

        // Fallback: generic token usage (only if no step_finish seen yet)
        if (event.usage && !hasStepFinishTokens) {
          inputTokens += event.usage.input_tokens || event.usage.prompt_tokens || 0;
          outputTokens += event.usage.output_tokens || event.usage.completion_tokens || 0;
        }
        if (event.message?.usage && !hasStepFinishTokens) {
          inputTokens += event.message.usage.input_tokens || event.message.usage.prompt_tokens || 0;
          outputTokens += event.message.usage.output_tokens || event.message.usage.completion_tokens || 0;
        }
      } catch (parseErr) {
        // Re-throw intentional errors (e.g. from error event handling above)
        if (!(parseErr instanceof SyntaxError)) {
          throw parseErr;
        }
        // Not a JSON line, might be plain text output
        // Try to use it as content if we haven't found any JSON content
        if (!content && line.trim()) {
          content = line.trim();
        }
      }
    }

    const tokensUsed = inputTokens + outputTokens > 0 ? inputTokens + outputTokens : undefined;

    if (tokensUsed) {
      logger.info('SDK', 'OpenCode usage', {
        inputTokens,
        outputTokens,
        totalTokens: tokensUsed,
      });
    }

    return { content: content.trim(), tokensUsed };
  }

  private getOpenCodeConfig(): { model: string; maxTokens: number; skipPermissions: boolean } {
    const settingsPath = USER_SETTINGS_PATH;
    const settings = SettingsDefaultsManager.loadFromFile(settingsPath);

    // Empty string means use opencode's default model
    const model = settings.CLAUDE_MEM_OPENCODE_MODEL || '';
    const maxTokens = parseInt(settings.CLAUDE_MEM_OPENCODE_MAX_TOKENS, 10) || DEFAULT_MAX_ESTIMATED_TOKENS;
    // Coerce to string so a manually-edited settings.json with a real
    // boolean value (`"...": true`) and the typed string default ("true")
    // both resolve identically without tripping strict-mode comparisons.
    const skipPermissions = String(settings.CLAUDE_MEM_OPENCODE_SKIP_PERMISSIONS) === 'true';

    return { model, maxTokens, skipPermissions };
  }
}

let _openCodeAvailableCache: boolean | null = null;
let _missingBinaryWarned = false;

/**
 * Reset the cached availability check. Tests use this; production code
 * shouldn't need it (the cache is intentionally process-lifetime).
 */
export function resetOpenCodeAvailableCache(): void {
  _openCodeAvailableCache = null;
  _missingBinaryWarned = false;
}

export function isOpenCodeAvailable(): boolean {
  if (_openCodeAvailableCache !== null) {
    return _openCodeAvailableCache;
  }
  try {
    // execFileSync (no shell) — args are static so injection isn't a real
    // risk, but using execFile keeps the call consistent with project-wide
    // PATH-lookup conventions and avoids `cmd /c` flashing a console window
    // on Windows. windowsHide on stdio is the explicit hide.
    const lookupCmd = process.platform === 'win32' ? 'where' : 'which';
    execFileSync(lookupCmd, ['opencode'], {
      stdio: ['ignore', 'ignore', 'ignore'],
      windowsHide: true,
      timeout: 5000,
    });
    _openCodeAvailableCache = true;
    return true;
  } catch {
    _openCodeAvailableCache = false;
    // One-shot warning when the user selected opencode but the binary isn't
    // on PATH — otherwise the worker silently falls back to another provider
    // and the user wonders why their selection isn't taking effect.
    if (!_missingBinaryWarned && isOpenCodeSelected()) {
      _missingBinaryWarned = true;
      logger.warn(
        'SDK',
        "CLAUDE_MEM_PROVIDER=opencode but the 'opencode' binary is not on PATH. " +
        'Falling back to other providers. Install opencode (https://github.com/sst/opencode) and restart the worker, ' +
        'or change CLAUDE_MEM_PROVIDER in ~/.claude-mem/settings.json.',
      );
    }
    return false;
  }
}

export function isOpenCodeSelected(): boolean {
  const settingsPath = USER_SETTINGS_PATH;
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return settings.CLAUDE_MEM_PROVIDER === 'opencode';
}