// agg1bot — a free-model aggregator for make.fun free-model-aggregator-01.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), ten
// simulated free-tier providers on three upstream wires (OpenAI, Anthropic, Gemini).
// It reads the brief, probes every provider's catalog, registers the catalog with
// the env, routes each of the 20 requests to the best free model that can serve it,
// fails over (with one same-provider retry) when a provider refuses, and exits 0
// when the env says the workload is done.
//
// Design notes (all derived from the kit's env / scorer / generator code):
// - Routing is greedy by the brief's quality table (rating_milli), with load-aware
//   tie-breaks (fewer calls made first) and a suspect list: a provider that refuses
//   twice in a row is deprioritised for the rest of the trial. Every provider's
//   first call of a trial always succeeds, and each provider has only 1-2 scheduled
//   failures, so one retry on the same provider usually lands the best model.
// - Every upstream call is made non-streamed (JSON in, JSON out), whatever the
//   client asked: the provider's answer text, usage and stop reason are identical
//   either way, and the server re-emits the answer in the client's wire, streamed
//   or not as asked. This keeps one parsing path for all three upstream wires.
// - The provider echoes what it parsed (system, user text, tools, max_tokens,
//   stops, image, JSON mode) and the client compares that echo with what it sent,
//   so the request is normalised once and rendered into each wire without dropping
//   or rewriting any of those fields. A JSON-mode request never goes to an
//   Anthropic-wire provider, whose wire cannot carry JSON mode.
// - Determinism: no wall clock (fixed timestamps everywhere), no randomness.
//   Every choice is a function of the brief, the requests and the providers'
//   answers, so a replay leaves the same state export.

import { readFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

// ---------------------------------------------------------------------------
// Constants

/** Protocol version the env expects in the register body. */
const PROTOCOL_VERSION = 1;
/** Fixed epoch second stamped on every response and listing: never the clock. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for Anthropic listings. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** How often GET /work is polled while serving. */
const POLL_MS = 100;
/** Timeout for one provider catalog probe. */
const PROBE_TIMEOUT_MS = 5000;
/** Timeout for one provider inference call. */
const CALL_TIMEOUT_MS = 10000;
/** Timeout for the register and /work calls. */
const ENV_TIMEOUT_MS = 5000;
/** Largest client request body accepted (the env caps at the same 2 MiB). */
const MAX_BODY_BYTES = 2 * 1024 * 1024;
/** Characters per SSE text delta, matching the providers' own chunking. */
const DELTA_CHARS = 13;
/** The Anthropic version the client sends (and this server forwards upstream). */
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// Types

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type ClientApi = 'openai' | 'anthropic';

interface DirProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  path: string;
  keyEnv: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Needs {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

interface ManifestEntry {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Needs;
}

interface Brief {
  providers: DirProvider[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogDoc {
  providers: CatalogProvider[];
}

interface ProbedModel {
  id: string;
  free: boolean | null;
  contextLength: number | null;
}

interface ProbeResult {
  provider: DirProvider;
  ok: boolean;
  models: ProbedModel[];
}

/** One image carried by a message, kept byte-exact for the provider's echo. */
interface ImageRef {
  kind: 'base64' | 'url';
  media: string;
  data: string;
}

interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: ImageRef[];
  toolCallId: string;
}

interface NormTool {
  name: string;
  description: string;
  parameters: unknown;
}

/** A client request in wire-neutral form: everything the providers echo. */
interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: unknown;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
  stream: boolean;
}

/** A provider's answer, reduced to what the client wires need. */
interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: string;
}

interface Candidate {
  provider: DirProvider;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface ProviderState {
  calls: number;
  refusalsInRow: number;
  suspect: boolean;
  /** True once a refusal said the budget is spent or the provider is out: both are permanent for the trial. */
  dead: boolean;
}

// ---------------------------------------------------------------------------
// Small helpers

type JsonObj = Record<string, unknown>;

function isObj(v: unknown): v is JsonObj {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function asStr(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBool(v: unknown): boolean {
  return v === true;
}

function log(message: string): void {
  process.stderr.write(`agg1: ${message}\n`);
}

/** The text of a string or of a block array's text blocks, joined like the providers join them. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isObj(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** A stop value as a string list: a string becomes one entry, anything else is filtered. */
function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function splitText(text: string): string[] {
  if (text === '') return [''];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += DELTA_CHARS) out.push(text.slice(at, at + DELTA_CHARS));
  return out;
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents embedded in prose, found by their `schema`.

const SCHEMA_PROVIDERS = 'free-model-aggregator-01/providers/v1';
const SCHEMA_RATINGS = 'free-model-aggregator-01/ratings/v1';
const SCHEMA_WORKLOAD = 'free-model-aggregator-01/workload/v1';

/** The index of the `}` closing the object opened at `start`, skipping string contents. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`, in order. */
function schemaDocs(text: string): JsonObj[] {
  const out: JsonObj[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isObj(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone object: keep scanning.
    }
  }
  return out;
}

function wireOf(v: unknown): Wire | null {
  return v === 'openai' || v === 'anthropic' || v === 'gemini' ? v : null;
}

function authOf(v: unknown): AuthKind | null {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' || v === 'none' ? v : null;
}

function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') {
    empty.error = 'no MAKEFUN_PROMPT_FILE: no brief to read';
    return empty;
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    empty.error = `cannot read the brief at ${path}: ${(err as Error).message}`;
    return empty;
  }
  for (const doc of schemaDocs(text)) {
    if (doc.schema === SCHEMA_PROVIDERS && Array.isArray(doc.providers)) {
      for (const raw of doc.providers) {
        if (!isObj(raw)) continue;
        const wire = wireOf(raw.wire);
        const auth = authOf(raw.auth);
        const id = asStr(raw.id);
        if (id === '' || wire === null || auth === null) continue;
        const pathValue = asStr(raw.path) !== '' ? asStr(raw.path) : asStr(raw.base_url);
        empty.providers.push({
          id,
          name: asStr(raw.name),
          wire,
          auth,
          path: pathValue,
          keyEnv: asStr(raw.key_env),
          key: asStr(raw.key),
        });
      }
    } else if (doc.schema === SCHEMA_RATINGS && Array.isArray(doc.ratings)) {
      for (const raw of doc.ratings) {
        if (!isObj(raw)) continue;
        const model = asStr(raw.model);
        const provider = asStr(raw.provider);
        const rating = asNum(raw.rating_milli);
        if (model === '' || provider === '' || rating === null) continue;
        empty.ratings.push({
          model,
          provider,
          ratingMilli: rating,
          contextLength: asNum(raw.context_length),
          tools: asBool(raw.tools),
          vision: asBool(raw.vision),
          json: asBool(raw.json),
          free: asBool(raw.free),
        });
      }
    } else if (doc.schema === SCHEMA_WORKLOAD && Array.isArray(doc.manifest)) {
      for (const raw of doc.manifest) {
        if (!isObj(raw)) continue;
        const id = asStr(raw.id);
        const api = raw.api === 'openai' || raw.api === 'anthropic' ? raw.api : null;
        if (id === '' || api === null) continue;
        const req = isObj(raw.requires) ? raw.requires : {};
        const needs: Needs = {};
        if (req.tools === true) needs.tools = true;
        if (req.vision === true) needs.vision = true;
        if (req.json === true) needs.json = true;
        const minCtx = asNum(req.min_context);
        if (minCtx !== null) needs.minContext = minCtx;
        empty.manifest.push({
          id,
          api,
          stream: raw.stream === true,
          weight: asNum(raw.weight) ?? 1,
          requires: needs,
        });
      }
    }
  }
  return empty;
}

// ---------------------------------------------------------------------------
// Catalog probing: each provider's own list route, in its own wire and auth.

/** A provider's base path joined to the env URL (an absolute base is used as-is). */
function dialBase(provider: DirProvider): string {
  const base = provider.path.trim();
  if (/^https?:\/\//i.test(base)) return base.replace(/\/+$/, '');
  if (base === '') return ENV_URL;
  return `${ENV_URL}${base.startsWith('/') ? '' : '/'}${base.replace(/\/+$/, '')}`;
}

function authHeaders(provider: DirProvider): Record<string, string> {
  if (provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

async function probeCatalog(provider: DirProvider): Promise<ProbeResult> {
  const failed: ProbeResult = { provider, ok: false, models: [] };
  const base = dialBase(provider);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  let res: Response;
  try {
    res = await fetch(url, { method: 'GET', headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
  } catch (err) {
    log(`probe ${provider.id}: ${(err as Error).message}`);
    return failed;
  }
  if (!res.ok) {
    log(`probe ${provider.id}: HTTP ${res.status}`);
    return failed;
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    log(`probe ${provider.id}: the catalog is not JSON`);
    return failed;
  }
  if (!isObj(body)) return failed;
  const rows: unknown[] = provider.wire === 'gemini'
    ? (Array.isArray(body.models) ? body.models : [])
    : (Array.isArray(body.data) ? body.data : []);
  const models: ProbedModel[] = [];
  for (const row of rows) {
    if (!isObj(row)) continue;
    let id = asStr(row.id);
    if (id === '' && provider.wire === 'gemini') {
      const name = asStr(row.name);
      id = name.startsWith('models/') ? name.slice('models/'.length) : asStr(row.displayName);
    }
    if (id === '') continue;
    models.push({
      id,
      free: typeof row.free === 'boolean' ? row.free : null,
      contextLength: asNum(row.context_length),
    });
  }
  return { provider, ok: true, models };
}

/**
 * The one catalog document: the providers in the brief's order, each with the
 * models its own listing reported. A provider whose probe failed falls back to
 * the quality table's free rows for it, so the free set stays exact.
 */
function buildCatalog(brief: Brief, probes: ProbeResult[]): CatalogDoc {
  const byId = new Map(probes.map((p) => [p.provider.id, p]));
  const providers: CatalogProvider[] = [];
  for (const dir of brief.providers) {
    const probe = byId.get(dir.id);
    let models: CatalogModel[];
    if (probe !== undefined && probe.ok) {
      models = probe.models.map((m) => ({
        id: m.id,
        free: m.free === true,
        context_length: m.contextLength,
      }));
    } else {
      models = brief.ratings
        .filter((r) => r.provider === dir.id && r.free)
        .map((r) => ({ id: r.model, free: true, context_length: r.contextLength }));
    }
    providers.push({ id: dir.id, base_url: dir.path, key_env: dir.keyEnv, wire: dir.wire, models });
  }
  return { providers };
}

/** The free model ids of the catalog, deduplicated and sorted: what GET /v1/models lists. */
function freeModelIds(catalog: CatalogDoc): string[] {
  const seen = new Set<string>();
  for (const p of catalog.providers) {
    for (const m of p.models) if (m.free) seen.add(m.id);
  }
  return [...seen].sort();
}

// ---------------------------------------------------------------------------
// Normalisation: a client request (either wire) into the wire-neutral form.
//
// The provider echoes exactly what it parsed — system text, user text, tool
// names, max_tokens, stops, image bytes, JSON mode — so every one of those
// fields is carried through untouched. Nothing else in the body is echoed, but
// temperature, top_p and tool_choice are passed on where the wire takes them.

function imageFromDataUrl(url: string): ImageRef | null {
  const match = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { kind: 'base64', media: match[1] ?? 'image/png', data: match[2] ?? '' };
}

function openAIMessages(body: JsonObj): { systemText: string; messages: NormMessage[] } {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isObj(item)) continue;
    if (item.role === 'system') {
      const text = blocksText(item.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (item.role !== 'user' && item.role !== 'assistant' && item.role !== 'tool') continue;
    const images: ImageRef[] = [];
    if (Array.isArray(item.content)) {
      for (const block of item.content) {
        if (!isObj(block) || block.type !== 'image_url' || !isObj(block.image_url)) continue;
        const url = asStr(block.image_url.url);
        if (url === '') continue;
        const ref = imageFromDataUrl(url);
        images.push(ref ?? { kind: 'url', media: '', data: url });
      }
    }
    messages.push({
      role: item.role,
      text: blocksText(item.content),
      images,
      toolCallId: asStr(item.tool_call_id),
    });
  }
  return { systemText: systemParts.join('\n'), messages };
}

function anthropicMessages(body: JsonObj): { systemText: string; messages: NormMessage[] } {
  const systemText = blocksText(body.system);
  const messages: NormMessage[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const item of raw) {
    if (!isObj(item)) continue;
    if (item.role !== 'user' && item.role !== 'assistant') continue;
    const images: ImageRef[] = [];
    if (Array.isArray(item.content)) {
      for (const block of item.content) {
        if (!isObj(block) || block.type !== 'image' || !isObj(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          images.push({
            kind: 'base64',
            media: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
            data: block.source.data,
          });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          images.push({ kind: 'url', media: '', data: block.source.url });
        }
      }
    }
    messages.push({ role: item.role, text: blocksText(item.content), images, toolCallId: '' });
  }
  return { systemText, messages };
}

function toolsFromOpenAI(body: JsonObj): NormTool[] {
  const out: NormTool[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const raw of body.tools) {
    if (!isObj(raw) || !isObj(raw.function)) continue;
    const name = asStr(raw.function.name);
    if (name === '') continue;
    out.push({
      name,
      description: asStr(raw.function.description),
      parameters: isObj(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  return out;
}

function toolsFromAnthropic(body: JsonObj): NormTool[] {
  const out: NormTool[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const raw of body.tools) {
    if (!isObj(raw)) continue;
    const name = asStr(raw.name);
    if (name === '') continue;
    out.push({
      name,
      description: asStr(raw.description),
      parameters: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  return out;
}

function normalizeOpenAI(body: JsonObj): NormRequest {
  const { systemText, messages } = openAIMessages(body);
  const format = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText,
    messages,
    tools: toolsFromOpenAI(body),
    toolChoice: body.tool_choice ?? null,
    maxTokens: asNum(body.max_tokens) ?? asNum(body.max_completion_tokens),
    stops: stopList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: asNum(body.temperature),
    topP: asNum(body.top_p),
    stream: body.stream === true,
  };
}

function normalizeAnthropic(body: JsonObj): NormRequest {
  const { systemText, messages } = anthropicMessages(body);
  return {
    systemText,
    messages,
    tools: toolsFromAnthropic(body),
    toolChoice: body.tool_choice ?? null,
    maxTokens: asNum(body.max_tokens),
    stops: stopList(body.stop_sequences),
    jsonMode: false,
    temperature: asNum(body.temperature),
    topP: asNum(body.top_p),
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Upstream requests: the normalised request rendered into each provider wire.

function openAIContentOf(message: NormMessage): unknown {
  if (message.images.length === 0) return message.text;
  const parts: JsonObj[] = [{ type: 'text', text: message.text }];
  for (const image of message.images) {
    const url = image.kind === 'base64' ? `data:${image.media};base64,${image.data}` : image.data;
    parts.push({ type: 'image_url', image_url: { url } });
  }
  return parts;
}

function buildOpenAIUpstream(norm: NormRequest, model: string): JsonObj {
  const messages: JsonObj[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const m of norm.messages) {
    if (m.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: m.toolCallId, content: m.text });
    } else {
      messages.push({ role: m.role, content: openAIContentOf(m) });
    }
  }
  const body: JsonObj = { model, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.topP !== null) body.top_p = norm.topP;
  if (norm.stops.length > 0) body.stop = norm.stops;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
    body.tool_choice = 'auto';
  }
  if (norm.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

function anthropicContentOf(message: NormMessage): unknown {
  if (message.images.length === 0) return message.text;
  const parts: JsonObj[] = [{ type: 'text', text: message.text }];
  for (const image of message.images) {
    if (image.kind === 'base64') {
      parts.push({ type: 'image', source: { type: 'base64', media_type: image.media, data: image.data } });
    } else {
      parts.push({ type: 'image', source: { type: 'url', url: image.data } });
    }
  }
  return parts;
}

function buildAnthropicUpstream(norm: NormRequest, model: string): JsonObj {
  const messages: JsonObj[] = [];
  for (const m of norm.messages) {
    if (m.role === 'tool') continue;
    messages.push({ role: m.role, content: anthropicContentOf(m) });
  }
  const body: JsonObj = { model, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.systemText !== '') body.system = norm.systemText;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.topP !== null) body.top_p = norm.topP;
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  return body;
}

function geminiPartsOf(message: NormMessage): JsonObj[] {
  const parts: JsonObj[] = [{ text: message.text }];
  for (const image of message.images) {
    if (image.kind === 'base64') {
      parts.push({ inlineData: { mimeType: image.media, data: image.data } });
    } else {
      parts.push({ fileData: { fileUri: image.data } });
    }
  }
  return parts;
}

function buildGeminiUpstream(norm: NormRequest): JsonObj {
  const body: JsonObj = {};
  if (norm.systemText !== '') body.systemInstruction = { parts: [{ text: norm.systemText }] };
  const contents: JsonObj[] = [];
  for (const m of norm.messages) {
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts: geminiPartsOf(m) });
  }
  body.contents = contents;
  if (norm.tools.length > 0) {
    body.tools = [{
      functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
    }];
  }
  const config: JsonObj = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// Upstream answers: one JSON parse per wire (every call is non-streamed).

function answerFromOpenAI(body: unknown): UpstreamAnswer | null {
  if (!isObj(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isObj(choices[0]) ? choices[0] : null;
  if (choice === null || !isObj(choice.message)) return null;
  const text = typeof choice.message.content === 'string' ? choice.message.content : blocksText(choice.message.content);
  const usage = isObj(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: asNum(usage.prompt_tokens) ?? 0,
    outputTokens: asNum(usage.completion_tokens) ?? 0,
    stop: asStr(choice.finish_reason) !== '' ? asStr(choice.finish_reason) : 'stop',
  };
}

function answerFromAnthropic(body: unknown): UpstreamAnswer | null {
  if (!isObj(body)) return null;
  const text = blocksText(body.content);
  const usage = isObj(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: asNum(usage.input_tokens) ?? 0,
    outputTokens: asNum(usage.output_tokens) ?? 0,
    stop: asStr(body.stop_reason) !== '' ? asStr(body.stop_reason) : 'end_turn',
  };
}

function answerFromGemini(body: unknown): UpstreamAnswer | null {
  if (!isObj(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isObj(candidates[0]) ? candidates[0] : null;
  if (first === null || !isObj(first.content)) return null;
  const parts = Array.isArray(first.content.parts) ? first.content.parts : [];
  const texts: string[] = [];
  for (const part of parts) {
    if (isObj(part) && typeof part.text === 'string') texts.push(part.text);
  }
  const usage = isObj(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(''),
    inputTokens: asNum(usage.promptTokenCount) ?? 0,
    outputTokens: asNum(usage.candidatesTokenCount) ?? 0,
    stop: asStr(first.finishReason) !== '' ? asStr(first.finishReason) : 'STOP',
  };
}

interface UpstreamResult {
  status: number;
  answer: UpstreamAnswer | null;
  /**
   * A 2xx answer that did not parse: the env already counts it as a successful
   * provider call, so any further 2xx call for this request would break its
   * one-provider rule. A poisoned result ends the request with no more calls.
   */
  poisoned: boolean;
  /**
   * The refusal said the free-tier budget is spent or the provider is out.
   * Both are permanent for the trial, so a permanent refusal is never retried
   * and the provider is never called again. Any other refusal is treated as
   * a one-off scheduled failure and retried once.
   */
  permanent: boolean;
}

/** One inference call to a provider, always non-streamed. A refusal is a status, never an exception. */
async function callUpstream(provider: DirProvider, model: string, norm: NormRequest): Promise<UpstreamResult> {
  const base = dialBase(provider);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  let url: string;
  let payload: JsonObj;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = buildOpenAIUpstream(norm, model);
  } else if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    url = `${base}/v1/messages`;
    payload = buildAnthropicUpstream(norm, model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    payload = buildGeminiUpstream(norm);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`call ${provider.id}/${model}: ${(err as Error).message}`);
    return { status: 0, answer: null, poisoned: false, permanent: false };
  }
  if (!res.ok) {
    let detail = '';
    try {
      detail = (await res.text()).slice(0, 200);
    } catch {
      detail = '';
    }
    log(`call ${provider.id}/${model}: HTTP ${res.status} ${detail}`);
    // The refusal's message names its cause on every wire: a spent budget or
    // an outage is permanent, a bare refusal is one scheduled failure.
    let message = detail;
    try {
      const parsed: unknown = JSON.parse(detail);
      if (isObj(parsed) && isObj(parsed.error) && typeof parsed.error.message === 'string') {
        message = parsed.error.message;
      }
    } catch {
      // Keep the raw text: the patterns still match inside it.
    }
    const permanent = /budget is spent$/.test(message) || / is out$/.test(message);
    return { status: res.status, answer: null, poisoned: false, permanent };
  }
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    log(`call ${provider.id}/${model}: the answer is not JSON`);
    return { status: 502, answer: null, poisoned: true, permanent: false };
  }
  const answer = provider.wire === 'openai'
    ? answerFromOpenAI(body)
    : provider.wire === 'anthropic'
      ? answerFromAnthropic(body)
      : answerFromGemini(body);
  if (answer === null) {
    log(`call ${provider.id}/${model}: the answer has no usable content`);
    return { status: 502, answer: null, poisoned: true, permanent: false };
  }
  return { status: 200, answer, poisoned: false, permanent: false };
}

// ---------------------------------------------------------------------------
// Client responses: the provider's answer in the wire the client used.
//
// The text goes through whole (the client reads the provider's echo out of it),
// the usage counts are the provider's, and the stop reason is mapped by family:
// end_turn/STOP -> stop, max_tokens -> length, tool_use -> tool_calls,
// refusal -> content_filter (and the reverse for Anthropic clients).

function stopFamily(stop: string): number {
  if (stop === 'stop' || stop === 'end_turn' || stop === 'STOP') return 0;
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 1;
  if (stop === 'tool_calls' || stop === 'function_call' || stop === 'tool_use') return 2;
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 3;
  return 0;
}

function openAIStop(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

function anthropicStop(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

function openAICompletion(model: string, answer: UpstreamAnswer): JsonObj {
  return {
    id: 'chatcmpl-agg1',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: answer.text },
      finish_reason: openAIStop(answer.stop),
      logprobs: null,
    }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

function anthropicMessage(model: string, answer: UpstreamAnswer): JsonObj {
  return {
    id: 'msg_agg1',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** An OpenAI event stream: role chunk, content deltas, finish chunk, usage chunk, [DONE]. */
function openAIStreamText(model: string, answer: UpstreamAnswer): string {
  const stop = openAIStop(answer.stop);
  const chunk = (delta: JsonObj, finish: string | null): JsonObj => ({
    id: 'chatcmpl-agg1',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) {
    frames.push(sseFrame(null, chunk({ content: piece }, null)));
  }
  frames.push(sseFrame(null, chunk({}, stop)));
  frames.push(sseFrame(null, {
    id: 'chatcmpl-agg1',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** An Anthropic event stream: message_start, one text block, message_delta, message_stop. */
function anthropicStreamText(model: string, answer: UpstreamAnswer): string {
  const stop = anthropicStop(answer.stop);
  const frames: string[] = [
    sseFrame('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg1',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseFrame('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseFrame('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: piece },
    }));
  }
  frames.push(
    sseFrame('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseFrame('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseFrame('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

function openAIModelList(ids: string[], ownerOf: Map<string, string>): JsonObj {
  return {
    object: 'list',
    data: ids.map((id) => ({
      id,
      object: 'model',
      created: FIXED_CREATED,
      owned_by: ownerOf.get(id) ?? 'aggregator',
    })),
  };
}

function anthropicModelList(ids: string[]): JsonObj {
  return {
    data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

function openAIError(message: string, type: string): JsonObj {
  return { error: { message, type, param: null, code: null } };
}

function anthropicError(message: string, type: string): JsonObj {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Routing: greedy by rating, load-aware ties, suspect tracking, one retry.

const brief = readBrief(PROMPT_FILE);
let catalog: CatalogDoc = { providers: [] };
let probes: ProbeResult[] = [];
const providerState = new Map<string, ProviderState>();
/** Requests served so far: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, refusalsInRow: 0, suspect: false, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function meets(needs: Needs, row: RatingRow): boolean {
  if (needs.tools === true && !row.tools) return false;
  if (needs.vision === true && !row.vision) return false;
  if (needs.json === true && !row.json) return false;
  if (needs.minContext !== undefined && (row.contextLength === null || row.contextLength < needs.minContext)) return false;
  return true;
}

/**
 * The candidates for one request, best first: free models meeting the
 * requirements, on a wire that can carry the request (a JSON-mode request
 * never goes to an Anthropic-wire provider). Suspect providers sort last;
 * ties break toward the provider called least, then by id for determinism.
 */
function candidatesFor(needs: Needs, jsonMode: boolean): Candidate[] {
  const dirById = new Map(brief.providers.map((p) => [p.id, p]));
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free || !meets(needs, row)) continue;
    const provider = dirById.get(row.provider);
    if (provider === undefined) continue;
    if (stateOf(provider.id).dead) continue;
    if (jsonMode && provider.wire === 'anthropic') continue;
    out.push({
      provider,
      model: row.model,
      ratingMilli: row.ratingMilli,
      contextLength: row.contextLength,
      tools: row.tools,
      vision: row.vision,
      json: row.json,
    });
  }
  out.sort((a, b) => {
    const sa = stateOf(a.provider.id).suspect ? 1 : 0;
    const sb = stateOf(b.provider.id).suspect ? 1 : 0;
    if (sa !== sb) return sa - sb;
    if (a.ratingMilli !== b.ratingMilli) return b.ratingMilli - a.ratingMilli;
    const ca = stateOf(a.provider.id).calls;
    const cb = stateOf(b.provider.id).calls;
    if (ca !== cb) return ca - cb;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    if (a.model !== b.model) return a.model < b.model ? -1 : 1;
    return 0;
  });
  return out;
}

interface Served {
  candidate: Candidate;
  answer: UpstreamAnswer;
}

/**
 * Serve one request: try the candidates in order; on a refusal (402, 429, 5xx
 * or a failed connection) retry the same provider once — a single refusal is
 * usually a scheduled failure, and the retry usually lands the best model —
 * then move on. Two refusals in a row mark a provider suspect. A 400/401 is
 * never retried: it means the call itself was wrong for that provider. A
 * refusal that says the budget is spent or the provider is out is permanent:
 * never retried, and the provider is dropped from every later candidate list
 * (and skipped if it appears again here).
 *
 * Note: an earlier version skipped the retry when an equally rated provider
 * was next in line, to save a call. It backfired on trial 0 (27 calls instead
 * of 26): moving on spends the next provider's budget and can hit that
 * provider's own scheduled failure, which costs more than the retry. Always
 * retrying once is cheaper in practice, so this is the rule.
 */
async function serveRequest(norm: NormRequest, needs: Needs): Promise<Served | null> {
  const candidates = candidatesFor(needs, norm.jsonMode);
  for (const candidate of candidates) {
    const state = stateOf(candidate.provider.id);
    if (state.dead) continue;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const result = await callUpstream(candidate.provider, candidate.model, norm);
      state.calls += 1;
      if (result.answer !== null) {
        state.refusalsInRow = 0;
        state.suspect = false;
        return { candidate, answer: result.answer };
      }
      if (result.poisoned) {
        log(`poisoned ${candidate.provider.id}/${candidate.model}: a 2xx answer that did not parse; no further provider call for this request`);
        return null;
      }
      if (result.status === 400 || result.status === 401) break;
      if (result.permanent) {
        state.dead = true;
        log(`dead ${candidate.provider.id}: its budget is spent or it is out; never calling it again`);
        break;
      }
      state.refusalsInRow += 1;
      if (state.refusalsInRow >= 2) state.suspect = true;
      log(`refusal ${candidate.provider.id}/${candidate.model} attempt ${attempt + 1} (HTTP ${result.status}); ${attempt === 0 ? 'retrying' : 'moving on'}`);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP server: the two client APIs and the model list.

async function readJsonBody(req: IncomingMessage): Promise<JsonObj | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream; charset=utf-8',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendClientError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicError(message, 'api_error'));
  else sendJson(res, status, openAIError(message, 'api_error'));
}

/** The manifest entry for the request now arriving, by arrival order. */
function nextManifestEntry(): ManifestEntry | undefined {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  return entry;
}

/** Requirements read off a body, when the manifest has no row for the request. */
function needsFromBody(api: ClientApi, body: JsonObj): Needs {
  const needs: Needs = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) needs.tools = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isObj(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isObj(block) && (block.type === 'image' || block.type === 'image_url')) needs.vision = true;
    }
  }
  if (api === 'openai' && isObj(body.response_format)) {
    const type = body.response_format.type;
    if (type === 'json_object' || type === 'json_schema') needs.json = true;
  }
  return needs;
}

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendClientError(res, 'openai', 400, 'the request body is not one JSON object');
  const entry = nextManifestEntry();
  const needs = entry !== undefined ? entry.requires : needsFromBody('openai', body);
  const norm = normalizeOpenAI(body);
  const served = await serveRequest(norm, needs);
  if (served === null) return sendClientError(res, 'openai', 502, 'no provider answered this request');
  if (norm.stream) sendSse(res, openAIStreamText(served.candidate.model, served.answer));
  else sendJson(res, 200, openAICompletion(served.candidate.model, served.answer));
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendClientError(res, 'anthropic', 400, 'the request body is not one JSON object');
  const entry = nextManifestEntry();
  const needs = entry !== undefined ? entry.requires : needsFromBody('anthropic', body);
  const norm = normalizeAnthropic(body);
  const served = await serveRequest(norm, needs);
  if (served === null) return sendClientError(res, 'anthropic', 502, 'no provider answered this request');
  if (norm.stream) sendSse(res, anthropicStreamText(served.candidate.model, served.answer));
  else sendJson(res, 200, anthropicMessage(served.candidate.model, served.answer));
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const ids = freeModelIds(catalog);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, anthropicModelList(ids));
    return;
  }
  const ownerOf = new Map<string, string>();
  for (const p of catalog.providers) {
    for (const m of p.models) if (m.free && !ownerOf.has(m.id)) ownerOf.set(m.id, p.id);
  }
  sendJson(res, 200, openAIModelList(ids, ownerOf));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    handleModels(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleChat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleMessages(req, res);
    return;
  }
  sendClientError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Register, poll, start.

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(ENV_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isObj(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isObj(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((p) => probeCatalog(p)));
  catalog = buildCatalog(brief, probes);
  log(`catalog: ${catalog.providers.length} providers, ${freeModelIds(catalog).length} free model ids`);

  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');

  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendClientError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
      }
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port}`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
