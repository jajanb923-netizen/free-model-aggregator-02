// free-model-aggregator-02: one server, two client wires, and a real free model
// behind every answer.
//
// The trial runs this server as `node /bot/main.ts` from the repository's root,
// with Node 22's standard library alone (nothing is installed; TypeScript runs
// directly, and a relative import needs its .ts extension). The env is at
// $MAKEFUN_ENV_URL; model calls go through the grader's gateway at
// $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on
// the key the entrant pushed to the competition. The brief at
// $MAKEFUN_PROMPT_FILE carries two documents: the free model list (each free
// OpenRouter model's gateway id, its published rating in milli-points, what it
// can do, and its free flag) and the six prompts (each one's id, its api, its
// streaming, its weight, and the requirements a model must meet to serve it).
//
// The server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. serves GET /v1/models (answering in the shape of the auth header it
//      got: x-api-key means Anthropic, Authorization means OpenAI),
//      POST /v1/chat/completions and POST /v1/messages, each with a JSON body
//      or with server-sent events;
//   4. answers each prompt by dialling ONE free model through the gateway.
//      OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//      OpenRouter key the entrant pushed; the real key never enters the sandbox;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog is that same document, byte for
// byte.
//
// ROUTING (quality is 60% of the score)
//   Every prompt is served by the best free model that meets its requirements,
//   ranked by the brief's rating_milli, ties by the brief's order: inkling-2-small
//   (1475) for every prompt except the JSON-mode ones, which need a JSON-capable
//   model, so super-vl (1303) leads there, then gemma-4-31b (1210), then the
//   1150-rated JSON-capable models in brief order. A model that can meet the
//   prompt never loses to a weaker one on a guess.
//
// FAILOVER (10% of the score) AND THE TRAIL
//   A gateway call answered with a 4xx or a 5xx is not an answer: the same
//   candidate is retried up to twice (three attempts in all) before the server
//   fails over to the next model that meets the prompt, and it keeps trying
//   eligible models until one serves the prompt or the list is exhausted. A
//   prompt given up on scores nothing, so giving up early is the only real
//   loss. Every refusal is recorded in the order it happened, and the rows it
//   recovered from are reported on the response in the header
//   `x-makefun-upstream-failures` (at most 8 rows of {status, code, model}):
//   the header is the only way the failover column sees a retried refusal,
//   because the env sees the answers and never the gateway's traffic. The
//   header is set on a served answer and on a 502 that gave up, alike.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the
//   model it actually dialled. The house records every gateway call and scores
//   the model that really answered. This server sends the brief's model id in
//   the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's
//   `openrouter/` prefix), so its claim is the call it made, never another.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the gateway's
//   answers: no wall clock and no unseeded randomness anywhere, so a replay
//   leaves the same state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  candidatesFor,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  errorCodeOf,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway refuses a call with no output cap, or one over this cap. */
const MAX_GATEWAY_TOKENS = 8192;
/** The output cap a gateway call names when the client named none. */
const DEFAULT_GATEWAY_TOKENS = 1024;
/** Attempts on one candidate before failing over to the next (retry twice). */
const ATTEMPTS_PER_MODEL = 3;
/** The most refusal rows the failures trail carries, per the rules. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible
 * root (`…/v1` on the trial network); a base that does not already end in
 * `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1')
    ? `${GATEWAY_URL}/chat/completions`
    : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's
  // requirements; this fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) {
        vision = true;
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One refusal row for the failures trail, in the order the calls refused. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/**
 * A Chat Completions body the gateway accepts: the client's body with the
 * dialled model's id, an output cap always named (the gateway refuses a call
 * with none, or one over 8192), and usage included on a streamed call.
 */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens : DEFAULT_GATEWAY_TOKENS;
  out.max_tokens = Math.min(Math.max(1, Math.floor(cap)), MAX_GATEWAY_TOKENS);
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
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
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The failures trail as one header value, or null when there is nothing to report. */
function trailHeader(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  return JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS));
}

/** Headers every response carries: JSON, and the failures trail when the prompt recovered from refusals. */
function responseHeaders(failures: readonly FailureRow[]): Record<string, string> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  return headers;
}

function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  failures: readonly FailureRow[] = [],
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { ...responseHeaders(failures), 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  failures: readonly FailureRow[] = [],
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** Headers a streamed response carries: SSE, and the failures trail when the prompt recovered. */
function streamHeaders(failures: readonly FailureRow[]): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  return headers;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(gatewayBody(body, model)),
  });
}

/**
 * The outcome of dialling one candidate once: the gateway's response when it
 * answered at all, and the refusal row when it did not answer 2xx.
 */
interface DialOutcome {
  response: Response | null;
  failure: FailureRow | null;
}

async function dialOnce(model: string, body: Json): Promise<DialOutcome> {
  let upstream: Response;
  try {
    upstream = await dial(model, body);
  } catch (err) {
    return {
      response: null,
      failure: {
        status: 502,
        code: 'upstream_unreachable',
        model,
      },
    };
  }
  if (upstream.ok && upstream.body !== null) return { response: upstream, failure: null };
  let text = '';
  try {
    text = await upstream.text();
  } catch {
    // The refusal body could not be read: the status alone is the row.
  }
  return {
    response: null,
    failure: { status: upstream.status, code: errorCodeOf(text), model },
  };
}

/**
 * Dial the candidates in order, retrying each one up to twice before failing
 * over to the next, until one serves the prompt or the list is exhausted.
 * Always returns the refusal trail: on success, the serving response and the
 * model that served it; on exhaustion, nulls and the full trail, so the 502
 * that gives up still reports the refusals behind it.
 */
async function dialWithFailover(
  candidates: readonly ModelRow[],
  body: Json,
): Promise<{ response: Response | null; model: string | null; failures: FailureRow[] }> {
  const failures: FailureRow[] = [];
  for (const candidate of candidates) {
    for (let attempt = 0; attempt < ATTEMPTS_PER_MODEL; attempt += 1) {
      const outcome = await dialOnce(candidate.id, body);
      if (outcome.response !== null) {
        return { response: outcome.response, model: candidate.id, failures };
      }
      if (outcome.failure !== null) failures.push(outcome.failure);
    }
  }
  return { response: null, model: null, failures };
}

/**
 * A prompt every candidate refused, as a 502 in the client's wire, with the
 * trail attached: the env reads the trail off the response, so the failover
 * column sees the gateway refusals behind even a prompt this server lost.
 */
function gaveUp(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  failures: readonly FailureRow[],
): void {
  const last = failures[failures.length - 1];
  const detail =
    last === undefined
      ? 'no free model in the brief meets this request'
      : `the gateway refused every eligible model (last: HTTP ${last.status}${last.code === '' ? '' : ` ${last.code}`} on ${last.model ?? 'no model'})`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail, failures);
  else sendError(res, wire, 502, 'upstream_error', detail, failures);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const candidates = candidatesFor(brief, requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  const served = await dialWithFailover(candidates, body);
  if (served.response === null || served.model === null) {
    return gaveUp(res, 'openai', served.failures);
  }
  const { response: upstream, model, failures } = served;
  if (body.stream === true) {
    res.writeHead(200, streamHeaders(failures));
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const candidates = candidatesFor(brief, requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  const upstreamBody = anthropicToOpenAI(body, candidates[0].id);
  // The translated body is dialled as-is: the model is replaced per candidate
  // inside dialWithFailover by gatewayBody, which keeps the translation.
  const served = await dialWithFailover(candidates, upstreamBody);
  if (served.response === null || served.model === null) {
    return gaveUp(res, 'anthropic', served.failures);
  }
  const { response: upstream, model, failures } = served;
  if (body.stream === true) {
    res.writeHead(200, streamHeaders(failures));
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) {
    throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  }
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') {
    log(`registered: the trial holds ${reply.requests} prompts`);
  }
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
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
        if (isRecord(work) && work.done === true) {
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
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
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
