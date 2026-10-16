/**
 * Unit tests for RemoteLLM — the OpenAI-compatible API backend.
 *
 * Mocks global fetch to assert request shape and response mapping without any
 * network. Tokenization delegates to a stub local backend.
 */

import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";
import { RemoteLLM } from "../src/llm-remote.js";
import type { LlamaCpp } from "../src/llm.js";

// Minimal local tokenizer: only the members RemoteLLM delegates to.
const localStub = {
  tokenize: vi.fn(async () => []),
  detokenize: vi.fn(async () => ""),
  dispose: vi.fn(async () => {}),
} as LlamaCpp;

function makeLLM() {
  return new RemoteLLM({
    baseURL: "https://gw.example/platform/models/",
    apiKey: "test-key",
    embedModel: "cohere.embed-v4:0",
    generateModel: "bedrock.claude-sonnet-5",
    dimensions: 4,
    localTokenizer: localStub,
    timeoutMs: 1000,
  });
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const jsonResponse = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers });

describe("RemoteLLM.embed", () => {
  test("maps data[].embedding and reorders by index", async () => {
    // Return out-of-order to prove reordering.
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        data: [
          { index: 1, embedding: [9, 9] },
          { index: 0, embedding: [1, 2] },
        ],
      }),
    );
    const llm = makeLLM();
    const results = await llm.embedBatch(["a", "b"], { isQuery: false });
    expect(results.map((r) => r?.embedding)).toEqual([[1, 2], [9, 9]]);

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body.input_type).toBe("search_document");
    expect(body.model).toBe("cohere.embed-v4:0");
    expect(body.output_dimension).toBe(4);
  });

  test("sends input_type=search_query for queries", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ data: [{ index: 0, embedding: [1] }] }));
    await makeLLM().embed("q", { isQuery: true });
    const body = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(body.input_type).toBe("search_query");
  });
  test("sends input_type=search_query for query batches", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ data: [{ index: 0, embedding: [1] }, { index: 1, embedding: [2] }] }));
    await makeLLM().embedBatch(["original", "hyde"], { isQuery: true });
    const body = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string);
    expect(body.input_type).toBe("search_query");
    expect(body.input).toEqual(["original", "hyde"]);
  });
});

describe("RemoteLLM.expandQuery", () => {
  test("parses lex/vec/hyde lines sharing a query term", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        choices: [{ message: { content: "lex: auth login\nvec: how authentication works\nnoise line" } }],
      }),
    );
    const q = await makeLLM().expandQuery("auth");
    expect(q).toEqual([
      { type: "lex", text: "auth login" },
      { type: "vec", text: "how authentication works" },
    ]);
  });

  test("falls back to raw query when model returns nothing usable", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ choices: [{ message: { content: "" } }] }));
    const q = await makeLLM().expandQuery("widgets");
    expect(q.some((x) => x.text.includes("widgets"))).toBe(true);
  });
});

describe("RemoteLLM.postJSON retry", () => {
  test("retries a 429 then succeeds", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ error: "rate limited" }, 429, { "retry-after": "0" }))
      .mockResolvedValueOnce(jsonResponse({ data: [{ index: 0, embedding: [1, 2, 3, 4] }] }));
    const result = await makeLLM().embed("x");
    expect(result?.embedding).toEqual([1, 2, 3, 4]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test("does not retry an authentication failure", async () => {
    fetchMock.mockResolvedValue(jsonResponse({ error: "unauthorized" }, 401));
    await expect(makeLLM().embed("x")).rejects.toThrow("HTTP 401");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("RemoteLLM.warmUp", () => {
  test("does not load a local reranker or call the remote API", async () => {
    await makeLLM().warmUp();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
