import { afterEach, describe, expect, test, vi } from "vitest";
import { MissingTypeSafeKeyError, TypeSafeLLM } from "../src/llm-typesafe.js";
import type { LLM } from "../src/llm.js";

const base = {} as LLM;
const previousKey = process.env.TYPESAFE_API_KEY;
afterEach(() => {
  vi.unstubAllGlobals();
  if (previousKey === undefined) delete process.env.TYPESAFE_API_KEY;
  else process.env.TYPESAFE_API_KEY = previousKey;
});

describe("TypeSafe reranking", () => {
  test("requires a key for reranked queries", async () => {
    delete process.env.TYPESAFE_API_KEY;
    await expect(new TypeSafeLLM(base).rerank("query", [{ file: "a", text: "text" }]))
      .rejects.toBeInstanceOf(MissingTypeSafeKeyError);
  });

  test("scores each passage and preserves its source index", async () => {
    process.env.TYPESAFE_API_KEY = "test-key";
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string);
      const score = body.state.passage === "answer" ? 0.9 : 0.1;
      return { ok: true, json: async () => ({ answers: { relevant: { type: "noul", noul: score } } }) };
    });
    vi.stubGlobal("fetch", fetchMock);
    const result = await new TypeSafeLLM(base).rerank("question", [
      { file: "a", text: "other" }, { file: "b", text: "answer" },
    ]);
    expect(result.results).toEqual([
      { file: "b", index: 1, score: 0.9 },
      { file: "a", index: 0, score: 0.1 },
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test("rejects invalid scores so they cannot enter the cache", async () => {
    process.env.TYPESAFE_API_KEY = "test-key";
    vi.stubGlobal("fetch", async () => ({ ok: true, json: async () => ({ answers: { relevant: { type: "noul", noul: 2 } } }) }));
    await expect(new TypeSafeLLM(base).rerank("query", [{ file: "a", text: "text" }]))
      .rejects.toThrow("invalid score");
  });
});
