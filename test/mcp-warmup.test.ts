import { afterEach, describe, expect, test, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LlamaCpp } from "../src/llm.js";
import { startMcpHttpServer } from "../src/mcp/server.js";

const directories: string[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(directories.splice(0).map(path => rm(path, { recursive: true, force: true })));
});

describe("HTTP model warm-up", () => {
  test("warms the reranker only — embed/generate stay lazy", async () => {
    const llm = new LlamaCpp();
    const embed = vi.spyOn(llm, "embed").mockResolvedValue({ embedding: [1], model: "embed" });
    const generate = vi.spyOn(llm, "generate").mockResolvedValue({ text: "ready", model: "generate", done: true });
    const expandQuery = vi.spyOn(llm, "expandQuery").mockResolvedValue([{ type: "vec", text: "readiness" }]);
    const rerank = vi.spyOn(llm, "rerank").mockResolvedValue({ results: [], model: "rerank" });
    await llm.warmUp();
    expect(rerank).toHaveBeenCalledOnce();
    expect(embed).not.toHaveBeenCalled();
    expect(generate).not.toHaveBeenCalled();
    expect(expandQuery).not.toHaveBeenCalled();
    await llm.dispose();
  });

  test("does not listen until all models are warmed", async () => {
    const directory = await mkdtemp(join(tmpdir(), "qmd-warmup-"));
    directories.push(directory);
    let release!: () => void;
    const gate = new Promise<void>(resolve => { release = resolve; });
    const warmUp = vi.spyOn(LlamaCpp.prototype, "warmUp").mockImplementation(() => gate);
    const startup = startMcpHttpServer(0, { dbPath: join(directory, "index.sqlite"), quiet: true, warmModels: true });
    await vi.waitFor(() => expect(warmUp).toHaveBeenCalledOnce());
    release();
    const server = await startup;
    try {
      const response = await fetch(`http://localhost:${server.port}/health`);
      expect(response.status).toBe(200);
    } finally {
      await server.stop();
    }
  });

  test("fails startup when warm-up fails", async () => {
    const directory = await mkdtemp(join(tmpdir(), "qmd-warmup-"));
    directories.push(directory);
    vi.spyOn(LlamaCpp.prototype, "warmUp").mockRejectedValue(new Error("model unavailable"));
    await expect(startMcpHttpServer(0, {
      dbPath: join(directory, "index.sqlite"), quiet: true, warmModels: true,
    })).rejects.toThrow("model unavailable");
  });
});
