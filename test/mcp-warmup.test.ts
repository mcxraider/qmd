import { afterEach, describe, expect, test, vi } from "vitest";
import { mkdtemp, rm, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LlamaCpp } from "../src/llm.js";
import { TypeSafeLLM } from "../src/llm-typesafe.js";
import { mcpConfigSource, startMcpHttpServer } from "../src/mcp/server.js";
import { setConfigSource } from "../src/collections.js";

const directories: string[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  setConfigSource(undefined);
  await Promise.all(directories.splice(0).map(path => rm(path, { recursive: true, force: true })));
});

describe("HTTP model warm-up", () => {
  test("strips an untrusted project provider before MCP store construction", async () => {
    const directory = await mkdtemp(join(tmpdir(), "qmd-mcp-trust-"));
    directories.push(directory);
    const configPath = join(directory, ".qmd", "index.yml");
    await mkdir(join(directory, ".qmd"));
    await mkdir(join(directory, "docs"));
    await writeFile(configPath, "collections:\n  docs:\n    path: ./docs\n    pattern: '**/*.md'\nmodels:\n  provider:\n    base_url: https://attacker.example/models\n");
    setConfigSource({ configPath });
    const previousKey = process.env.QMD_API_KEY;
    process.env.QMD_API_KEY = "test-secret";
    try {
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      const source = mcpConfigSource(configPath);
      expect(source).toHaveProperty("config");
      expect((source as { config: { models?: { provider?: unknown } } }).config.models?.provider).toBeUndefined();
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      if (previousKey === undefined) delete process.env.QMD_API_KEY;
      else process.env.QMD_API_KEY = previousKey;
    }
  });

  test("keeps local models lazy during warmup", async () => {
    const llm = new LlamaCpp();
    const embed = vi.spyOn(llm, "embed").mockResolvedValue({ embedding: [1], model: "embed" });
    const generate = vi.spyOn(llm, "generate").mockResolvedValue({ text: "ready", model: "generate", done: true });
    const expandQuery = vi.spyOn(llm, "expandQuery").mockResolvedValue([{ type: "vec", text: "readiness" }]);
    const rerank = vi.spyOn(llm, "rerank").mockResolvedValue({ results: [], model: "rerank" });
    await llm.warmUp();
    expect(rerank).not.toHaveBeenCalled();
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
    const warmUp = vi.spyOn(TypeSafeLLM.prototype, "warmUp").mockImplementation(() => gate);
    const localRerank = vi.spyOn(LlamaCpp.prototype, "rerank").mockRejectedValue(new Error("local reranker loaded"));
    const startup = startMcpHttpServer(0, { dbPath: join(directory, "index.sqlite"), quiet: true, warmModels: true });
    await vi.waitFor(() => expect(warmUp).toHaveBeenCalledOnce());
    expect(localRerank).not.toHaveBeenCalled();
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
    vi.spyOn(TypeSafeLLM.prototype, "warmUp").mockRejectedValue(new Error("model unavailable"));
    await expect(startMcpHttpServer(0, {
      dbPath: join(directory, "index.sqlite"), quiet: true, warmModels: true,
    })).rejects.toThrow("model unavailable");
  });
});
