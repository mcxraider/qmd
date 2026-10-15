/**
 * Remote (API) LLM backend for QMD.
 *
 * Speaks the OpenAI-compatible REST surface exposed by gateways such as the
 * government PlatformAI gateway (verified shapes: `/v1/embeddings`,
 * `/v1/chat/completions`). Embeddings and query expansion / generation go over
 * the network. The TypeSafe wrapper handles reranking in query paths.
 *
 * Uses native `fetch` — no vendor SDK dependency.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
  LlamaCpp,
} from "./llm.js";
import { parseExpansionLines, expansionFallback, registerRemoteEmbedModel } from "./llm.js";
import type { Token as LlamaToken } from "node-llama-cpp";

// Cohere embed-v4 accepts up to 96 inputs per request (mirrors the reference impl).
const MAX_BATCH_SIZE = 96;
const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504]);
const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_RETRIES = 3;

export type RemoteLLMConfig = {
  baseURL: string;
  apiKey: string;
  embedModel: string;
  generateModel: string;
  /** output_dimension sent to the embeddings endpoint (e.g. 1536 for cohere.embed-v4:0) */
  dimensions?: number;
  /** On-device backend used for reranking (and tokenizer-based chunking upstream). */
  localForRerank: LlamaCpp;
  timeoutMs?: number;
};

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class RemoteLLM implements LLM {
  private baseURL: string;
  private apiKey: string;
  private embedModel: string;
  private generateModel: string;
  private dimensions?: number;
  private local: LlamaCpp;
  private timeoutMs: number;

  constructor(config: RemoteLLMConfig) {
    this.baseURL = config.baseURL.trim().replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.embedModel = config.embedModel;
    this.generateModel = config.generateModel;
    this.dimensions = config.dimensions;
    this.local = config.localForRerank;
    this.timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    // Tell the embedding formatters to leave text untouched for this model:
    // the API uses the `input_type` request field, not an in-text task prefix.
    registerRemoteEmbedModel(this.embedModel);
  }

  /** POST JSON with Bearer auth, a timeout, and retry/backoff on transient status codes. */
  private async postJSON(path: string, body: unknown): Promise<any> {
    const url = `${this.baseURL}${path}`;
    let lastErr: unknown;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      let resp: Response;
      try {
        resp = await fetch(url, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(this.timeoutMs),
        });
      } catch (err) {
        lastErr = err;
        // Only transport failures and timeouts reach this catch.
        if (attempt < MAX_RETRIES) {
          await sleep(Math.min(8000, 500 * 2 ** attempt) + Math.random() * 250);
          continue;
        }
        break;
      }
      if (resp.ok) return await resp.json();
      if (RETRYABLE_STATUS.has(resp.status) && attempt < MAX_RETRIES) {
        const retryAfter = Number(resp.headers.get("retry-after"));
        const delay = Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter * 1000
          : Math.min(8000, 500 * 2 ** attempt) + Math.random() * 250;
        await sleep(delay);
        continue;
      }
      const responseText = await resp.text().catch(() => "");
      throw new Error(`Remote LLM ${path} failed: HTTP ${resp.status} ${responseText.slice(0, 300)}`);
    }
    throw lastErr instanceof Error ? lastErr : new Error(`Remote LLM ${path} failed`);
  }

  private async embedInputs(inputs: string[], isQuery: boolean): Promise<EmbeddingResult[]> {
    const data = await this.postJSON("/v1/embeddings", {
      model: this.embedModel,
      input: inputs,
      input_type: isQuery ? "search_query" : "search_document",
      ...(this.dimensions ? { output_dimension: this.dimensions } : {}),
      truncate: "NONE",
      embedding_types: ["float"],
    });
    const items = (data?.data ?? []) as { index: number; embedding: number[] }[];
    // Reorder by index — the gateway does not guarantee response order.
    const ordered = [...items].sort((a, b) => a.index - b.index);
    if (ordered.length !== inputs.length) {
      throw new Error(`Remote embeddings returned ${ordered.length} vectors for ${inputs.length} inputs`);
    }
    return ordered.map((it) => ({ embedding: it.embedding, model: this.embedModel }));
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const [result] = await this.embedInputs([text], options.isQuery ?? false);
    return result ?? null;
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    const isQuery = options.isQuery ?? false;
    const out: (EmbeddingResult | null)[] = [];
    for (let i = 0; i < texts.length; i += MAX_BATCH_SIZE) {
      const batch = texts.slice(i, i + MAX_BATCH_SIZE);
      const results = await this.embedInputs(batch, isQuery);
      out.push(...results);
    }
    return out;
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const data = await this.postJSON("/v1/chat/completions", {
      model: options.model ?? this.generateModel,
      messages: [{ role: "user", content: prompt }],
      ...(options.maxTokens != null ? { max_tokens: options.maxTokens } : {}),
    });
    const content = data?.choices?.[0]?.message?.content;
    if (typeof content !== "string") return null;
    return { text: content, model: data?.model ?? this.generateModel, done: true };
  }

  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;
    try {
      const data = await this.postJSON("/v1/chat/completions", {
        model: this.generateModel,
        messages: [
          {
            role: "system",
            content:
              "You expand a search query into alternative sub-queries for a hybrid retrieval system. " +
              "Output one sub-query per line, each prefixed with its type and a colon:\n" +
              "lex: <keyword query>\nvec: <semantic paraphrase>\nhyde: <a hypothetical answer passage>\n" +
              "Emit 2-5 lines. Reuse the query's key terms. Output ONLY these lines, no commentary.",
          },
          { role: "user", content: query },
        ],
        max_tokens: 600,
      });
      const content: string = data?.choices?.[0]?.message?.content ?? "";
      const parsed = parseExpansionLines(content, query, includeLexical);
      return parsed.length > 0 ? parsed : expansionFallback(query, includeLexical);
    } catch (err) {
      console.error("Remote query expansion failed:", err);
      return expansionFallback(query, includeLexical);
    }
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    // No rerank endpoint on the gateway — use the on-device reranker.
    return this.local.rerank(query, documents, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async dispose(): Promise<void> {
    await this.local.dispose();
  }

  get embedModelName(): string {
    return this.embedModel;
  }

  get generateModelName(): string {
    return this.generateModel;
  }

  get rerankModelName(): string {
    return this.local.rerankModelName;
  }

  // Token-boundary chunking uses the local tokenizer; it does not need to match
  // the remote embedding tokenizer (chunk sizes are approximate budgets).
  tokenize(text: string): Promise<readonly LlamaToken[]> {
    return this.local.tokenize(text);
  }

  detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    return this.local.detokenize(tokens);
  }

  async warmUp(): Promise<void> {
    // Only the local reranker loads a model into memory; embed/generate are
    // remote API calls with nothing to warm (and we avoid a paid call on every
    // daemon startup).
    const ranking = await this.rerank("readiness", [{ file: "warm-up", text: "QMD readiness check" }]);
    if (ranking.model === "fallback") throw new Error("Reranking model warm-up failed");
  }
}
