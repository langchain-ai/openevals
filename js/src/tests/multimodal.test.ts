import * as ls from "langsmith/vitest";
import { expect, beforeAll } from "vitest";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { _attachmentToContentBlock } from "../utils.js";
import { _createLLMAsJudgeScorer, createLLMAsJudge } from "../llm.js";
import { IMAGE_RELEVANCE_PROMPT } from "../prompts/image/image_relevance.js";
import { AUDIO_QUALITY_PROMPT } from "../prompts/voice/audio_quality.js";
import { TRANSCRIPTION_ACCURACY_PROMPT } from "../prompts/voice/transcription_accuracy.js";
import { VOCAL_AFFECT_PROMPT } from "../prompts/voice/vocal_affect.js";
import { DIALOGUE_FLOW_PROMPT } from "../prompts/voice/dialogue_flow.js";

const TINY_PNG_DATA_URI =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";

// Minimal silent WAV (44-byte header, no samples) — satisfies {attachments} template.
const TINY_WAV_DATA_URI =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=";
const TINY_PNG_BASE64 = TINY_PNG_DATA_URI.split(",")[1];

const PUBLIC_IMAGE_URL =
  "https://images.everydayhealth.com/images/2025/fruits-with-protein-help-boost-intake-pomegranate-1440x810.jpg";

// ── _attachmentToContentBlock unit tests ──────────────────────────────────────

ls.describe("_attachmentToContentBlock", () => {
  // ── image/* ─────────────────────────────────────────────────────────────────

  ls.test("image/png data URI → image_url", { inputs: {} }, async () => {
    const block = _attachmentToContentBlock({ mime_type: "image/png", data: TINY_PNG_DATA_URI });
    expect(block).toEqual({ type: "image_url", image_url: { url: TINY_PNG_DATA_URI } });
  });

  ls.test("image/jpeg data URI → image_url", { inputs: {} }, async () => {
    const data = "data:image/jpeg;base64,/9j/abc123";
    const block = _attachmentToContentBlock({ mime_type: "image/jpeg", data });
    expect(block).toEqual({ type: "image_url", image_url: { url: data } });
  });

  ls.test("plain URL string → image_url", { inputs: {} }, async () => {
    const url = "https://example.com/photo.jpg";
    const block = _attachmentToContentBlock(url);
    expect(block).toEqual({ type: "image_url", image_url: { url } });
  });

  // ── application/pdf ──────────────────────────────────────────────────────────

  ls.test("pdf with name → file block", { inputs: {} }, async () => {
    const data = "data:application/pdf;base64,JVBERi0x";
    const block = _attachmentToContentBlock({ mime_type: "application/pdf", data, name: "invoice.pdf" });
    expect(block).toEqual({ type: "file", file: { filename: "invoice.pdf", file_data: data } });
  });

  ls.test("pdf without name → attachment.pdf default", { inputs: {} }, async () => {
    const data = "data:application/pdf;base64,JVBERi0x";
    const block = _attachmentToContentBlock({ mime_type: "application/pdf", data });
    expect(block).toEqual({ type: "file", file: { filename: "attachment.pdf", file_data: data } });
  });

  // ── audio/* ──────────────────────────────────────────────────────────────────

  ls.test("audio/mp3 strips data: prefix", { inputs: {} }, async () => {
    const rawBase64 = "SUQzBAAAAAAAI1RTU0U=";
    const data = `data:audio/mp3;base64,${rawBase64}`;
    const block = _attachmentToContentBlock({ mime_type: "audio/mp3", data });
    expect(block).toEqual({ type: "input_audio", input_audio: { data: rawBase64, format: "mp3" } });
  });

  ls.test("audio/wav without data: prefix passes through", { inputs: {} }, async () => {
    const rawBase64 = "UklGRiQAAABXQVZF";
    const block = _attachmentToContentBlock({ mime_type: "audio/wav", data: rawBase64 });
    expect(block).toEqual({ type: "input_audio", input_audio: { data: rawBase64, format: "wav" } });
  });

  ls.test("audio/mpeg normalizes to mp3 format", { inputs: {} }, async () => {
    const rawBase64 = "SUQzBAA";
    const data = `data:audio/mpeg;base64,${rawBase64}`;
    const block = _attachmentToContentBlock({ mime_type: "audio/mpeg", data });
    expect(block).toEqual({ type: "input_audio", input_audio: { data: rawBase64, format: "mp3" } });
  });

  // ── pre-formatted passthrough ────────────────────────────────────────────────

  ls.test("pre-formatted image_url block passes through", { inputs: {} }, async () => {
    const contentBlock = { type: "image_url", image_url: { url: "https://example.com/img.png" } };
    expect(_attachmentToContentBlock(contentBlock)).toEqual(contentBlock);
  });

  ls.test("pre-formatted input_audio block passes through", { inputs: {} }, async () => {
    const contentBlock = { type: "input_audio", input_audio: { data: "abc", format: "mp3" } };
    expect(_attachmentToContentBlock(contentBlock)).toEqual(contentBlock);
  });

  // ── error cases ──────────────────────────────────────────────────────────────

  ls.test("unsupported MIME type throws", { inputs: {} }, async () => {
    expect(() =>
      _attachmentToContentBlock({ mime_type: "video/mp4", data: "data:video/mp4;base64,abc" })
    ).toThrow("Unsupported attachment MIME type");
  });

  ls.test("non-string non-object raises", { inputs: {} }, async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(() => _attachmentToContentBlock(12345 as any)).toThrow("Unsupported attachment type");
  });
});

// ── _createLLMAsJudgeScorer message construction (mocked judge) ───────────────

ls.describe("_createLLMAsJudgeScorer message construction", () => {
  function makeMockJudge() {
    let capturedMessages: unknown[] = [];
    const judge = {
      _modelType: () => "base_chat_model" as const,
      withStructuredOutput: (_schema: unknown) => ({
        invoke: async (messages: unknown[]) => {
          capturedMessages = [...messages];
          return { score: true, reasoning: "ok" };
        },
      }),
    } as unknown as BaseChatModel;
    return { judge, getMessages: () => capturedMessages };
  }

  ls.test("single attachment becomes list content", { inputs: {} }, async () => {
    const { judge, getMessages } = makeMockJudge();
    const scorer = _createLLMAsJudgeScorer({
      prompt: "Evaluate this image: {outputs}\n{attachments}",
      judge,
    });

    await scorer({
      outputs: "a fruit bowl",
      attachments: { mime_type: "image/png", data: TINY_PNG_DATA_URI },
    });

    const messages = getMessages() as Array<{ role: string; content: unknown }>;
    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe("user");
    const content = messages[0].content as Array<Record<string, unknown>>;
    expect(Array.isArray(content)).toBe(true);
    expect(content[0].type).toBe("text");
    expect(content[1].type).toBe("image_url");
    expect((content[1].image_url as Record<string, unknown>).url).toBe(TINY_PNG_DATA_URI);
  });

  ls.test("multiple attachments all appended", { inputs: {} }, async () => {
    const { judge, getMessages } = makeMockJudge();
    const scorer = _createLLMAsJudgeScorer({
      prompt: "Evaluate: {outputs}\n{attachments}",
      judge,
    });

    await scorer({
      outputs: "two images",
      attachments: [
        { mime_type: "image/png", data: TINY_PNG_DATA_URI },
        { mime_type: "image/jpeg", data: "data:image/jpeg;base64,/9j/" },
      ],
    });

    const messages = getMessages() as Array<{ role: string; content: unknown }>;
    const content = messages[0].content as Array<Record<string, unknown>>;
    expect(content).toHaveLength(3); // text + 2 images
    expect(content[0].type).toBe("text");
    expect(content[1].type).toBe("image_url");
    expect(content[2].type).toBe("image_url");
  });

  ls.test("no attachment content is plain string", { inputs: {} }, async () => {
    const { judge, getMessages } = makeMockJudge();
    const scorer = _createLLMAsJudgeScorer({
      prompt: "Evaluate: {outputs}",
      judge,
    });

    await scorer({ outputs: "some output" });

    const messages = getMessages() as Array<{ role: string; content: unknown }>;
    expect(typeof messages[0].content).toBe("string");
  });
});

// ── LLM integration (requires API key + vision model) ────────────────────────

ls.describe("LLM Judge Image Relevance", () => {
  let fruitImageB64: string | undefined;

  beforeAll(async () => {
    try {
      const resp = await fetch(PUBLIC_IMAGE_URL, {
        headers: { "User-Agent": "openevals-test/1.0" },
      });
      if (!resp.ok) return;
      const buffer = await resp.arrayBuffer();
      const base64 = Buffer.from(buffer).toString("base64");
      fruitImageB64 = `data:image/jpeg;base64,${base64}`;
    } catch {
      // fruitImageB64 stays undefined; tests will skip
    }
  });

  ls.test("image relevance — relevant image", {
    inputs: {
      inputs: "Show me a picture of fruits",
      outputs: "Here is an image of various fruits",
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    if (!fruitImageB64) return;
    const evaluator = createLLMAsJudge({
      prompt: IMAGE_RELEVANCE_PROMPT,
      feedbackKey: "image_relevance",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "image/jpeg", data: fruitImageB64 },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("image relevance — irrelevant image", {
    inputs: {
      inputs: "Show me a photo of a sports car",
      outputs: "Here is a red Ferrari",
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    if (!fruitImageB64) return;
    const evaluator = createLLMAsJudge({
      prompt: IMAGE_RELEVANCE_PROMPT,
      feedbackKey: "image_relevance",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "image/jpeg", data: fruitImageB64 },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });
});

// ── LLM integration: voice (requires API key + audio model) ──────────────────

ls.describe("LLM Judge Voice", () => {
  ls.test("audio quality — issues detected", {
    inputs: {
      inputs: "Customer service call recording",
      outputs: "Audio exhibits severe clipping throughout with loud pops and dropouts every few seconds",
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: AUDIO_QUALITY_PROMPT,
      feedbackKey: "audio_quality",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("audio quality — clean", {
    inputs: {
      inputs: "Customer service call recording",
      outputs: "Audio is clear with no distortion, clipping, or glitches detected",
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: AUDIO_QUALITY_PROMPT,
      feedbackKey: "audio_quality",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  ls.test("transcription — accurate", {
    inputs: {
      inputs: "Please reschedule the meeting with Dr. Johnson to Thursday at 3pm.",
      outputs: "Please reschedule the meeting with Dr. Johnson to Thursday at 3pm.",
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: TRANSCRIPTION_ACCURACY_PROMPT,
      feedbackKey: "transcription_accuracy",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("transcription — inaccurate", {
    inputs: {
      inputs: "Please reschedule the meeting with Dr. Johnson to Thursday at 3pm.",
      outputs: "Please cancel the meeting with Dr. Jackson to Friday at 2pm.",
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: TRANSCRIPTION_ACCURACY_PROMPT,
      feedbackKey: "transcription_accuracy",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  ls.test("vocal affect — appropriate", {
    inputs: {
      inputs: "User asks for help with a billing issue",
      outputs: "Agent speaks with warm, empathetic tone throughout, using natural pacing and appropriate emphasis",
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: VOCAL_AFFECT_PROMPT,
      feedbackKey: "vocal_affect",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("vocal affect — inappropriate", {
    inputs: {
      inputs: "User explains they just lost their job and need urgent financial help",
      outputs: "Agent responds with cheerful, upbeat tone as if reading a promotional script, with rapid delivery",
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: VOCAL_AFFECT_PROMPT,
      feedbackKey: "vocal_affect",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  ls.test("dialogue flow — natural", {
    inputs: {
      inputs: "Support call between agent and customer",
      outputs: "Clean turn-taking throughout; each speaker waited for the other to finish before responding",
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: DIALOGUE_FLOW_PROMPT,
      feedbackKey: "dialogue_flow",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("dialogue flow — unnatural", {
    inputs: {
      inputs: "Support call between agent and customer",
      outputs: "Frequent cross-talk with agent and user speaking simultaneously throughout, causing confusion and requiring constant repetition",
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: DIALOGUE_FLOW_PROMPT,
      feedbackKey: "dialogue_flow",
      model: "google-genai:gemini-2.0-flash",
    });
    const result = await evaluator({
      inputs: inputs.inputs,
      outputs: inputs.outputs,
      attachments: { mime_type: "audio/wav", data: TINY_WAV_DATA_URI },
    });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });
});
