import * as ls from "langsmith/vitest";
import { expect } from "vitest";
import { createLLMAsJudge } from "../llm.js";
import {
  TASK_COMPLETION_PROMPT,
  USER_SATISFACTION_PROMPT,
  AGENT_TONE_PROMPT,
  KNOWLEDGE_RETENTION_PROMPT,
  LANGUAGE_DETECTION_PROMPT,
} from "../prompts/trajectory/index.js";

ls.describe("LLM Judge Trajectory", () => {
  // ── TASK_COMPLETION_PROMPT ──────────────────────────────────────────────────

  ls.test("task completion — all requests fulfilled", {
    inputs: {
      conversation: [
        { role: "user", content: "Can you summarize this article for me?" },
        { role: "assistant", content: "Sure! The article discusses climate change impacts on coastal cities, rising sea levels, and mitigation strategies." },
        { role: "user", content: "Thanks, that was exactly what I needed." },
      ],
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: TASK_COMPLETION_PROMPT,
      feedbackKey: "task_completion",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("task completion — request not fulfilled", {
    inputs: {
      conversation: [
        { role: "user", content: "Can you book a flight from NYC to Paris?" },
        { role: "assistant", content: "I can provide flight information but cannot book flights." },
        { role: "user", content: "I asked you to book it, not give info. Please just do it." },
        { role: "assistant", content: "I'm unable to make bookings." },
      ],
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: TASK_COMPLETION_PROMPT,
      feedbackKey: "task_completion",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  // ── USER_SATISFACTION_PROMPT ────────────────────────────────────────────────

  ls.test("user satisfaction — happy user", {
    inputs: {
      conversation: [
        { role: "user", content: "How do I reset my password?" },
        { role: "assistant", content: "Go to Settings > Account > Reset Password." },
        { role: "user", content: "Perfect, that worked! Thank you so much." },
      ],
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: USER_SATISFACTION_PROMPT,
      feedbackKey: "user_satisfaction",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("user satisfaction — frustrated user", {
    inputs: {
      conversation: [
        { role: "user", content: "I need a refund for my order." },
        { role: "assistant", content: "Please contact our billing department." },
        { role: "user", content: "I already did that three times. This is useless. I'll just dispute it with my bank." },
      ],
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: USER_SATISFACTION_PROMPT,
      feedbackKey: "user_satisfaction",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  // ── AGENT_TONE_PROMPT ───────────────────────────────────────────────────────

  ls.test("agent tone — appropriate", {
    inputs: {
      conversation: [
        { role: "user", content: "I'm confused about how this works." },
        { role: "assistant", content: "No worries at all — it can be tricky at first! Let me walk you through it step by step." },
        { role: "user", content: "That makes much more sense now." },
        { role: "assistant", content: "Great! Let me know if you have any other questions." },
      ],
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: AGENT_TONE_PROMPT,
      feedbackKey: "agent_tone",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("agent tone — condescending", {
    inputs: {
      conversation: [
        { role: "user", content: "I don't understand this feature." },
        { role: "assistant", content: "This is extremely basic. It's in the documentation. Did you even read it?" },
        { role: "user", content: "I did but it wasn't clear to me." },
        { role: "assistant", content: "Well, most users find it obvious. Maybe try reading more carefully." },
      ],
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: AGENT_TONE_PROMPT,
      feedbackKey: "agent_tone",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  // ── KNOWLEDGE_RETENTION_PROMPT ──────────────────────────────────────────────

  ls.test("knowledge retention — agent remembers context", {
    inputs: {
      conversation: [
        { role: "user", content: "My name is Alex and I'm planning a trip to Japan in March." },
        { role: "assistant", content: "That sounds exciting, Alex! March is cherry blossom season." },
        { role: "user", content: "What should I pack?" },
        { role: "assistant", content: "For Japan in March, bring layers and a light jacket for cherry blossom viewing, Alex." },
      ],
    },
    referenceOutputs: { score: true },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: KNOWLEDGE_RETENTION_PROMPT,
      feedbackKey: "knowledge_retention",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeTruthy();
  });

  ls.test("knowledge retention — agent forgets context", {
    inputs: {
      conversation: [
        { role: "user", content: "My name is Sarah and I'm vegetarian." },
        { role: "assistant", content: "Nice to meet you, Sarah!" },
        { role: "user", content: "Can you recommend a restaurant for dinner?" },
        { role: "assistant", content: "I'd suggest a great steakhouse — excellent grilled meats and a fantastic ribeye." },
      ],
    },
    referenceOutputs: { score: false },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: KNOWLEDGE_RETENTION_PROMPT,
      feedbackKey: "knowledge_retention",
      model: "openai:gpt-5-mini",
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: result.score });
    expect(result.score).toBeFalsy();
  });

  // ── LANGUAGE_DETECTION_PROMPT ───────────────────────────────────────────────

  ls.test("language detection — Spanish", {
    inputs: {
      conversation: [
        { role: "user", content: "Hola, ¿cómo estás?" },
        { role: "assistant", content: "¡Hola! Estoy bien, gracias." },
        { role: "user", content: "Necesito ayuda con mi cuenta." },
      ],
    },
    referenceOutputs: { score: "Spanish" },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: LANGUAGE_DETECTION_PROMPT,
      feedbackKey: "language_detection",
      model: "openai:gpt-5-mini",
      outputSchema: {
        type: "object",
        properties: {
          reasoning: { type: "string" },
          score: { type: "string", description: "The detected language name in English" },
        },
        required: ["reasoning", "score"],
        additionalProperties: false,
      },
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: (result as any).score });
    expect((result as any).score.toLowerCase()).toBe("spanish");
  });

  ls.test("language detection — French", {
    inputs: {
      conversation: [
        { role: "user", content: "Bonjour, j'ai besoin d'aide." },
        { role: "assistant", content: "Bonjour! Comment puis-je vous aider?" },
        { role: "user", content: "Je ne comprends pas cette fonctionnalité." },
      ],
    },
    referenceOutputs: { score: "French" },
  }, async ({ inputs }) => {
    const evaluator = createLLMAsJudge({
      prompt: LANGUAGE_DETECTION_PROMPT,
      feedbackKey: "language_detection",
      model: "openai:gpt-5-mini",
      outputSchema: {
        type: "object",
        properties: {
          reasoning: { type: "string" },
          score: { type: "string", description: "The detected language name in English" },
        },
        required: ["reasoning", "score"],
        additionalProperties: false,
      },
    });
    const result = await evaluator({ outputs: inputs.conversation });
    ls.logOutputs({ score: (result as any).score });
    expect((result as any).score.toLowerCase()).toBe("french");
  });
});
