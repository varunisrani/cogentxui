import { LLMManager } from '~/lib/modules/llm/manager';
import type { Template } from '~/types/template';

export const WORK_DIR_NAME = 'project';
export const WORK_DIR = `/home/${WORK_DIR_NAME}`;
export const MODIFICATIONS_TAG_NAME = 'bolt_file_modifications';
export const MODEL_REGEX = /^\[Model: (.*?)\]\n\n/;
export const PROVIDER_REGEX = /\[Provider: (.*?)\]\n\n/;
export const DEFAULT_MODEL = 'claude-3-5-sonnet-latest';
export const PROMPT_COOKIE_KEY = 'cachedPrompt';

const llmManager = LLMManager.getInstance(import.meta.env);

export const PROVIDER_LIST = llmManager.getAllProviders();
export const DEFAULT_PROVIDER = llmManager.getDefaultProvider();

// Empty starter templates array
export const STARTER_TEMPLATES: Template[] = [];

// Show raw API responses in UI for debugging
export const SHOW_RAW_API_RESPONSES = true;

// Provider base URL environment keys mapping
export const providerBaseUrlEnvKeys = {
  OpenAI: 'OPENAI_API_BASE_URL',
  Anthropic: 'ANTHROPIC_API_BASE_URL',
  Ollama: 'OLLAMA_API_BASE_URL',
  LMStudio: 'LMSTUDIO_API_BASE_URL',
  OpenAILike: 'OPENAI_LIKE_API_BASE_URL',
  Mistral: 'MISTRAL_API_BASE_URL',
  Cohere: 'COHERE_API_BASE_URL',
  HuggingFace: 'HUGGINGFACE_API_BASE_URL',
  Together: 'TOGETHER_API_BASE_URL',
  Google: 'GOOGLE_API_BASE_URL',
  Perplexity: 'PERPLEXITY_API_BASE_URL',
  AmazonBedrock: 'AMAZON_BEDROCK_API_BASE_URL',
  Groq: 'GROQ_API_BASE_URL',
  OpenRouter: 'OPENROUTER_API_BASE_URL',
  XAI: 'XAI_API_BASE_URL',
  Deepseek: 'DEEPSEEK_API_BASE_URL',
  Hyperbolic: 'HYPERBOLIC_API_BASE_URL'
};
