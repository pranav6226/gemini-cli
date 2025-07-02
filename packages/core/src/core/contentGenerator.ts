/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  GoogleGenAI,
} from '@google/genai';
import { createCodeAssistContentGenerator } from '../code_assist/codeAssist.js';
import { DEFAULT_GEMINI_MODEL } from '../config/models.js';
import { getEffectiveModel } from './modelCheck.js';
import { OpenAI } from 'openai';

/**
 * Interface abstracting the core functionalities for generating content and counting tokens.
 */
export interface ContentGenerator {
  generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse>;

  generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>>;

  countTokens(request: CountTokensParameters): Promise<CountTokensResponse>;

  embedContent(request: EmbedContentParameters): Promise<EmbedContentResponse>;
}

export enum AuthType {
  LOGIN_WITH_GOOGLE = 'oauth-personal',
  USE_GEMINI = 'gemini-api-key',
  USE_VERTEX_AI = 'vertex-ai',
  USE_OPENAI = 'openai',
}

export type ContentGeneratorConfig = {
  model: string;
  apiKey?: string;
  vertexai?: boolean;
  authType?: AuthType | undefined;
};

export async function createContentGeneratorConfig(
  model: string | undefined,
  authType: AuthType | undefined,
  config?: { getModel?: () => string },
): Promise<ContentGeneratorConfig> {
  const geminiApiKey = process.env.GEMINI_API_KEY;
  const googleApiKey = process.env.GOOGLE_API_KEY;
  const googleCloudProject = process.env.GOOGLE_CLOUD_PROJECT;
  const googleCloudLocation = process.env.GOOGLE_CLOUD_LOCATION;
  const openaiApiKey = process.env.OPENAI_API_KEY;

  // Use runtime model from config if available, otherwise fallback to parameter or default
  const effectiveModel = config?.getModel?.() || model || DEFAULT_GEMINI_MODEL;

  const contentGeneratorConfig: ContentGeneratorConfig = {
    model: effectiveModel,
    authType,
  };

  // if we are using google auth nothing else to validate for now
  if (authType === AuthType.LOGIN_WITH_GOOGLE) {
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_GEMINI && geminiApiKey) {
    contentGeneratorConfig.apiKey = geminiApiKey;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );
    return contentGeneratorConfig;
  }

  if (
    authType === AuthType.USE_VERTEX_AI &&
    !!googleApiKey &&
    googleCloudProject &&
    googleCloudLocation
  ) {
    contentGeneratorConfig.apiKey = googleApiKey;
    contentGeneratorConfig.vertexai = true;
    contentGeneratorConfig.model = await getEffectiveModel(
      contentGeneratorConfig.apiKey,
      contentGeneratorConfig.model,
    );
    return contentGeneratorConfig;
  }

  if (authType === AuthType.USE_OPENAI && openaiApiKey) {
    contentGeneratorConfig.apiKey = openaiApiKey;
    // Model is set above, can be overridden by config
    return contentGeneratorConfig;
  }

  return contentGeneratorConfig;
}

export async function createContentGenerator(
  config: ContentGeneratorConfig,
  sessionId?: string,
): Promise<ContentGenerator> {
  const version = process.env.CLI_VERSION || process.version;
  const httpOptions = {
    headers: {
      'User-Agent': `GeminiCLI/${version} (${process.platform}; ${process.arch})`,
    },
  };
  if (config.authType === AuthType.LOGIN_WITH_GOOGLE) {
    return createCodeAssistContentGenerator(
      httpOptions,
      config.authType,
      sessionId,
    );
  }

  if (
    config.authType === AuthType.USE_GEMINI ||
    config.authType === AuthType.USE_VERTEX_AI
  ) {
    const googleGenAI = new GoogleGenAI({
      apiKey: config.apiKey === '' ? undefined : config.apiKey,
      vertexai: config.vertexai,
      httpOptions,
    });
    return googleGenAI.models;
  }

  if (config.authType === AuthType.USE_OPENAI && config.apiKey) {
    const openai = new OpenAI({ apiKey: config.apiKey });
    // Return a ContentGenerator-compatible wrapper for OpenAI
    return {
      async generateContent(request) {
        // Only support basic text prompt for now
        let prompt = '';
        if (typeof request.contents === 'string') {
          prompt = request.contents;
        } else if (Array.isArray(request.contents)) {
          // Try to extract text from each content object
          prompt = request.contents.map((c) => {
            if (typeof c === 'string') return c;
            if ('text' in c) return (c.text ?? '');
            if ('parts' in c && Array.isArray(c.parts)) {
              return c.parts.map((p) => (typeof p === 'string' ? p : (p.text ?? ''))).join('\n');
            }
            return '';
          }).join('\n');
        } else if (request.contents && typeof request.contents === 'object' && 'text' in request.contents) {
          prompt = (request.contents as any).text ?? '';
        }
        const completion = await openai.chat.completions.create({
          model: config.model || 'o4-mini',
          messages: [{ role: 'user', content: prompt }],
        });
        // Return a Gemini-like response structure
        return {
          candidates: [
            {
              content: {
                parts: [{ text: completion.choices[0].message.content }],
              },
            },
          ],
        } as GenerateContentResponse;
      },
      async generateContentStream(request) {
        throw new Error('OpenAI streaming not implemented');
      },
      async countTokens(request) {
        throw new Error('OpenAI token counting not implemented');
      },
      async embedContent(request) {
        throw new Error('OpenAI embedding not implemented');
      },
    };
  }

  throw new Error(
    `Error creating contentGenerator: Unsupported authType: ${config.authType}`,
  );
}
