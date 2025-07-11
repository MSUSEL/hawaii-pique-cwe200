import { Injectable } from '@nestjs/common';
import { ILanguageAnalyzer } from '../interfaces/language-analyzer.interface';
import { JavaAnalyzer } from '../analyzers/java.analyzer';

/**
 * Factory for creating language-specific analyzers.
 * Follows the Factory Pattern and Open/Closed Principle.
 * New languages can be added without modifying existing code.
 */
@Injectable()
export class LanguageAnalyzerFactory {
  private analyzers: Map<string, ILanguageAnalyzer> = new Map();

  constructor(
    private javaAnalyzer: JavaAnalyzer,
    // Will inject other language analyzers here as they're implemented
    // private pythonAnalyzer: PythonAnalyzer,
    // private javascriptAnalyzer: JavaScriptAnalyzer,
  ) {
    this.registerAnalyzers();
  }

  /**
   * Register all available language analyzers
   */
  private registerAnalyzers(): void {
    this.analyzers.set('java', this.javaAnalyzer);
    // Add other languages here:
    // this.analyzers.set('python', this.pythonAnalyzer);
    // this.analyzers.set('javascript', this.javascriptAnalyzer);
    // this.analyzers.set('typescript', this.typescriptAnalyzer);
  }

  /**
   * Get an analyzer for the specified language
   * @param language - The programming language
   * @returns Language-specific analyzer
   * @throws Error if language is not supported
   */
  getAnalyzer(language: string): ILanguageAnalyzer {
    const normalizedLanguage = language.toLowerCase();
    const analyzer = this.analyzers.get(normalizedLanguage);
    
    if (!analyzer) {
      throw new Error(`Unsupported language: ${language}. Supported languages: ${this.getSupportedLanguages().join(', ')}`);
    }
    
    return analyzer;
  }

  /**
   * Get list of all supported languages
   */
  getSupportedLanguages(): string[] {
    return Array.from(this.analyzers.keys());
  }

  /**
   * Check if a language is supported
   */
  isLanguageSupported(language: string): boolean {
    return this.analyzers.has(language.toLowerCase());
  }
}
