/**
 * Interface for language-specific analyzers following the Interface Segregation Principle.
 * Each language analyzer must implement all methods for the complete analysis pipeline.
 */
export interface ILanguageAnalyzer {
  /**
   * Get the supported language for this analyzer
   */
  getSupportedLanguage(): string;

  /**
   * Discover source files for the specific language in a directory
   * @param sourcePath - Path to the source directory
   * @returns Array of file paths for the specific language
   */
  discoverSourceFiles(sourcePath: string): Promise<string[]>;

  /**
   * Parse source files and extract language-specific elements
   * @param filePaths - Array of file paths to parse
   * @param sourcePath - Root source directory path
   * @returns Parsed results containing variables, strings, comments, etc.
   */
  parseSourceFiles(filePaths: string[], sourcePath: string): Promise<any>;

  /**
   * Create a CodeQL database for the specific language
   * @param sourcePath - Path to the source directory
   * @param config - Analysis configuration
   */
  createCodeQLDatabase(sourcePath: string, config: any): Promise<void>;

  /**
   * Run language-specific CWE queries
   * @param sourcePath - Path to the source directory
   * @param config - Analysis configuration
   */
  runCWEQueries(sourcePath: string, config: any): Promise<void>;

  /**
   * Get the CodeQL language identifier for database creation
   */
  getCodeQLLanguage(): string;

  /**
   * Get file extensions for the specific language
   */
  getFileExtensions(): string[];

  /**
   * Validate if the project contains valid source files for this language
   * @param sourcePath - Path to the source directory
   */
  validateProject(sourcePath: string): Promise<boolean>;
}
