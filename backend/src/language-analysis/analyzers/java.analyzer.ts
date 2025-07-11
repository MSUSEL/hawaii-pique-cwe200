import { Injectable } from '@nestjs/common';
import { ILanguageAnalyzer } from '../interfaces/language-analyzer.interface';
import { FileUtilService } from '../../files/fileUtilService';
import { JavaParserService } from '../../java-parser/java-parser.service';
import { CodeQlService } from '../../code-ql/code-ql.service';
import * as path from 'path';

/**
 * Java-specific implementation of the language analyzer interface.
 * Encapsulates all Java-specific analysis logic.
 */
@Injectable()
export class JavaAnalyzer implements ILanguageAnalyzer {
  
  constructor(
    private fileUtilService: FileUtilService,
    private javaParserService: JavaParserService,
    private codeqlService: CodeQlService,
  ) {}

  getSupportedLanguage(): string {
    return 'java';
  }

  getCodeQLLanguage(): string {
    return 'java';
  }

  getFileExtensions(): string[] {
    return ['.java'];
  }

  async discoverSourceFiles(sourcePath: string): Promise<string[]> {
    return await this.fileUtilService.getJavaFilesInDirectory(sourcePath);
  }

  async parseSourceFiles(filePaths: string[], sourcePath: string): Promise<any> {
    return await this.javaParserService.wrapper(filePaths, sourcePath);
  }

  async createCodeQLDatabase(sourcePath: string, config: any): Promise<void> {
    // Set Java version if specified
    if (config.languageVersion || config.javaVersion) {
      const javaVersion = config.languageVersion || config.javaVersion;
      try {
        this.fileUtilService.setJavaVersion(javaVersion);
      } catch (error) {
        console.error(`Failed to set Java version ${javaVersion}:`, error);
      }

    // Use the existing CodeQL service for Java database creation
    await this.codeqlService.createDatabase(sourcePath, config);
    }
  }

  async runCWEQueries(sourcePath: string, config: any): Promise<void> {
    // Use the existing CodeQL service for Java CWE queries
    await this.codeqlService.runCWEQueries(sourcePath, config);
  }

  async validateProject(sourcePath: string): Promise<boolean> {
    try {
      const javaFiles = await this.discoverSourceFiles(sourcePath);
      return javaFiles.length > 0;
    } catch (error) {
      return false;
    }
  }
}
