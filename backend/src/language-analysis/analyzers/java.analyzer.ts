import { Injectable } from '@nestjs/common';
import { ILanguageAnalyzer } from '../interfaces/language-analyzer.interface';
import { FileUtilService } from '../../files/fileUtilService';
import { JavaParserService } from '../../parsers/implementations/java-parser.service';
import { CodeQlService } from '../../code-ql/code-ql.service';
import * as path from 'path';

/**
 * Java-specific implementation of the language analyzer interface.
 * Encapsulates all Java-specific analysis logic.
 */
@Injectable()
export class JavaAnalyzer implements ILanguageAnalyzer {

  private javaVersion: number | null = null;

  constructor(
    private fileUtilService: FileUtilService,
    private javaParserService: JavaParserService,
    private codeqlService: CodeQlService,
  ) {}

  getSupportedLanguage(): string {
    return 'java';
  }

  setVersion(version: string): void {
    try{
      this.javaVersion = parseInt(version, 10);
    } catch (error) {
      console.error(`Failed to set Java version ${version}:`, error);
    }
  }

  getFileExtensions(): string[] {
    return ['.java'];
  }

  async discoverSourceFiles(sourcePath: string): Promise<string[]> {
    return await this.fileUtilService.getSourceFilesInDirectory(sourcePath, '.java');
  }

  async parseSourceFiles(filePaths: string[], sourcePath: string): Promise<any> {
    return await this.javaParserService.wrapper(filePaths, sourcePath);
  }

  async createCodeQLDatabase(sourcePath: string, config: any): Promise<void> {
    // Set Java version if specified
    
      // try {
      //   this.fileUtilService.setJavaVersion(this.javaVersion);
      // } catch (error) {
      //   console.error(`Failed to set Java version ${this.javaVersion}:`, error);
      // }

    // Use the existing CodeQL service for Java database creation
    await this.codeqlService.createDatabase(sourcePath, config, this.getSupportedLanguage());
    
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
