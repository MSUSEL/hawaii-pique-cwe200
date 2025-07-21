import { Injectable } from '@nestjs/common';
import { ILanguageAnalyzer } from '../interfaces/language-analyzer.interface';
import { FileUtilService } from '../../files/fileUtilService';
import { PythonParserService } from '../../parsers/implementations/python-parser.service';
import { CodeQlService } from '../../code-ql/code-ql.service';

/**
 * Python-specific implementation of the language analyzer interface.
 * Encapsulates all Python-specific analysis logic.
 */
@Injectable()
export class PythonAnalyzer implements ILanguageAnalyzer {

  constructor(
    private fileUtilService: FileUtilService,
    private pythonParserService: PythonParserService,
    private codeqlService: CodeQlService,
  ) {}

  getSupportedLanguage(): string {
    return 'python';
  }
  
  getFileExtensions(): string[] {   
    return ['.py'];
  }

  setVersion(version: string): void {
    // Python version handling can be added here if needed
    // Currently, Python does not require version-specific handling in this context
  }

  async discoverSourceFiles(sourcePath: string): Promise<string[]> {
    return await this.fileUtilService.getSourceFilesInDirectory(sourcePath, '.py');
  }

  async parseSourceFiles(filePaths: string[], sourcePath: string): Promise<any> {
    return await this.pythonParserService.wrapper(filePaths, sourcePath);
  }

  async createCodeQLDatabase(sourcePath: string, config: any): Promise<void> {
    // Use the existing CodeQL service for Python database creation
    await this.codeqlService.createDatabase(sourcePath, config, this.getSupportedLanguage());
    
  }

  async runCWEQueries(sourcePath: string, config: any): Promise<void> {
    // Use the existing CodeQL service for Python CWE queries
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
