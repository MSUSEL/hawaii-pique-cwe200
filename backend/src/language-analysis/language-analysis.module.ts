import { Module } from '@nestjs/common';
import { LanguageAnalyzerFactory } from './factories/language-analyzer.factory';
import { JavaAnalyzer } from './analyzers/java.analyzer';
import { PythonAnalyzer } from './analyzers/python.analyzer';
import { FilesModule } from '../files/files.module';
import { ParserModule } from '../parsers/parser.module';
import { CodeQlModule } from '../code-ql/code-ql.module';

/**
 * Module for language-specific analysis components.
 * This module provides language analyzers and the factory to create them.
 */
@Module({
  imports: [
    FilesModule,
    ParserModule, 
    CodeQlModule,
  ],
  providers: [
    LanguageAnalyzerFactory,
    JavaAnalyzer,
    PythonAnalyzer,
  ],
  exports: [
    LanguageAnalyzerFactory,
    JavaAnalyzer,
    PythonAnalyzer,
  ],
})
export class LanguageAnalysisModule {}
