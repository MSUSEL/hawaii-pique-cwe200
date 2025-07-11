import { Module } from '@nestjs/common';
import { LanguageAnalyzerFactory } from './factories/language-analyzer.factory';
import { JavaAnalyzer } from './analyzers/java.analyzer';
import { FilesModule } from '../files/files.module';
import { JavaParserModule } from '../java-parser/java-parser.module';
import { CodeQlModule } from '../code-ql/code-ql.module';

/**
 * Module for language-specific analysis components.
 * This module provides language analyzers and the factory to create them.
 */
@Module({
  imports: [
    FilesModule,
    JavaParserModule, 
    CodeQlModule,
  ],
  providers: [
    LanguageAnalyzerFactory,
    JavaAnalyzer,
  ],
  exports: [
    LanguageAnalyzerFactory,
    JavaAnalyzer,
  ],
})
export class LanguageAnalysisModule {}
