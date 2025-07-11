import { Module } from '@nestjs/common';
import { AnalyzeService } from 'src/analyze/analyze.service';
import { AnalyzeController } from 'src/analyze/analyze.controller';
import { FilesModule } from 'src/files/files.module';
import { EventsModule } from 'src/events/events.module';
import { BertModule } from 'src/bert/bert.module';
import { LLMModule } from 'src/llm/llm.module';
import { CodeQlModule } from 'src/code-ql/code-ql.module';
import { LanguageAnalysisModule } from 'src/language-analysis/language-analysis.module';


@Module({
  imports: [FilesModule, EventsModule, BertModule, LLMModule, CodeQlModule, LanguageAnalysisModule],
  controllers: [AnalyzeController],
  providers: [AnalyzeService], 
  exports: [AnalyzeService], 
})
export class AnalyzeModule{}
