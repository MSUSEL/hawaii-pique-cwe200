import { Module } from '@nestjs/common';
import { CodeQlService } from './code-ql.service';
import { CodeQlController } from './code-ql.controller';
import { FilesModule } from 'src/files/files.module';
import { CodeQlParserService } from './codql-parser-service';
import { ChatGptModule } from 'src/chat-gpt/chat-gpt.module';
import { EventsModule } from 'src/events/events.module';
import { BertModule } from 'src/bert/bert.module';
import { BertService } from 'src/bert/bert.service';
import { LLMModule } from 'src/llm/llm.module';
import { LLMService } from 'src/llm/llm.service';

@Module({
    imports:[ChatGptModule,EventsModule, BertModule, LLMModule],
    controllers: [CodeQlController],
    providers: [CodeQlService, CodeQlParserService, BertService, LLMService],

})
export class CodeQlModule {}
