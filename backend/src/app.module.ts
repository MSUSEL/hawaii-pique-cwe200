import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AnalyzeModule } from './analyze/analyze.module';
import { BertModule } from './bert/bert.module';
import { FilesModule } from './files/files.module';
import { ChatGptModule } from './chat-gpt/chat-gpt.module';
import { ConfigModule } from '@nestjs/config';
import { CodeQlModule } from './code-ql/code-ql.module';
import { EventsGateway } from './events/events.gateway';
import { EventsModule } from './events/events.module';
import { LLMModule } from './llm/llm.module';
import { ParserModule } from './parsers/parser.module';
@Module({
    imports: [
        ConfigModule.forRoot({
            envFilePath: '.env',
            isGlobal: true,
        }),
        FilesModule,
        EventsModule,
        ChatGptModule,
        CodeQlModule,
        BertModule,
        LLMModule,
        ParserModule,
        AnalyzeModule,

    ],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule {}
