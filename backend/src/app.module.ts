import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { BertModule } from './bert/bert.module';
import { FilesModule } from './files/files.module';
import { ChatGptModule } from './chat-gpt/chat-gpt.module';
import { ConfigModule } from '@nestjs/config';
import { CodeQlModule } from './code-ql/code-ql.module';
import { EventsGateway } from './events/events.gateway';
import { EventsModule } from './events/events.module';
import { LLMModule } from './llm/llm.module';
import { JavaParserModule } from './java-parser/java-parser.module';
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
        JavaParserModule,

    ],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule {}
