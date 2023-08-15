import { Module } from '@nestjs/common';
import { CodeQlService } from './code-ql.service';
import { CodeQlController } from './code-ql.controller';
import { FilesModule } from 'src/files/files.module';
import { CodeQlParserService } from './codql-parser-service';
import { ChatGptModule } from 'src/chat-gpt/chat-gpt.module';

@Module({
    imports:[ChatGptModule],
    controllers: [CodeQlController],
    providers: [CodeQlService,CodeQlParserService],

})
export class CodeQlModule {}
