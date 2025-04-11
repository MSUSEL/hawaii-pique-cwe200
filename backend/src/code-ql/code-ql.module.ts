import { Module } from '@nestjs/common';
import { CodeQlService } from './code-ql.service';
import { CodeQlController } from './code-ql.controller';
import { FilesModule } from 'src/files/files.module';
import { CodeQlParserService } from './codql-parser-service';
import { EventsModule } from 'src/events/events.module';


@Module({
    imports:[EventsModule, FilesModule],
    controllers: [CodeQlController],
    providers: [CodeQlService, CodeQlParserService],
    exports: [CodeQlService, CodeQlParserService] 

})
export class CodeQlModule {}
