import { Global, Module } from '@nestjs/common';
import { JavaParserService } from './implementations/java-parser.service';
import { JavaParserController } from './parser.controller';
import { EventsModule } from 'src/events/events.module';
import { PythonParserService } from './implementations/python-parser.service';
@Global()
@Module({
    imports:[EventsModule],
    controllers: [JavaParserController],
    providers: [JavaParserService, PythonParserService],
    exports: [JavaParserService, PythonParserService],
})
export class ParserModule {}
