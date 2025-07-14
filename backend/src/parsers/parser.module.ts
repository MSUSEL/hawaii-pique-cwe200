import { Global, Module } from '@nestjs/common';
import { JavaParserService } from './implementations/java-parser.service';
import { JavaParserController } from './parser.controller';
import { EventsModule } from 'src/events/events.module';
@Global()
@Module({
    imports:[EventsModule],
    controllers: [JavaParserController],
    providers: [JavaParserService],
    exports: [JavaParserService],
})
export class ParserModule {}
