import { Global, Module } from '@nestjs/common';
import { JavaParserService } from './java-parser.service';
import { JavaParserController } from './java-parser.controller';
import { EventsModule } from 'src/events/events.module';
@Global()
@Module({
    imports:[EventsModule],
    controllers: [JavaParserController],
    providers: [JavaParserService],
    exports: [JavaParserService],
})
export class JavaParserModule {}
