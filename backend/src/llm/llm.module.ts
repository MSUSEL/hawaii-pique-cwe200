import { Module } from '@nestjs/common';
import { EventsModule } from 'src/events/events.module';
import { LLMService } from './llm.service';
import { LLMController } from './llm.controller';


@Module({
    imports:[EventsModule],
    controllers: [LLMController],
    providers: [LLMService],
    exports:[LLMService]
})
export class LLMModule {}
