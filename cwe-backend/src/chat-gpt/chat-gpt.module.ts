import { Module } from '@nestjs/common';
import { ChatGptService } from './chat-gpt.service';
import { ChatGptController } from './chat-gpt.controller';
import { EventsModule } from 'src/events/events.module';

@Module({
    imports:[EventsModule],
    controllers: [ChatGptController],
    providers: [ChatGptService],
    exports:[ChatGptService]
})
export class ChatGptModule {}
