import { Module } from '@nestjs/common';
import { ChatGptService } from './chat-gpt.service';
import { ChatGptController } from './chat-gpt.controller';

@Module({
    controllers: [ChatGptController],
    providers: [ChatGptService],
    exports:[ChatGptService]
})
export class ChatGptModule {}
