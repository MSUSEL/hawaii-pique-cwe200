import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
} from '@nestjs/common';
import { ChatGptService } from './chat-gpt.service';

@Controller('chatgpt')
export class ChatGptController {
    constructor(private readonly chatGptService: ChatGptService) {}
    @Post()
    async chatGptQuery(@Body() createChatGptDto: any) {
        return await this.chatGptService.getFileGptResponse(createChatGptDto.filePath)
    }
}
