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
        if(createChatGptDto.model=='gpt')
            return await this.chatGptService.createGpt(createChatGptDto.filecontents);
        else
            return await  this.chatGptService.createDavinci(createChatGptDto.filecontents); 
    }
}
