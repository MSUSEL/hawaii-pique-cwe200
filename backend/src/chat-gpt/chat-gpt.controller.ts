import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
    Query
} from '@nestjs/common';
import { ChatGptService } from './chat-gpt.service';

@Controller('chatgpt')
export class ChatGptController {
    constructor(private readonly chatGptService: ChatGptService) {}
    @Post()
    async chatGptQuery(@Body() createChatGptDto: any) {
        return await this.chatGptService.getFileGptResponse(createChatGptDto.filePath)
    }

    @Get()
    async getCostEstimate(@Query('project') projectPath : string){
        return await this.chatGptService.getCostEstimate(projectPath);
    }

    @Get('token')
    async getChatGptToken(){
        return await this.chatGptService.getChatGptToken();
    }

    @Post('token')
    async updateChatGptToken(@Body() body: { token: string }) {
        return await this.chatGptService.updateChatGptToken(body.token);
    }
}
