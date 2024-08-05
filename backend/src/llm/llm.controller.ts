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
import { LLMService } from './llm.service';

@Controller('llm')
export class LLMController {
    constructor(private readonly llmService: LLMService) {}
}
