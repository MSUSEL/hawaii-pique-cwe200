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
import { BertService } from './bert.service';

@Controller('chatgpt')
export class BertController {
    constructor(private readonly bertService: BertService) {}
}
