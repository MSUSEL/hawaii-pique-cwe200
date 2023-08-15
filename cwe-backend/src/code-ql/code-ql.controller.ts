import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
    UseInterceptors
} from '@nestjs/common';
import { CodeQlService } from './code-ql.service';


@Controller('codeql')
export class CodeQlController {
    constructor(private readonly codeQlService: CodeQlService) {}

    @Post()
    async runCodeQl(@Body() createCodeQlDto: any) {
        return await this.codeQlService.runCodeQl(createCodeQlDto);
    }

    @Get()
    async getSarifResults(){
        return {}
    }
}
