import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    Param,
    Delete,
    UseInterceptors,
    Query
} from '@nestjs/common';
import { CodeQlService } from './code-ql.service';


@Controller('codeql')
export class CodeQlController {
    constructor(private readonly codeQlService: CodeQlService) {}


    /**
     * POST endpoint
     * Run codeql query against a target project
     * Body: {"project":"project_name"}
     *
     * @param createCodeQlDto Data transfer object with project name
     */
    @Post()
    async runCodeQl(@Body() createCodeQlDto: any) {
        return await this.codeQlService.runCodeQl(createCodeQlDto);
    }

    @Get()
    async getSarifResults(@Query('project') project : string){
        return await this.codeQlService.getSarifResults(project);
    }
}
