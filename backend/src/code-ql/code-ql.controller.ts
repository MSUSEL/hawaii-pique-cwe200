import {
  Controller,
  Get,
  Post,
  Body,
  Query
} 
from '@nestjs/common';
import { CodeQlService } from './code-ql.service';


@Controller('codeql')
export class CodeQlController {
  constructor(private readonly codeQlService: CodeQlService) { }

  /**
   * This endpoint is used to get the SARIF results for a specific project that has already been analyzed.
   * @param project - The name of the project to get SARIF results for
   * @returns - The location of the detected vulnerabilities in the project
   */

  @Get()
  async getSarifResults(@Query('project') project: string) {
    return await this.codeQlService.getSarifResults(project);
  }

  /**
   * This endpoint is used to get every flow node for a specific detection along a flow path.
    * @param vulnerabilityId - The ID of the vulnerability to get flow nodes for
    * @param project - The name of the project to get flow nodes for
    * @param index - The index of the flow node to get
    * @returns - The flow nodes for the specified vulnerability
   */
  @Get('vulnerabilityTree')
  async getDataFlowTree(
    @Query('vulnerabilityId') vulnerabilityId: string,
    @Query('project') project: string,
    @Query('index') index: string

  ) {
    return await this.codeQlService.getDataFlowTree(vulnerabilityId, project, index);
  }

  /**
   * This endpoint is used to make labeling and creating the flow verification dataset easier.
   * It takes the label data from the frontend and submits it to the backend so that is can be saved.
   * @param labelData - The label data to be submitted
   * @returns 
   */
  @Post('flow-labels')
  async submitFlowLabels(@Body() labelData: any) {
    return await this.codeQlService.labelFlows(labelData);
  }
}

