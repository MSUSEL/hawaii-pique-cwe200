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
import { AnalyzeService } from 'src/analyze/analyze.service';


@Controller('analyze')
export class AnalyzeController {
  constructor(private readonly analyzeService: AnalyzeService) { }


  /**
   * POST endpoint
   * Run codeql query against a target project
   * Body: {"project":"project_name"}
   *
   * @param createAnalyzeDto Data transfer object with project name
   */
  @Post()
  async runAnalysis(@Body() createAnalyzeDto: any) {
    try {
      return await this.analyzeService.runAnalysis(createAnalyzeDto);
    }

    catch (error) {
      console.log('Project analysis stopped')
      return { 'error': error.message }
    }
  }

  // @Get()
  // async getSarifResults(@Query('project') project: string) {
  //   return await this.analyzeService.getSarifResults(project);
  // }

  // @Get('vulnerabilityTree')
  // async getDataFlowTree(
  //   @Query('vulnerabilityId') vulnerabilityId: string,
  //   @Query('project') project: string,
  //   @Query('index') index: string

  // ) {

  //   return await this.analyzeService.getDataFlowTree(vulnerabilityId, project, index);
  // }

  // @Post('flow-labels')
  // async submitFlowLabels(@Body() labelData: any) {
  //   console.log('Test')
  //   return await this.analyzeService.labelFlows(labelData);
  // }
}

