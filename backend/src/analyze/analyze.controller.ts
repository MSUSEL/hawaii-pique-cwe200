import {
  Controller,
  Get,
  Post,
  Body,

} 
from '@nestjs/common';
import { AnalyzeService } from 'src/analyze/analyze.service';


@Controller('analyze')
export class AnalyzeController {
  constructor(private readonly analyzeService: AnalyzeService) { }


  /**
   * POST endpoint
   * Run analysis on a project
   * Body: {"project":"project_name", 
   * ?"javaVersion": "[8, 11, 17, 21]", 
   * ?"extension": "[csv, json]"}
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

}

