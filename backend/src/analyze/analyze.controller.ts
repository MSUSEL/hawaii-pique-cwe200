import {
  Controller,
  Get,
  Post,
  Body,
  BadRequestException,
} 
from '@nestjs/common';
import { AnalyzeService } from 'src/analyze/analyze.service';
import { AnalyzeRequestDto } from 'src/types/analysis-config.type';


@Controller('analyze')
export class AnalyzeController {
  constructor(private readonly analyzeService: AnalyzeService) { }


  /**
   * POST endpoint
   * Run analysis on a project
   * Body: {
   *   "project": "project_name", 
   *   "language": "java|python|javascript", 
   *   ?"javaVersion": "[8, 11, 17, 21]", 
   *   ?"extension": "[csv, json, sarif]"
   * }
   *
   * @param analyzeDto Data transfer object with project name, language, and optional parameters
   */
  @Post()
  async runAnalysis(@Body() analyzeDto: AnalyzeRequestDto) {
    analyzeDto.language = 'java';
    try {
      // Validate required fields
      if (!analyzeDto.project) {
        throw new BadRequestException('Project name is required');
      }
      
      if (!analyzeDto.language) {
        throw new BadRequestException('Language is required. Supported languages: java');
      }

      return await this.analyzeService.runAnalysis(analyzeDto);
    } catch (error) {
      console.log('Project analysis stopped:', error.message);
      return { 'error': error.message };
    }
  }

  /**
   * GET endpoint
   * Get list of supported programming languages
   * @returns Array of supported language names
   */
  @Get('languages')
  async getSupportedLanguages() {
    return await this.analyzeService.getSupportedLanguages();
  }

}

