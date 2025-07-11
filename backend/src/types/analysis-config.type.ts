/**
 * DTO for the analyze endpoint request
 */
export interface AnalyzeRequestDto {
  /** Name of the project to analyze */
  project: string;
  
  /** Programming language to analyze */
  language: string;
  
  /** Optional output format */
  extension?: 'csv' | 'json' | 'sarif';
  
  /** Optional language version (e.g., javaVersion for Java projects) */
  javaVersion?: number;
  
  /** Optional Python version for Python projects */
  pythonVersion?: string;
  
  /** Optional Node.js version for JavaScript projects */
  nodeVersion?: string;
}
