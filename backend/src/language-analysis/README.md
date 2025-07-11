# Multi-Language Analysis Pipeline

This refactoring introduces support for multiple programming languages in the analysis pipeline while following SOLID principles.

## Architecture Overview

The new architecture uses:
- **Interface Segregation Principle**: `ILanguageAnalyzer` interface defines language-specific operations
- **Open/Closed Principle**: New languages can be added without modifying existing code
- **Single Responsibility Principle**: Each analyzer handles one language
- **Factory Pattern**: `LanguageAnalyzerFactory` creates appropriate analyzers
- **Dependency Injection**: NestJS handles service dependencies

## New API Usage

### Updated Endpoint

**POST** `/analyze`

**Request Body:**
```json
{
  "project": "project_name",
  "language": "java",
  "javaVersion": 17,
  "extension": "sarif"
}
```

**Required Fields:**
- `project`: Name of the project to analyze
- `language`: Programming language (`java`, `python`, `javascript` - currently only `java` is fully implemented)

**Optional Fields:**
- `extension`: Output format (`csv`, `json`, `sarif`)
- `javaVersion`: Java version (8, 11, 17, 21) - for Java projects
- `pythonVersion`: Python version - for Python projects (placeholder)
- `nodeVersion`: Node.js version - for JavaScript projects (placeholder)

## Architecture Components

### 1. Language Analyzer Interface (`ILanguageAnalyzer`)
Defines standard operations every language analyzer must implement:
- `discoverSourceFiles()`: Find source files for the language
- `parseSourceFiles()`: Parse files and extract elements
- `createCodeQLDatabase()`: Create language-specific CodeQL database
- `runCWEQueries()`: Execute language-specific security queries
- `validateProject()`: Check if project contains valid source files

### 2. Language Analyzers
- **JavaAnalyzer**: Implements Java-specific analysis (fully functional)
- **PythonAnalyzer**: Placeholder for Python support
- **JavaScriptAnalyzer**: Placeholder for JavaScript support

### 3. Language Analyzer Factory
Creates appropriate analyzer instances based on the requested language.

### 4. Updated Analyze Service
Now delegates language-specific operations to the appropriate analyzer while handling common operations (BERT analysis, flow verification) centrally.

## Analysis Pipeline

1. **Language Validation**: Factory validates if the requested language is supported
2. **Project Validation**: Language analyzer checks if project contains valid source files
3. **File Discovery**: Language-specific file discovery (e.g., `.java` files for Java)
4. **Parsing**: Language-specific parsing using appropriate parser
5. **Attack Surface Detection**: Common BERT-based analysis
6. **CodeQL Database**: Language-specific database creation
7. **Security Analysis**: Language-specific CWE queries
8. **Flow Verification**: Common flow verification using BERT
9. **Results**: Return in requested format

## Adding New Languages

To add a new language (e.g., Python):

1. **Create Language Analyzer** (`python.analyzer.ts`):
```typescript
@Injectable()
export class PythonAnalyzer implements ILanguageAnalyzer {
  getSupportedLanguage() { return 'python'; }
  getCodeQLLanguage() { return 'python'; }
  getFileExtensions() { return ['.py']; }
  
  async discoverSourceFiles(sourcePath: string) {
    // Implement Python file discovery
  }
  
  async parseSourceFiles(filePaths: string[], sourcePath: string) {
    // Implement Python parsing (AST, etc.)
  }
  
  async createCodeQLDatabase(sourcePath: string, config: any) {
    // Implement CodeQL database creation for Python
  }
  
  async runCWEQueries(sourcePath: string, config: any) {
    // Implement Python-specific CWE queries
  }
  
  async validateProject(sourcePath: string) {
    // Check for Python files
  }
}
```

2. **Register in Factory** (`language-analyzer.factory.ts`):
```typescript
constructor(
  private javaAnalyzer: JavaAnalyzer,
  private pythonAnalyzer: PythonAnalyzer, // Add this
) {}

private registerAnalyzers() {
  this.analyzers.set('java', this.javaAnalyzer);
  this.analyzers.set('python', this.pythonAnalyzer); // Add this
}
```

3. **Update Module** (`language-analysis.module.ts`):
```typescript
providers: [
  LanguageAnalyzerFactory,
  JavaAnalyzer,
  PythonAnalyzer, // Add this
],
```

## Benefits

1. **Extensibility**: Easy to add new languages without modifying existing code
2. **Maintainability**: Language-specific logic is isolated in separate analyzers
3. **Testability**: Each analyzer can be tested independently
4. **Type Safety**: Strong typing with TypeScript interfaces
5. **Scalability**: Factory pattern allows dynamic language support

## Current Implementation Status

- ✅ **Java**: Fully implemented and functional
- 🚧 **Python**: Interface ready, implementation needed
- 🚧 **JavaScript**: Interface ready, implementation needed
- 🚧 **TypeScript**: Interface ready, implementation needed

The Java analyzer uses the existing Java-specific services (JavaParserService, CodeQlService) while providing a clean interface for future language extensions.
