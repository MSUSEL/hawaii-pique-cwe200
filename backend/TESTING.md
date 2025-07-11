# API Testing Examples

## Test the new multi-language analysis endpoint

### 1. Get supported languages
```bash
curl -X GET http://localhost:3000/analyze/languages
```

**Expected Response:**
```json
["java"]
```

### 2. Analyze a Java project (existing functionality)
```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "project": "your-java-project",
    "language": "java",
    "javaVersion": 17,
    "extension": "sarif"
  }'
```

### 3. Try an unsupported language (demonstrates validation)
```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "project": "some-project",
    "language": "python"
  }'
```

**Expected Response:**
```json
{
  "error": "Unsupported language: python. Supported languages: java"
}
```

### 4. Missing required fields (demonstrates validation)
```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "project": "some-project"
  }'
```

**Expected Response:**
```json
{
  "error": "Language is required. Supported languages: java"
}
```

## What Changed in the Analysis Flow

### Before (Java-only)
```
1. Hard-coded Java file discovery
2. Direct JavaParserService usage
3. Hard-coded CodeQL for Java
4. No language validation
```

### After (Multi-language ready)
```
1. LanguageAnalyzerFactory.getAnalyzer(language)
2. LanguageAnalyzer.discoverSourceFiles()
3. LanguageAnalyzer.parseSourceFiles() 
4. LanguageAnalyzer.createCodeQLDatabase()
5. LanguageAnalyzer.runCWEQueries()
6. Language validation at factory level
```

### Analysis Pipeline Comparison

#### Old Analyze Service Method:
- Directly called `this.fileUtilService.getJavaFilesInDirectory()`
- Directly called `this.javaParserService.wrapper()`
- Hard-coded Java version handling
- No language extensibility

#### New Analyze Service Method:
- Calls `languageAnalyzer.discoverSourceFiles()`
- Calls `languageAnalyzer.parseSourceFiles()`
- Language-agnostic configuration handling
- Easy to extend for new languages

## Adding Python Support (Example)

When Python support is added, the same API will work:

```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "project": "my-python-project",
    "language": "python",
    "pythonVersion": "3.9",
    "extension": "sarif"
  }'
```

The system will automatically:
1. Use PythonAnalyzer for file discovery (*.py files)
2. Use Python-specific parsing
3. Create CodeQL database with `--language=python`
4. Run Python-specific CWE queries
