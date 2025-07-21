import { Injectable } from '@nestjs/common';
import { ILanguageAnalyzer } from '../interfaces/language-analyzer.interface';
import { FileUtilService } from '../../files/fileUtilService';
// import { JavaParserService } from '../../parsers/implementations/java-parser.service';
import { CodeQlService } from '../../code-ql/code-ql.service';
import * as path from 'path';


/**
 * C++-specific implementation of the language analyzer interface.
 * Encapsulates all C++-specific analysis logic.
 */
// @Injectable()
// export class CppAnalyzer implements ILanguageAnalyzer {

//     constructor(
//         private fileUtilService: FileUtilService,
//         private codeQlService: CodeQlService,
//     ) {}
//     getSupportedLanguage(): string {
//         return 'cpp';
//     }
//     getFileExtensions(): string[] {
//         return ['.cpp', '.h', '.hpp'];
//     }

// }