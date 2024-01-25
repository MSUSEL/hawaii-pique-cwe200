import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
@Injectable()
export class ChatGptService {
    openai: OpenAI.OpenAIApi = null;
    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileService: FileUtilService,
    ) {
        var api_key = this.configService.get('API_KEY');
        var configuration = new OpenAI.Configuration({
            apiKey: api_key,
        });
        this.openai = new OpenAI.OpenAIApi(configuration);
    }

    async openAiGetSensitiveVariables(files: string[]) {
        var variables = [];
        var fileList: any[] = [];

        // New Code for preprocessing
        let batchSize = 10;
        for (let i = 0; i < files.length; i += batchSize) {
            const batch = files.slice(i, i + batchSize);
            let processedFiles = await this.fileService.preprocessFiles(batch).catch((e) => console.error(e));
            if (typeof processedFiles === 'string') {
                // TODO: Update prompt to account for containing multiple files in each batch
                var response = await this.createGpt(processedFiles);
                // TODO: Create a way to split the response into individual responses
                // let individualResponses = this.splitBatchResponse(batchResponse, batch.length);

                // batch.forEach((file, index) => {
                //     let responseForFile = individualResponses[index];
                //     fileList.push({
                //         key: file,
                //         value: responseForFile
                //     });
                //     this.eventsGateway.emitDataToClients("data", file + ":");
                //     this.eventsGateway.emitDataToClients("data", responseForFile);
    
                //     var fileVariablesList = this.extractVariableNames(responseForFile);
                //     variables = variables.concat(fileVariablesList);
                // });

            }
        }
        
        // Previous Code
        for (var file of files) {
            var fileContents = await this.fileService.readFileAsync(file);
            var response = await this.createGpt(fileContents);
            fileList.push({
                key: file,
                value: response.message
            })
            this.eventsGateway.emitDataToClients("data", file + ":");
            this.eventsGateway.emitDataToClients("data", response.message)
            var fileVariablesList = this.extractVariableNames(response.message);
            variables = variables.concat(fileVariablesList);
        }
        return { variables, fileList };
    }

    async createDavinci(fileContents: string) {
        var prompt = this.createQuery(fileContents);
        var response = await this.createDavinciCompletion(prompt);
        this.extractVariableNames(response.message)
        return response;
    }

    async createGpt(fileContents: string) {
        var prompt = this.createQuery(fileContents);
        var response = await this.createGptFourCompletion(prompt);
        this.extractVariableNames(response.message);
        return response;

    }

    createQuery(code: string) {
        const message = `You are a security analyst tasked with identifying sensitive variables related to system configurations, database connections, and credentials, which could potentially have security issues in a given source code. Your goal is to identify variables that, if exposed to external users or hackers, could lead to security vulnerabilities or breaches. Please analyze the provided source code and list down any sensitive variables related to system configurations, database connections, or credentials that fit the criteria mentioned above. Please provide the names of the sensitive variables only, without disclosing any specific values. Your analysis will help in securing the application and preventing potential data leaks or unauthorized access.please I want your response in json format like 
        {
            "sensitiveVariables": [
              {
                "name": "variableName1",
                "description": "variableDescription"
              },
              {
                "name": "variableName2",
                "description": "variableDescription"
              },
              {
                "name": "variableNameN",
                "description": "variableDescription"
              }
            ]
        }


        // New Prompt
        // You are a security analyst tasked with identifying sensitive variables related to system configurations, database connections, and credentials in multiple source code files. Your goal is to identify variables that, if exposed to external users or hackers, could lead to security vulnerabilities or breaches. Please analyze the provided source code files and list down any sensitive variables related to system configurations, database connections, or credentials that fit the criteria mentioned above for each file separately. The beginning of each file is marked by "-----BEGIN FILE: [FileName]-----", and the end is marked by "-----END FILE: [FileName]-----". Please provide the names of the sensitive variables only, without disclosing any specific values, and format your response in JSON. Your analysis will help in securing the application and preventing potential data leaks or unauthorized access.

        // Please structure your response in the following JSON format for each file:

        // {
        // "files": [
        //     {
        //     "fileName": "FileName1.java",
        //     "sensitiveVariables": [
        //         {
        //         "name": "variableName1",
        //         "description": "variableDescription1"
        //         },
        //         {
        //         "name": "variableName2",
        //         "description": "variableDescription2"
        //         }
        //     ]
        //     },
        //     {
        //         "fileName": "FileName2.java",
        //         "sensitiveVariables": [
        //         {
        //             "name": "variableName1",
        //             "description": "variableDescription1"
        //         },
        //         {
        //             "name": "variableName2",
        //             "description": "variableDescription2"
        //         }
        //         ]
        //     }
        // ]
        // }

        ${code}`;
        return message;
    }

    async createDavinciCompletion(prompt: string) {
        const completion = await this.openai.createCompletion({
            model: 'text-davinci-003',
            prompt: prompt,
            temperature: 0.1,
            max_tokens: 250,
        });
        return { message: completion?.data.choices?.[0]?.text };
    }

    async createGptCompletion(prompt: string) {
        var completion = await this.openai.createChatCompletion({
            model: 'gpt-3.5-turbo',
            temperature: 0.2,
            messages: [{ role: 'user', content: prompt }],
        });
        return { message: completion.data.choices[0].message.content };
    }

    async createGptFourCompletion(prompt: string) {
        var completion = await this.openai.createChatCompletion({
            model: 'gpt-4',
            temperature: 0.2,
            messages: [{ role: 'user', content: prompt }],
        });
        console.log(completion.data.choices[0].message.content)
        return { message: completion.data.choices[0].message.content };
    }


    extractVariableNames(text: string): string[] {
        var variables = [];
        try {
            var json = JSON.parse(text)
            variables = json["sensitiveVariables"].map(variable => `\"${variable.name}\"`);
            return variables;

        } catch (e) {
            console.log(text);
            return variables;
        }
    }


    async getFileGptResponse(filePath: String) {
        var directories = filePath.split("\\");
        var jsonFilePath = path.join(directories[0], directories[1], "data.json");
        try{
            var jsonArray = await this.fileService.readJsonFile(jsonFilePath);
            for (const obj of jsonArray) {
                if (obj.key === filePath) {
                    return obj;
                }
            }
        }catch(e){
            return {value: "no data found for this file yet. please re run your project to get the results"}
        }

    }
}
