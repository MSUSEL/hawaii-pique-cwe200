import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
@Injectable()
export class ChatGptService {
    openai: OpenAI.OpenAIApi = null;
    constructor(
        private configService: ConfigService,
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
        for (var file of files) {
            var fileContents = await this.fileService.readFileAsync(file);
            var response = await this.createGpt(fileContents);
            var fileVariablesList = this.extractVariableNames(response.message);
            variables = variables.concat(fileVariablesList);
        }
        console.log(variables);
        return variables;
    }

    async createDavinci(fileContents: string) {
        var prompt = this.createQuery(fileContents);
        var response = await this.createDavinciCompletion(prompt);
        //var response=await this.createGptCompletion(prompt);
        this.extractVariableNames(response.message)
        return response;
    }

    async createGpt(fileContents: string) {
        var prompt = this.createQuery(fileContents);
        var response= await this.createGptFourCompletion(prompt);
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
        console.log(completion.data.choices[0].message.content )
        return { message: completion.data.choices[0].message.content };
    }


    extractVariableNames(text:string): string[] {
        var variables=[];
        try{
            var json=JSON.parse(text)
            variables= json["sensitiveVariables"].map(variable => `\"${variable.name}\"`);
            return variables;

        }catch(e){
            console.log(text);
            return variables;
        }

    }
}
