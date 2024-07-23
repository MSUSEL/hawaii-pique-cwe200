abstract class Parser {
    extract(file: any): any[] {
        let results: any[] = [];
        let uniqueEntries = new Set<string>();

        for (let key in file) {
            if (Array.isArray(file[key])) {
                file[key].forEach((entry: any) => {
                    const uniqueKey = JSON.stringify(entry);
                    if (!uniqueEntries.has(uniqueKey)) {
                        uniqueEntries.add(uniqueKey);
                        results.push(entry);
                    }
                });
            }
        }

        return results;
    }

    getNamesAsList(file: any): any[] {
        let names: any[] = [];

        for (let key in file) {
            if (Array.isArray(file[key])) {
                file[key].forEach((entry: any) => {
                    if (entry.name) {
                        names.push(entry.name);
                    }
                });
            }
        }

        return Array.from(new Set(names)); // Ensure no duplicates
    }    

    saveToJSON(json_output, fileName: string, type: string, file): any {
        const sensitiveData = this.extract(file);
        if (json_output[fileName]) {
            json_output[fileName][type] = Array.from(new Set(json_output[fileName][type].concat(sensitiveData)));
        } else {
            json_output[fileName] = {
                fileName: fileName,
                variables: [],
                strings: [],
                comments: [],
                sinks: [],
            };
            json_output[fileName][type] = (sensitiveData);
        }
    }

    
    saveToMapping(mapping, fileName, file){       
        const names = this.getNamesAsList(file)
        if (mapping[fileName]) {
            mapping[fileName] = mapping[fileName].concat(names);
        } else {
            mapping[fileName] = names;
        }
    }

}


export class VariableParser extends Parser {}

export class StringParser extends Parser {}

export class CommentParser extends Parser {}

export class SinkParser extends Parser {

    getTypesAsList(file: any): any[] {
        let types: any[] = [];

        for (let key in file) {
            if (Array.isArray(file[key])) {
                file[key].forEach((entry: any) => {
                    if (entry.type) {
                        types.push(entry.type);
                    }
                });
            }
        }
        return types;

    }

    saveToMapping(mapping, fileName, file){
        const names = this.getNamesAsList(file);
        const types = this.getTypesAsList(file);
        let values: string[][] = names.map((name, index) => [name, types[index]]);

        if (mapping.has(fileName)) {
            mapping.set(fileName, mapping.get(fileName)!.concat(values));
        } else {
            mapping.set(fileName, values);
        }
    }
}

