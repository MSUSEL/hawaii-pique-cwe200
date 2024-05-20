"""
This script counts the number of tokens in a given project and estimates the cost of querying and generating the output from ChatGPT.
To add a new project to estimate, add it to the /backend/Files directory.
Currently, the script uses the GPT-4o pricing of $5 per 1 million input tokens and $15 per 1 million output tokens.
If there are changes to the pricing, update the INPUT_COST and OUTPUT_COST accordingly.

Since we don't know the exact number of tokens in the output, we estimate it to be 30% of the input tokens.
This is based on the average output tokens we have seen in the projects we have run so far. 
Feel free to update this estimate if have run more projects and have a better estimate.
This part is done in the estimate_output_tokens() function.

"""

import re
import tiktoken
import os
import glob
import json
import math

# Ref https://openai.com/api/pricing/ 
INPUT_COST = (5 / 1_000_000) # GPT-4o pricing is $5 per 1 million input tokens
OUTPUT_COST = (15 / 1_000_000) # GPT-4o pricing is $15 per 1 million output tokens
results = {}


def main():
    base_directory = 'backend/Files'
    # base_directory = '../projects_to_test'
    projects = get_directories_in(base_directory)
    print(f'The costs are based on the GPT-4o pricing of $5 per 1 million tokens.')
    print(f'Proprocesing includes removing unnecessary whitespaces and batching files to fit the 10,000 token limit\n')
    for project in projects:
        print(f'Results for {project}')
        directory = os.path.join(base_directory, project)
        # os.chdir(directory)
        run_test(directory)
        print("\n")


def run_test(directory):
    #     Get file names
    files = find_java_files(directory)
    print(f' This project has {len(files)} Java files')

    token_count = 0
    token_count_preprocessed = 0

    # Count Tokens in file with and without preprocessing
    prompt = get_prompt()
    for file in files:
        file_txt = open_file(file)
        token_count += encode(prompt + file_txt)
    
    # Batch files here
    batch_files, _ = dynamicBatching(files)
    for file in batch_files:
        token_count_preprocessed += encode(file)
    
    results[directory] = token_count_preprocessed


    print(f' Number of tokens used for this project without preprocessing {token_count:,}')
    print(f' Number of tokens used for this project with preprocessing {token_count_preprocessed:,}')

    print(f' Tokens reduced by {100 - ((token_count_preprocessed / token_count) * 100):.2f}% after preprocessing file')

    input_token_cost = token_count * INPUT_COST
    output_token_cost = token_count * OUTPUT_COST * 0.30
    total_token_cost = input_token_cost + output_token_cost


    input_token_cost_preprocessed = token_count_preprocessed * INPUT_COST
    output_token_cost_preprocessed = token_count_preprocessed * OUTPUT_COST * 0.30
    total_preprocess_cost = input_token_cost_preprocessed + output_token_cost_preprocessed


    print(f' Estimated Total Cost without preprocessing ${total_token_cost :.2f} | Input Cost ${input_token_cost :.2f} | Output Cost ${output_token_cost :.2f}')
    print(f' Estimated Total Cost with preprocessing ${total_preprocess_cost :.2f} | Input Cost ${input_token_cost_preprocessed :.2f} | Output Cost ${output_token_cost_preprocessed :.2f}')

    return


def find_java_files(directory):    
    return glob.glob(os.path.join(directory, '**/*.java'), recursive=True)


def open_file(file):
    try:
        with open(file, 'r', encoding='utf-8') as txt:
            return txt.read()
    except Exception as e:
        # print(f'Error opening file {file}')
        return ''



def encode(file):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(file))


def preprocess_file(file):
    processed_file = []

    try:
        with open(file, 'r', encoding='utf-8') as java_file:
            for i, line in enumerate(java_file):
                stripped_line = line.strip()
                processed_file.append(stripped_line)
        return ''.join(processed_file)
    except Exception as e:
        return ''


def get_directories_in(directory):
    # List all items in the directory
    items = os.listdir(directory)

    # Filter out items that are not directories
    directories = [item for item in items if os.path.isdir(os.path.join(directory, item))]

    return directories

def get_tokenize_file_count_of_json():
    file = open('../cwe200/backend/Files/CWEToyDataset/data.json', "r")
    data = json.load(file)
    json_string = json.dumps(data)

    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(json_string))
    output_price = token_count * (0.03 / 1000)
    print(output_price)

# This is the same algorithm used in the backend to batch files backend/src/chat-gpt/chat-gpt.service.ts
def dynamicBatching(files):
        maxTokensPerBatch = 10000; # Maximum number of tokens per batch
        prompt = get_prompt() # Prompt to be used at the beginning of each batch
        promptTokenCount = encode(prompt)
        batchesOfText = []; # Array to hold all batches of text
        filesPerBatch = []; # Used later on by the progress bar
        batchText = prompt 
        totalBatchTokenCount = promptTokenCount
        currentFileCount = 0
        i = 0
    
        for file in files:
            # Clean up the file (Removes unnecessary whitespace)
            fileContent = process_java_file(file)
    
            # Add the start and end bounds to the file and then get its token count
            currentFileTokenCount = encode(add_file_boundary_markers(file, fileContent))
    
            if (currentFileTokenCount > maxTokensPerBatch):
                # print(f'File {file} is too large to fit in a single batch {i}')
    
                # If there is already content in the current batch, push it and start a new batch
                if (totalBatchTokenCount > promptTokenCount):
                    batchesOfText.append(batchText)
                    filesPerBatch.append(currentFileCount)
                    i = 0
                    batchText = prompt # Start with a fresh batch that includes the prompt
                    currentFileCount = 0 # Reset file count for the new batch
                
    
                startIndex = 0
                endIndex = 0
    
                while (startIndex < len(fileContent)):
                    endIndex = startIndex + min(maxTokensPerBatch * 3, len(fileContent) - startIndex)
    
                    sliceWithBounds = add_file_boundary_markers(file, fileContent[startIndex:endIndex])
                    totalsize = len(fileContent)
                    slicesize = len(sliceWithBounds)
    
                    # Create a new batch with the prompt and the current slice
                    batchText = prompt + sliceWithBounds
                    sliceTokens = encode(batchText)
    
                    batchesOfText.append(batchText);  # Push the new batch immediately
                    filesPerBatch.append(1); # Each slice from an oversized file is treated as a separate batch with 1 file
                    i = 0
                    # Update the startIndex for the next slice
                    startIndex = endIndex
                
            # Current batch full, push it and start a new one
            elif (totalBatchTokenCount + currentFileTokenCount > maxTokensPerBatch):
                batchesOfText.append(batchText)
                filesPerBatch.append(currentFileCount)
                i = 0
                batchText = prompt + add_file_boundary_markers(file, fileContent)
                totalBatchTokenCount = promptTokenCount + currentFileTokenCount
                currentFileCount = 1 # Start counting files in the new batch
            
            
            # The current file can be added to the current batch
            else:
                batchText += add_file_boundary_markers(file, fileContent)
                totalBatchTokenCount += currentFileTokenCount
                currentFileCount += 1 # Increment the count of files in the current batch
            
            i += 1
    
        # Add the last batch if it contains any content beyond the initial prompt
        if (len(batchText) > len(prompt)):
            batchesOfText.append(batchText)
            filesPerBatch.append(currentFileCount)
        
    
        return batchesOfText, filesPerBatch


def process_java_file(file_path: str) -> str:
    # Read file contents
    processed_lines = []
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            for line in file:
                trimmed_line = line.strip()
                processed_lines.append(trimmed_line)
    except FileNotFoundError:
        # print(f"File not found: {file_path}")
        return ''
    except Exception as e:
        # print(f"Error processing file {file_path}: {e}")
        return ''
    
    # Return processed file
    return '\n'.join(processed_lines)

def add_file_boundary_markers(file_path: str, file_content: str) -> str:
    file_name = os.path.basename(file_path)
    return f'-----BEGIN FILE: [{file_name}]----- \n{file_content}\n-----END FILE: [{file_name}]-----'
      
def get_prompt():
    with open('backend/src/chat-gpt/prompt.ts', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Regular expression to match text between backticks
    match = re.search(r'`([^`]*)`', content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return None
    
# This function estimates the number of tokens in the output of the projects based on ones that we have already run.
# So far, the avg is 0.30 tokens per input token
def estimate_output_tokens():
    base_directory = 'testing/Projects'
    projects = get_directories_in(base_directory)

    for project in projects:
        project_output = os.path.join(base_directory, project, 'data.json')
        with open(project_output, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f'{project} has {encode(content)} output tokens')


        


    
    # solr = 9,729,265, Jenkins = 3,320,776


if __name__ == "__main__":
    # get_tokenize_file_count_of_json()
    # estimate_output_tokens()
    main()
