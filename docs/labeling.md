### Labeling 

#### Attack Surface Detection Engine
To label new examples for the Attack Surface Detection Engine (ATDE), turn your attention to `testing/Labeling/Attack Surface`. To make it collaborative as possible we use excel files so that they can be shared via Google Sheets. 

Let's say you want to label some new data. 

1. You first need to run the associated data (The java files or a whole project) through to tool. We just need the `parsedResults.json` and the `data.json` for the data.
2. Next, open up the file `testing/Labeling/Attack Surface/json_to_excel.py` and set the name of the project one line `251` to match your project's name. 
3. Run the script and it will generate an excel file for you to label. 
4. You can either label it locally or put it in Google Drive.

> **Note:** You do not have to label every single attack surface if you decide you just want to partially label data. From example, say you pick a project with 2000 Java files. You can label as few as you want. 

> **Important:** Currently only the labels in the `Kyler` column count so just use that one.

5. When you are done labeling download the excel file and place it in `testing/Labeling/Attack Surface/Data` 
6. Run the script at `testing/Labeling/Attack Surface/parse_labels.py`. This will generate a JSON at `backend/src/bert/training/data/labels.json` which can be used with `backend/src/bert/training/train_attack_surface_models.py` to update the ASDE models.

#### Flow Verification Engine
To label the Flow Verification Engine (FVE) make sure you are using the branch `labeling-branch` this branch disables the FVE so that you can see all of the raw data. To make sure the code hasn't been changed check that both 
1. lines `99` through `109` are commented out in `backend/src/analyze/analyze.service.ts`
2. line `27` in `backend/src/code-ql/codql-parser-service.ts` are commented out

Then all you have to do is run a project though the tool and view the results on the frontend. Results that have dataflow associated with them will have Flow Nodes on the left side of the GUI. Ues the `Yes` or `No` buttons to label each of the flows then hit submit. This will save labeled flows for your project at `testing/Labeling/Flow Verification/FlowData`. Which is used to train the model at `backend/src/bert/training/train_flow_verification_model.py`.

>**Tip:** If you are unsure about a flow use an LLM. I find that Grok actually works best for this task, but feel free to try any. Something like "Can you tell me if this dataflow exposes sensitive information in the last node?" To help with this paste in the flow data for the result from `backend/Files/<project>/flowMapsByCWE.json`

For example:

```json
"flow": [
                        {
                            "step": 0,
                            "variableName": "dbConnectionString",
                            "startLine": 64,
                            "startColumn": 52,
                            "endLine": 64,
                            "endColumn": 70,
                            "uri": "CWEToyDataset/src/main/java/com/mycompany/app/CWE-201/BAD/BAD_ExposeErrorSensitiveDetailsInServletResponse.java",
                            "type": "String",
                            "code": "            out.println(\"\\n-- Session Information --\");\r\n            out.println(sessionInfo);\r\n            out.println(\"\\n-- Internal Configuration Details --\");\r\n            out.println(\"DB Connection String: \" + dbConnectionString);\r\n            out.println(\"API Key: \" + apiKey);\r\n            out.println(\"Configuration File: \" + configFilePath);\r\n            out.println(\"Environment Variables: \" + envVariables);\r"
                        },
                        {
                            "step": 1,
                            "variableName": "\"DB Connection String: \" + dbConnectionString",
                            "startLine": 64,
                            "startColumn": 25,
                            "endLine": 64,
                            "endColumn": 70,
                            "uri": "CWEToyDataset/src/main/java/com/mycompany/app/CWE-201/BAD/BAD_ExposeErrorSensitiveDetailsInServletResponse.java",
                            "type": "",
                            "code": "            out.println(\"\\n-- Session Information --\");\r\n            out.println(sessionInfo);\r\n            out.println(\"\\n-- Internal Configuration Details --\");\r\n            out.println(\"DB Connection String: \" + dbConnectionString);\r\n            out.println(\"API Key: \" + apiKey);\r\n            out.println(\"Configuration File: \" + configFilePath);\r\n            out.println(\"Environment Variables: \" + envVariables);\r"
                        }
                    ]
                    
