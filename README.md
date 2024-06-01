# CWE-200 Web App Tool Repo #

## What is this repository for? ##

### Quick summary ###

This tool is designed to detect and identify vulnerabilities related to CWE-200 (Exposure of Sensitive Information to an Unauthorized Actor) within Java projects. 
Sensitive information may be inadvertently exposed through various methods, including logging activities, revealing underlying program logic, displaying directory information, 
or printing usernames and passwords, among others. Given the complexity and subtlety of these exposures, our tool offers a comprehensive scanning solution. 
Users can upload their Java projects for a thorough analysis, ensuring all instances of such vulnerabilities are identified.

### Current Coverage ###

- CWE-200: Exposure of Sensitive Information to an Unauthorized Actor
- CWE-201: Insertion of Sensitive Information Into Sent Data
- CWE-204: Observable Response Discrepancy
- CWE-208: Observable Timing Discrepancy
- CWE-209: Generation of Error Message Containing Sensitive Information
- CWE-214: Invocation of Process Using Visible Sensitive Information
- CWE-215: Insertion of Sensitive Information Into Debugging Code
- CWE-531: Inclusion of Sensitive Information in Test Code
- CWE-532: Insertion of Sensitive Information into Log File
- CWE-535: Exposure of Information Through Shell Error Message
- CWE-536: Servlet Runtime Error Message Containing Sensitive Information
- CWE-537: Java Runtime Error Message Containing Sensitive Information
- CWE-538: Insertion of Sensitive Information into Externally-Accessible File or Directory
- CWE-540: Inclusion of Sensitive Information in Source Code
- CWE-548: Exposure of Information Through Directory Listing
- CWE-550: Server-generated Error Message Containing Sensitive Information
- CWE-598: Use of GET Request Method With Sensitive Query Strings
- CWE-615: Inclusion of Sensitive Information in Source Code Comments



### Future Coverage ###
- CWE-205: Observable Behavioral Discrepancy
- ~~CWE-210: Self-generated Error Message Containing Sensitive Information~~ (Covered by 209, look to see if it can be narrowed)
- ~~CWE-211: Externally-Generated Error Message Containing Sensitive Information~~ (Covered by 209, look to see if it can be narrowed)
- CWE-213: Exposure of Sensitive Information Due to Incompatible Policies
- ~~CWE-497: Exposure of Sensitive System Information to an Unauthorized Control Sphere~~ (Already covered by other queries)
- ~~CWE-541: Inclusion of Sensitive Information in an Include File~~ (Not relevant to Java)
- CWE-651: Exposure of WSDL File Containing Sensitive Information (Ibrahim is working on this)




### How do I get set up? ###

#### Dependencies
1. [Node.js](https://nodejs.org/en?ref=altcademy.com) You will need to add Node as a system env variable
2. [Maven](https://maven.apache.org/install.html) You will need to add MVN as a system env variable
3. [CodeQL CLI](https://github.com/github/codeql-cli-binaries/releases) You will need to add CodeQL as a system env variable

**In root directory**

1. Install dependencies
```bash 
npm ci
```

2. Build backend and frontend
```bash
npm run build
```

3. Setup Codeql dependencies 
```bash
npm run codeql-setup
```

4. To launch backend:
```bash
npm run start-backend
```

5. To launch frontend:
```bash
npm run start-frontend
```

**Note: ** 4 and 5 may be need to run in separate shells / consoles


### Query Testing ###
There is a labeled toy dataset located in `backend/Files/CWEToyDataset` that includes 5 test cases for each CWE. Within each specific CWE file are 5 .java files with 3 BAD (Vulnerable) and 2 GOOD (Secure) code examples. 
To test the results of the queries run this command `codeql database analyze backend/Files/CWEToyDataset/CWEToyDatasetdb --format=sarifv2.1.0 --output=out.sarif --rerun codeql/codeql-custom-queries-java` from the project root. 
After this is complete, navigate to `testing/toy-dataset-results.py` and run this file. You should see the accuracies printed out in the terminal. If you want to know exactly how each query classified their corresponding test cases. Look in the `codeql_results.json` file located in the same directory.

#### Making your queries work with the testing script ####
If you make a new query or modify one, make sure you add the @id CWE-<NUMBER> and @cwe CWE-<NUMBER> tags to the meta data at the top of the query.

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact