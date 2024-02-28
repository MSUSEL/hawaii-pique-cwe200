# README #

CWE 200 query repo and web app

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

#### Dependencies
1. Install [Node.js](https://nodejs.org/en?ref=altcademy.com)
2. Install [Maven](https://maven.apache.org/install.html)
3. Install [CodeQL CLI](https://github.com/github/codeql-cli-binaries/releases)

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

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact