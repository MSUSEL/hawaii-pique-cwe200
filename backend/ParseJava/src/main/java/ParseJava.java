import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.HashMap;
import java.util.Map;

public class ParseJava {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: ParseJava <file-path>");
            return;
        }

        String filePath = args[0];
        // String filePath = "src/sensFiles/ClassicPluginStrategy.java"; // Adjust the path as needed

        String fileName = filePath.substring(filePath.lastIndexOf("/") + 1);

        try {
            // Configure the JavaParser
            ParserConfiguration config = new ParserConfiguration();
            config.setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_8); // Adjust the language level as needed

            // Create a JavaParser instance with the configuration
            JavaParser javaParser = new JavaParser(config);

            try (FileInputStream in = new FileInputStream(filePath)) {
                ParseResult<CompilationUnit> result = javaParser.parse(in);

                if (result.isSuccessful() && result.getResult().isPresent()) {
                    CompilationUnit cu = result.getResult().get();

                    Set<String> comments = new HashSet<>();
                    Set<String> methodCalls = new HashSet<>();

                    // New data structures
                    Map<String, String> methodCodeMap = new HashMap<>(); // Map from method name to method code
                    Set<String> methodNames = new HashSet<>(); // Set of method names

                    // Variables now include name, type, methods
                    Map<String, JSONObject> variables = new HashMap<>(); // Map from variable name to its details

                    // Strings include name, methods
                    Map<String, JSONObject> strings = new HashMap<>(); // Map from string value to its details

                    // **New data structure for sinks**
                    Map<String, JSONObject> sinks = new HashMap<>(); // Map from sink name to its details

                    // Collect methods, variables, and strings within them
                    cu.accept(new MethodCollector(methodNames, methodCodeMap, variables, strings), null);

                    // Collect variables outside methods (e.g., global variables)
                    cu.accept(new GlobalVariableCollector(variables), null);

                    // Collect strings outside methods (global scope)
                    cu.accept(new GlobalStringCollector(strings), null);

                    // Existing collectors
                    cu.accept(new CommentCollector(), comments);

                    // **Collect sinks (method calls)**
                    cu.accept(new SinkCollector(sinks, methodCodeMap, cu), null);

                    // Process variables with global scope
                    for (Map.Entry<String, JSONObject> entry : variables.entrySet()) {
                        JSONObject varInfo = entry.getValue();
                        JSONArray methods = varInfo.getJSONArray("methods");
                        String variableName = entry.getKey();

                        if (methods.toList().contains("global")) {
                            // Collect lines of code where this variable appears in the global scope
                            Set<String> codeLines = new HashSet<>();
                            cu.accept(new VariableUsageCollector(variableName, true), codeLines);

                            if (!codeLines.isEmpty()) {
                                // Create custom method name
                                String customMethodName = variableName + "_global_lines_variable";

                                // Replace "global" in methods array with custom method name
                                JSONArray updatedMethods = new JSONArray();
                                for (int i = 0; i < methods.length(); i++) {
                                    String methodName = methods.getString(i);
                                    if (!methodName.equals("global")) {
                                        updatedMethods.put(methodName);
                                    }
                                }
                                updatedMethods.put(customMethodName);
                                varInfo.put("methods", updatedMethods);

                                // Concatenate code lines
                                StringBuilder codeBuilder = new StringBuilder();
                                for (String codeLine : codeLines) {
                                    codeBuilder.append(codeLine).append("\n");
                                }

                                // Add to methodCodeMap
                                methodCodeMap.put(customMethodName, codeBuilder.toString());
                            }
                        }
                    }

                    // Process strings with global scope
                    for (Map.Entry<String, JSONObject> entry : strings.entrySet()) {
                        JSONObject strInfo = entry.getValue();
                        JSONArray methods = strInfo.getJSONArray("methods");
                        String stringValue = entry.getKey();

                        if (methods.toList().contains("global")) {
                            // Collect lines of code where this string appears in the global scope
                            Set<String> codeLines = new HashSet<>();
                            cu.accept(new StringUsageCollector(stringValue, true), codeLines);

                            if (!codeLines.isEmpty()) {
                                // Create custom method name
                                String customMethodName = stringValue + "_global_lines_string";

                                // Replace "global" in methods array with custom method name
                                JSONArray updatedMethods = new JSONArray();
                                for (int i = 0; i < methods.length(); i++) {
                                    String methodName = methods.getString(i);
                                    if (!methodName.equals("global")) {
                                        updatedMethods.put(methodName);
                                    }
                                }
                                updatedMethods.put(customMethodName);
                                strInfo.put("methods", updatedMethods);

                                // Concatenate code lines
                                StringBuilder codeBuilder = new StringBuilder();
                                for (String codeLine : codeLines) {
                                    codeBuilder.append(codeLine).append("\n");
                                }

                                // Add to methodCodeMap
                                methodCodeMap.put(customMethodName, codeBuilder.toString());
                            }
                        }
                    }

                    // **Process sinks with global scope**
                    for (Map.Entry<String, JSONObject> entry : sinks.entrySet()) {
                        JSONObject sinkInfo = entry.getValue();
                        JSONArray methods = sinkInfo.getJSONArray("methods");
                        String sinkName = entry.getKey();

                        if (methods.toList().contains("global")) {
                            // Collect lines of code where this sink appears in the global scope
                            Set<String> codeLines = new HashSet<>();
                            cu.accept(new SinkUsageCollector(sinkName, true), codeLines);

                            if (!codeLines.isEmpty()) {
                                // Create custom method name
                                String customMethodName = sinkName + "_global_lines_sink";

                                // Replace "global" in methods array with custom method name
                                JSONArray updatedMethods = new JSONArray();
                                for (int i = 0; i < methods.length(); i++) {
                                    String methodName = methods.getString(i);
                                    if (!methodName.equals("global")) {
                                        updatedMethods.put(methodName);
                                    }
                                }
                                updatedMethods.put(customMethodName);
                                sinkInfo.put("methods", updatedMethods);

                                // Concatenate code lines
                                StringBuilder codeBuilder = new StringBuilder();
                                for (String codeLine : codeLines) {
                                    codeBuilder.append(codeLine).append("\n");
                                }

                                // Add to methodCodeMap
                                methodCodeMap.put(customMethodName, codeBuilder.toString());
                            }
                        }
                    }

                    JSONObject jsonOutput = new JSONObject();
                    jsonOutput.put("filename", fileName);

                    // Convert variables map to JSON array
                    JSONArray variablesArray = new JSONArray();
                    for (JSONObject varInfo : variables.values()) {
                        JSONObject orderedVarInfo = new JSONObject();
                        orderedVarInfo.put("name", varInfo.get("name"));
                        orderedVarInfo.put("type", varInfo.get("type"));
                        orderedVarInfo.put("methods", varInfo.get("methods"));
                        variablesArray.put(orderedVarInfo);
                    }
                    jsonOutput.put("variables", variablesArray);

                    // Convert strings map to JSON array
                    JSONArray stringsArray = new JSONArray();
                    for (JSONObject strInfo : strings.values()) {
                        JSONObject orderedStrInfo = new JSONObject();
                        orderedStrInfo.put("name", strInfo.get("name"));
                        orderedStrInfo.put("methods", strInfo.get("methods"));
                        stringsArray.put(orderedStrInfo);
                    }
                    jsonOutput.put("strings", stringsArray);

                    // Convert methodCodeMap to JSON
                    JSONObject methodCodeJson = new JSONObject();
                    for (Map.Entry<String, String> entry : methodCodeMap.entrySet()) {
                        methodCodeJson.put(entry.getKey(), entry.getValue());
                    }
                    jsonOutput.put("methodCodeMap", methodCodeJson);

                    // Convert comments to JSON array

                    JSONArray commentsArray = new JSONArray();
                    for (String comment : comments) {
                        JSONObject commentObj = new JSONObject();
                        commentObj.put("name", comment);
                        commentObj.put("methods", "");
                        commentsArray.put(commentObj);
                    }
                    jsonOutput.put("comments", commentsArray);

                    // Convert sinks map to JSON array
                    JSONArray sinksArray = new JSONArray();
                    for (JSONObject sinkInfo : sinks.values()) {
                        JSONObject orderedSinkInfo = new JSONObject();
                        orderedSinkInfo.put("name", sinkInfo.get("name"));
                        orderedSinkInfo.put("methods", sinkInfo.get("methods"));
                        sinksArray.put(orderedSinkInfo);
                    }
                    jsonOutput.put("sinks", sinksArray);

                    System.out.println(jsonOutput.toString(2));
                } else {
                    System.out.println("Parsing failed");

                    result.getProblems().forEach(problem -> System.out.println(problem.getMessage()));
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Visitor class to collect method names and code, variables, and strings within methods
    private static class MethodCollector extends VoidVisitorAdapter<Void> {
        private Set<String> methodNames;
        private Map<String, String> methodCodeMap;
        private Map<String, JSONObject> variables;
        private Map<String, JSONObject> strings;

        public MethodCollector(Set<String> methodNames, Map<String, String> methodCodeMap, Map<String, JSONObject> variables, Map<String, JSONObject> strings) {
            this.methodNames = methodNames;
            this.methodCodeMap = methodCodeMap;
            this.variables = variables;
            this.strings = strings;
        }

        @Override
        public void visit(MethodDeclaration md, Void arg) {
            try {
                super.visit(md, arg);

                String methodName = md.getNameAsString();
                methodNames.add(methodName);

                String methodCode = md.toString();
                methodCodeMap.put(methodName, methodCode);

                // Collect variables within this method
                VariableCollector variableCollector = new VariableCollector(variables, methodName);
                md.accept(variableCollector, null);

                // Collect strings within this method
                StringCollector stringCollector = new StringCollector(strings, methodName);
                md.accept(stringCollector, null);
            } catch (Exception e) {
                System.err.println("Error collecting method: " + e.getMessage());
            }
        }

        @Override
        public void visit(ConstructorDeclaration cd, Void arg) {
            try {
                super.visit(cd, arg);

                String methodName = cd.getNameAsString();
                methodNames.add(methodName);

                String methodCode = cd.toString();
                methodCodeMap.put(methodName, methodCode);

                // Collect variables within this constructor
                VariableCollector variableCollector = new VariableCollector(variables, methodName);
                cd.accept(variableCollector, null);

                // Collect strings within this constructor
                StringCollector stringCollector = new StringCollector(strings, methodName);
                cd.accept(stringCollector, null);
            } catch (Exception e) {
                System.err.println("Error collecting constructor: " + e.getMessage());
            }
        }
    }

    // Visitor class to collect variables within methods
    private static class VariableCollector extends VoidVisitorAdapter<Void> {
        private Map<String, JSONObject> variables;
        private String currentMethodName;

        public VariableCollector(Map<String, JSONObject> variables, String currentMethodName) {
            this.variables = variables;
            this.currentMethodName = currentMethodName != null ? currentMethodName : "global";
        }

        @Override
        public void visit(VariableDeclarator vd, Void arg) {
            try {
                super.visit(vd, arg);
                String varName = vd.getNameAsString();
                String varType = vd.getType().asString();

                // Update variable info
                JSONObject varInfo = variables.getOrDefault(varName, new JSONObject());
                varInfo.put("name", varName);
                varInfo.put("type", varType);

                // Add method to the methods list
                JSONArray methods = varInfo.has("methods") ? varInfo.getJSONArray("methods") : new JSONArray();
                if (!methods.toList().contains(currentMethodName)) {
                    methods.put(currentMethodName);
                }
                varInfo.put("methods", methods);

                variables.put(varName, varInfo);
            } catch (Exception e) {
                System.err.println("Error collecting variable in method: " + e.getMessage());
            }
        }

        @Override
        public void visit(Parameter param, Void arg) {
            try {
                super.visit(param, arg);
                String varName = param.getNameAsString();
                String varType = param.getType().asString();

                // Update variable info
                JSONObject varInfo = variables.getOrDefault(varName, new JSONObject());
                varInfo.put("name", varName);
                varInfo.put("type", varType);

                // Add method to the methods list
                JSONArray methods = varInfo.has("methods") ? varInfo.getJSONArray("methods") : new JSONArray();
                if (!methods.toList().contains(currentMethodName)) {
                    methods.put(currentMethodName);
                }
                varInfo.put("methods", methods);

                variables.put(varName, varInfo);
            } catch (Exception e) {
                System.err.println("Error collecting parameter in method: " + e.getMessage());
            }
        }

        @Override
        public void visit(CatchClause cc, Void arg) {
            try {
                super.visit(cc, arg);
                String varName = cc.getParameter().getNameAsString();
                String varType = cc.getParameter().getType().asString();

                // Update variable info
                JSONObject varInfo = variables.getOrDefault(varName, new JSONObject());
                varInfo.put("name", varName);
                varInfo.put("type", varType);

                // Add method to the methods list
                JSONArray methods = varInfo.has("methods") ? varInfo.getJSONArray("methods") : new JSONArray();
                if (!methods.toList().contains(currentMethodName)) {
                    methods.put(currentMethodName);
                }
                varInfo.put("methods", methods);

                variables.put(varName, varInfo);
            } catch (Exception e) {
                System.err.println("Error collecting catch clause variable in method: " + e.getMessage());
            }
        }
    }

    // Visitor class to collect strings within methods
    private static class StringCollector extends VoidVisitorAdapter<Void> {
        private Map<String, JSONObject> strings;
        private String currentMethodName;

        public StringCollector(Map<String, JSONObject> strings, String currentMethodName) {
            this.strings = strings;
            this.currentMethodName = currentMethodName != null ? currentMethodName : "global";
        }

        @Override
        public void visit(StringLiteralExpr sle, Void arg) {
            try {
                super.visit(sle, arg);
                String value = sle.getValue();

                // Update string info
                JSONObject strInfo = strings.getOrDefault(value, new JSONObject());
                strInfo.put("name", value);

                // Add method to the methods list
                JSONArray methods = strInfo.has("methods") ? strInfo.getJSONArray("methods") : new JSONArray();
                if (!methods.toList().contains(currentMethodName)) {
                    methods.put(currentMethodName);
                }
                strInfo.put("methods", methods);

                strings.put(value, strInfo);
            } catch (Exception e) {
                System.err.println("Error collecting string in method: " + e.getMessage());
            }
        }
    }

    // Visitor class to collect global variables (outside methods)
    private static class GlobalVariableCollector extends VoidVisitorAdapter<Void> {
        private Map<String, JSONObject> variables;

        public GlobalVariableCollector(Map<String, JSONObject> variables) {
            this.variables = variables;
        }

        @Override
        public void visit(FieldDeclaration fd, Void arg) {
            try {
                super.visit(fd, arg);
                String varType = fd.getElementType().asString();
                for (VariableDeclarator vd : fd.getVariables()) {
                    String varName = vd.getNameAsString();

                    // Update variable info
                    JSONObject varInfo = variables.getOrDefault(varName, new JSONObject());
                    varInfo.put("name", varName);
                    varInfo.put("type", varType);

                    // Add "global" to the methods list
                    JSONArray methods = varInfo.has("methods") ? varInfo.getJSONArray("methods") : new JSONArray();
                    if (!methods.toList().contains("global")) {
                        methods.put("global");
                    }
                    varInfo.put("methods", methods);

                    variables.put(varName, varInfo);
                }
            } catch (Exception e) {
                System.err.println("Error collecting global variable: " + e.getMessage());
            }
        }
    }

    // Visitor class to collect strings in the global scope
    private static class GlobalStringCollector extends VoidVisitorAdapter<Void> {
        private Map<String, JSONObject> strings;

        public GlobalStringCollector(Map<String, JSONObject> strings) {
            this.strings = strings;
        }

        @Override
        public void visit(StringLiteralExpr sle, Void arg) {
            try {
                super.visit(sle, arg);
                if (isInGlobalScope(sle)) {
                    String value = sle.getValue();

                    // Update string info
                    JSONObject strInfo = strings.getOrDefault(value, new JSONObject());
                    strInfo.put("name", value);

                    // Add "global" to the methods list
                    JSONArray methods = strInfo.has("methods") ? strInfo.getJSONArray("methods") : new JSONArray();
                    if (!methods.toList().contains("global")) {
                        methods.put("global");
                    }
                    strInfo.put("methods", methods);

                    strings.put(value, strInfo);
                }
            } catch (Exception e) {
                System.err.println("Error collecting global string: " + e.getMessage());
            }
        }

        private boolean isInGlobalScope(com.github.javaparser.ast.Node node) {
            boolean inMethodOrConstructor = node.findAncestor(MethodDeclaration.class).isPresent() ||
                                            node.findAncestor(ConstructorDeclaration.class).isPresent();

            boolean inClass = node.findAncestor(ClassOrInterfaceDeclaration.class).isPresent();

            boolean inInitializerBlock = node.findAncestor(BlockStmt.class)
                .map(block -> block.getParentNode()
                    .filter(parent -> parent instanceof ClassOrInterfaceDeclaration)
                    .isPresent())
                .orElse(false);

            return !inMethodOrConstructor && inClass && !inInitializerBlock;
        }
    }

    // Visitor class to collect usages of variables, including declarations
    private static class VariableUsageCollector extends VoidVisitorAdapter<Set<String>> {
        private String variableName;
        private boolean onlyGlobalScope;

        public VariableUsageCollector(String variableName, boolean onlyGlobalScope) {
            this.variableName = variableName;
            this.onlyGlobalScope = onlyGlobalScope;
        }

        @Override
        public void visit(NameExpr ne, Set<String> codeLines) {
            super.visit(ne, codeLines);
            if (ne.getNameAsString().equals(variableName)) {
                if (!onlyGlobalScope || isInGlobalScope(ne)) {
                    // Get the line of code where the variable is used
                    codeLines.add(getFullLine(ne));
                }
            }
        }

        @Override
        public void visit(VariableDeclarator vd, Set<String> codeLines) {
            super.visit(vd, codeLines);
            if (vd.getNameAsString().equals(variableName)) {
                if (!onlyGlobalScope || isInGlobalScope(vd)) {
                    // Get the line of code where the variable is declared
                    codeLines.add(getFullLine(vd));
                }
            }
        }

        private boolean isInGlobalScope(com.github.javaparser.ast.Node node) {
            boolean inMethodOrConstructor = node.findAncestor(MethodDeclaration.class).isPresent() ||
                                            node.findAncestor(ConstructorDeclaration.class).isPresent();

            boolean inClass = node.findAncestor(ClassOrInterfaceDeclaration.class).isPresent();

            boolean inInitializerBlock = node.findAncestor(BlockStmt.class)
                .map(block -> block.getParentNode()
                    .filter(parent -> parent instanceof ClassOrInterfaceDeclaration)
                    .isPresent())
                .orElse(false);

            return !inMethodOrConstructor && inClass && !inInitializerBlock;
        }

        private String getFullLine(com.github.javaparser.ast.Node node) {
            int line = node.getBegin().map(pos -> pos.line).orElse(-1);
            String codeLine = node.findCompilationUnit()
                    .flatMap(cu -> cu.getStorage())
                    .flatMap(storage -> {
                        try {
                            Path path = storage.getPath();
                            java.util.List<String> lines = Files.readAllLines(path);
                            if (line > 0 && line <= lines.size()) {
                                return java.util.Optional.of(lines.get(line - 1));
                            } else {
                                return java.util.Optional.empty();
                            }
                        } catch (Exception e) {
                            return java.util.Optional.empty();
                        }
                    }).orElse(node.toString());
            return codeLine.trim();
        }
    }

    // Visitor class to collect usages of strings, including declarations
    private static class StringUsageCollector extends VoidVisitorAdapter<Set<String>> {
        private String stringValue;
        private boolean onlyGlobalScope;

        public StringUsageCollector(String stringValue, boolean onlyGlobalScope) {
            this.stringValue = stringValue;
            this.onlyGlobalScope = onlyGlobalScope;
        }

        @Override
        public void visit(StringLiteralExpr sle, Set<String> codeLines) {
            super.visit(sle, codeLines);
            if (sle.getValue().equals(stringValue)) {
                if (!onlyGlobalScope || isInGlobalScope(sle)) {
                    // Get the line of code where the string is used
                    codeLines.add(getFullLine(sle));
                }
            }
        }

        private boolean isInGlobalScope(com.github.javaparser.ast.Node node) {
            boolean inMethodOrConstructor = node.findAncestor(MethodDeclaration.class).isPresent() ||
                                            node.findAncestor(ConstructorDeclaration.class).isPresent();

            boolean inClass = node.findAncestor(ClassOrInterfaceDeclaration.class).isPresent();

            boolean inInitializerBlock = node.findAncestor(BlockStmt.class)
                .map(block -> block.getParentNode()
                    .filter(parent -> parent instanceof ClassOrInterfaceDeclaration)
                    .isPresent())
                .orElse(false);

            return !inMethodOrConstructor && inClass && !inInitializerBlock;
        }

        private String getFullLine(com.github.javaparser.ast.Node node) {
            int line = node.getBegin().map(pos -> pos.line).orElse(-1);
            String codeLine = node.findCompilationUnit()
                    .flatMap(cu -> cu.getStorage())
                    .flatMap(storage -> {
                        try {
                            Path path = storage.getPath();
                            java.util.List<String> lines = Files.readAllLines(path);
                            if (line > 0 && line <= lines.size()) {
                                return java.util.Optional.of(lines.get(line - 1));
                            } else {
                                return java.util.Optional.empty();
                            }
                        } catch (Exception e) {
                            return java.util.Optional.empty();
                        }
                    }).orElse(node.toString());
            return codeLine.trim();
        }
    }

    // Visitor class to collect comments from the AST
    private static class CommentCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(CompilationUnit cu, Set<String> collector) {
            try {
                super.visit(cu, collector);
                for (Comment comment : cu.getAllContainedComments()) {
                    collector.add(comment.getContent());
                }
            } catch (Exception e) {
                System.err.println("Error collecting comments in CompilationUnit: " + e.getMessage());
            }
        }
    }

    // **Visitor class to collect sinks (method calls)**
    private static class SinkCollector extends VoidVisitorAdapter<Void> {
        private Map<String, JSONObject> sinks;
        private String currentMethodName;
        private Map<String, String> methodCodeMap;
        private CompilationUnit cu;

        public SinkCollector(Map<String, JSONObject> sinks, Map<String, String> methodCodeMap, CompilationUnit cu) {
            this.sinks = sinks;
            this.currentMethodName = "global";
            this.methodCodeMap = methodCodeMap;
            this.cu = cu;
        }

        @Override
        public void visit(MethodDeclaration md, Void arg) {
            String previousMethodName = currentMethodName;
            currentMethodName = md.getNameAsString();
            super.visit(md, arg);
            currentMethodName = previousMethodName;
        }

        @Override
        public void visit(ConstructorDeclaration cd, Void arg) {
            String previousMethodName = currentMethodName;
            currentMethodName = cd.getNameAsString();
            super.visit(cd, arg);
            currentMethodName = previousMethodName;
        }

        @Override
        public void visit(MethodCallExpr mce, Void arg) {
            super.visit(mce, arg);
            String sinkName = mce.getNameAsString();

            // Update sink info
            JSONObject sinkInfo = sinks.getOrDefault(sinkName, new JSONObject());
            sinkInfo.put("name", sinkName);

            String methodName = currentMethodName;

            if (isInGlobalScope(mce)) {
                methodName = "global";
            }

            // Add method to the methods list
            JSONArray methods = sinkInfo.has("methods") ? sinkInfo.getJSONArray("methods") : new JSONArray();
            if (!methods.toList().contains(methodName)) {
                methods.put(methodName);
            }
            sinkInfo.put("methods", methods);

            sinks.put(sinkName, sinkInfo);
        }

        private boolean isInGlobalScope(com.github.javaparser.ast.Node node) {
            boolean inMethodOrConstructor = node.findAncestor(MethodDeclaration.class).isPresent() ||
                                            node.findAncestor(ConstructorDeclaration.class).isPresent();

            boolean inClass = node.findAncestor(ClassOrInterfaceDeclaration.class).isPresent();

            boolean inInitializerBlock = node.findAncestor(BlockStmt.class)
                .map(block -> block.getParentNode()
                    .filter(parent -> parent instanceof ClassOrInterfaceDeclaration)
                    .isPresent())
                .orElse(false);

            return !inMethodOrConstructor && inClass && !inInitializerBlock;
        }
    }

    // **Visitor class to collect sink usages in global scope**
    private static class SinkUsageCollector extends VoidVisitorAdapter<Set<String>> {
        private String sinkName;
        private boolean onlyGlobalScope;

        public SinkUsageCollector(String sinkName, boolean onlyGlobalScope) {
            this.sinkName = sinkName;
            this.onlyGlobalScope = onlyGlobalScope;
        }

        @Override
        public void visit(MethodCallExpr mce, Set<String> codeLines) {
            super.visit(mce, codeLines);
            if (mce.getNameAsString().equals(sinkName)) {
                if (!onlyGlobalScope || isInGlobalScope(mce)) {
                    // Get the line of code where the sink is used
                    codeLines.add(getFullLine(mce));
                }
            }
        }

        private boolean isInGlobalScope(com.github.javaparser.ast.Node node) {
            boolean inMethodOrConstructor = node.findAncestor(MethodDeclaration.class).isPresent() ||
                                            node.findAncestor(ConstructorDeclaration.class).isPresent();

            boolean inClass = node.findAncestor(ClassOrInterfaceDeclaration.class).isPresent();

            boolean inInitializerBlock = node.findAncestor(BlockStmt.class)
                .map(block -> block.getParentNode()
                    .filter(parent -> parent instanceof ClassOrInterfaceDeclaration)
                    .isPresent())
                .orElse(false);

            return !inMethodOrConstructor && inClass && !inInitializerBlock;
        }

        private String getFullLine(com.github.javaparser.ast.Node node) {
            int line = node.getBegin().map(pos -> pos.line).orElse(-1);
            String codeLine = node.findCompilationUnit()
                    .flatMap(cu -> cu.getStorage())
                    .flatMap(storage -> {
                        try {
                            Path path = storage.getPath();
                            java.util.List<String> lines = Files.readAllLines(path);
                            if (line > 0 && line <= lines.size()) {
                                return java.util.Optional.of(lines.get(line - 1));
                            } else {
                                return java.util.Optional.empty();
                            }
                        } catch (Exception e) {
                            return java.util.Optional.empty();
                        }
                    }).orElse(node.toString());
            return codeLine.trim();
        }
    }
}
