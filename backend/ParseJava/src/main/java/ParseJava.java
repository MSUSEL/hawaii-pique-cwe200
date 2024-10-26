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
import com.github.javaparser.ast.body.EnumDeclaration;
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
        // String filePath = "src/sensFiles/TemporaryFolder.java"; // Adjust the path as needed
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
                    Set<String> strings = new HashSet<>();
                    Set<String> methodCalls = new HashSet<>();

                    // New data structures
                    Map<String, String> methodCodeMap = new HashMap<>(); // Map from method name to method code
                    Set<String> methodNames = new HashSet<>(); // Set of method names

                    // Variables now include name, type, methods
                    Map<String, JSONObject> variables = new HashMap<>(); // Map from variable name to its details

                    // Collect methods and variables within them
                    cu.accept(new MethodCollector(methodNames, methodCodeMap, variables), null);

                    // Collect variables outside methods (e.g., global variables)
                    cu.accept(new GlobalVariableCollector(variables), null);

                    // Existing collectors
                    cu.accept(new CommentCollector(), comments);
                    cu.accept(new StringLiteralCollector(), strings);
                    cu.accept(new MethodCallCollector(), methodCalls);

                    // For global variables that only appear in "global", create custom method mappings
                    for (Map.Entry<String, JSONObject> entry : variables.entrySet()) {
                        JSONObject varInfo = entry.getValue();
                        JSONArray methods = varInfo.getJSONArray("methods");
                        if (methods.length() == 1 && methods.getString(0).equals("global")) {
                            String variableName = entry.getKey();
                            // Collect lines of code where this variable appears
                            Set<String> codeLines = new HashSet<>();
                            cu.accept(new VariableUsageCollector(variableName), codeLines);

                            // Create custom method name
                            String customMethodName = variableName + "_lines";

                            // Replace "global" in methods array with custom method name
                            methods.remove(0);
                            methods.put(customMethodName);

                            // Concatenate code lines
                            StringBuilder codeBuilder = new StringBuilder();
                            for (String codeLine : codeLines) {
                                codeBuilder.append(codeLine).append("\n");
                            }

                            // Add to methodCodeMap
                            methodCodeMap.put(customMethodName, codeBuilder.toString());

                            // Remove "codeLines" from varInfo if it exists
                            varInfo.remove("codeLines");
                        }
                    }

                    JSONObject jsonOutput = new JSONObject();
                    jsonOutput.put("filename", fileName);

                    // Convert variables map to JSON array, ensuring fields are ordered as name, type, methods
                    JSONArray variablesArray = new JSONArray();
                    for (JSONObject varInfo : variables.values()) {
                        // Reorder the keys
                        JSONObject orderedVarInfo = new JSONObject();
                        orderedVarInfo.put("name", varInfo.get("name"));
                        orderedVarInfo.put("type", varInfo.get("type"));
                        orderedVarInfo.put("methods", varInfo.get("methods"));
                        variablesArray.put(orderedVarInfo);
                    }
                    jsonOutput.put("variables", variablesArray);

                    jsonOutput.put("comments", new JSONArray(comments));
                    jsonOutput.put("strings", new JSONArray(strings));
                    jsonOutput.put("methodCalls", new JSONArray(methodCalls));
                    jsonOutput.put("methodNames", new JSONArray(methodNames));

                    // Convert methodCodeMap to JSON
                    JSONObject methodCodeJson = new JSONObject();
                    for (Map.Entry<String, String> entry : methodCodeMap.entrySet()) {
                        methodCodeJson.put(entry.getKey(), entry.getValue());
                    }
                    jsonOutput.put("methodCodeMap", methodCodeJson);

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

    // Visitor class to collect method names and code, and variables within methods
    private static class MethodCollector extends VoidVisitorAdapter<Void> {
        private Set<String> methodNames;
        private Map<String, String> methodCodeMap;
        private Map<String, JSONObject> variables;

        public MethodCollector(Set<String> methodNames, Map<String, String> methodCodeMap, Map<String, JSONObject> variables) {
            this.methodNames = methodNames;
            this.methodCodeMap = methodCodeMap;
            this.variables = variables;
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

    // Visitor class to collect usages of variables, including declarations
    private static class VariableUsageCollector extends VoidVisitorAdapter<Set<String>> {
        private String variableName;

        public VariableUsageCollector(String variableName) {
            this.variableName = variableName;
        }

        @Override
        public void visit(NameExpr ne, Set<String> codeLines) {
            super.visit(ne, codeLines);
            if (ne.getNameAsString().equals(variableName)) {
                // Get the line of code where the variable is used
                codeLines.add(getFullLine(ne));
            }
        }

        @Override
        public void visit(VariableDeclarator vd, Set<String> codeLines) {
            super.visit(vd, codeLines);
            if (vd.getNameAsString().equals(variableName)) {
                // Get the line of code where the variable is declared
                codeLines.add(getFullLine(vd));
            }
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
                    collector.add(comment.getContent().trim());
                }
            } catch (Exception e) {
                System.err.println("Error collecting comments in CompilationUnit: " + e.getMessage());
            }
        }

        // ... [Rest of your existing CommentCollector methods]
    }

    // Visitor class to collect string literals from the AST
    private static class StringLiteralCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(StringLiteralExpr sle, Set<String> collector) {
            try {
                super.visit(sle, collector);
                String value = sle.getValue().replace("\\", "").replace("'", "").trim();
                if (!value.isEmpty() && !value.equals(" ")) {
                    collector.add(value);
                }
            } catch (Exception e) {
                System.err.println("Error collecting string literal: " + e.getMessage());
            }
        }
    }

    // Visitor class to collect method calls from the AST
    private static class MethodCallCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(MethodCallExpr mce, Set<String> collector) {
            try {
                super.visit(mce, collector);
                collector.add(mce.getNameAsString());
            } catch (Exception e) {
                System.err.println("Error collecting method call: " + e.getMessage());
            }
        }
    }
}
