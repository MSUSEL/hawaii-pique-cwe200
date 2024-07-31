import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.body.Parameter;
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
import java.util.HashSet;
import java.util.Set;

public class ParseJava {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: ParseJava <file-path>");
            return;
        }

        String filePath = args[0];
        // String filePath = "src/sensFiles/TemporaryFolder.java"; // Adjust the path as needed
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

                    Set<String> variables = new HashSet<>();
                    Set<String> comments = new HashSet<>();
                    Set<String> strings = new HashSet<>();
                    Set<String> methodCalls = new HashSet<>();

                    cu.accept(new VariableCollector(), variables);
                    cu.accept(new CommentCollector(), comments);
                    cu.accept(new StringLiteralCollector(), strings);
                    cu.accept(new MethodCallCollector(), methodCalls);

                    JSONObject jsonOutput = new JSONObject();
                    jsonOutput.put("filename", fileName);
                    jsonOutput.put("variables", new JSONArray(variables));
                    jsonOutput.put("comments", new JSONArray(comments));
                    jsonOutput.put("strings", new JSONArray(strings));
                    jsonOutput.put("sinks", new JSONArray(methodCalls));

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

    // Visitor class to collect variable names from the AST
    private static class VariableCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(VariableDeclarator vd, Set<String> collector) {
            try {
                super.visit(vd, collector);
                collector.add(vd.getNameAsString());
            } catch (Exception e) {
                System.err.println("Error collecting variable: " + e.getMessage());
            }
        }

        @Override
        public void visit(Parameter param, Set<String> collector) {
            try {
                super.visit(param, collector);
                collector.add(param.getNameAsString());
            } catch (Exception e) {
                System.err.println("Error collecting parameter: " + e.getMessage());
            }
        }

        @Override
        public void visit(FieldDeclaration fd, Set<String> collector) {
            try {
                super.visit(fd, collector);
                for (VariableDeclarator vd : fd.getVariables()) {
                    collector.add(vd.getNameAsString());
                }
            } catch (Exception e) {
                System.err.println("Error collecting field declaration: " + e.getMessage());
            }
        }

        @Override
        public void visit(CatchClause cc, Set<String> collector) {
            try {
                super.visit(cc, collector);
                collector.add(cc.getParameter().getNameAsString());
            } catch (Exception e) {
                System.err.println("Error collecting catch clause: " + e.getMessage());
            }
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

        @Override
        public void visit(ClassOrInterfaceDeclaration cid, Set<String> collector) {
            try {
                super.visit(cid, collector);
                if (cid.getComment().isPresent()) {
                    collector.add(cid.getComment().get().getContent().trim());
                }
            } catch (Exception e) {
                System.err.println("Error collecting comments in ClassOrInterfaceDeclaration: " + e.getMessage());
            }
        }

        @Override
        public void visit(MethodDeclaration md, Set<String> collector) {
            try {
                super.visit(md, collector);
                if (md.getComment().isPresent()) {
                    collector.add(md.getComment().get().getContent().trim());
                }
                md.getBody().ifPresent(body -> {
                    for (Comment comment : body.getAllContainedComments()) {
                        collector.add(comment.getContent().trim());
                    }
                });
            } catch (Exception e) {
                System.err.println("Error collecting comments in MethodDeclaration: " + e.getMessage());
            }
        }

        @Override
        public void visit(FieldDeclaration fd, Set<String> collector) {
            try {
                super.visit(fd, collector);
                if (fd.getComment().isPresent()) {
                    collector.add(fd.getComment().get().getContent().trim());
                }
            } catch (Exception e) {
                System.err.println("Error collecting comments in FieldDeclaration: " + e.getMessage());
            }
        }

        @Override
        public void visit(ConstructorDeclaration cd, Set<String> collector) {
            try {
                super.visit(cd, collector);
                if (cd.getComment().isPresent()) {
                    collector.add(cd.getComment().get().getContent().trim());
                }
                cd.getBody().getAllContainedComments().forEach(comment -> {
                    collector.add(comment.getContent().trim());
                });
            } catch (Exception e) {
                System.err.println("Error collecting comments in ConstructorDeclaration: " + e.getMessage());
            }
        }

        @Override
        public void visit(EnumDeclaration ed, Set<String> collector) {
            try {
                super.visit(ed, collector);
                if (ed.getComment().isPresent()) {
                    collector.add(ed.getComment().get().getContent().trim());
                }
            } catch (Exception e) {
                System.err.println("Error collecting comments in EnumDeclaration: " + e.getMessage());
            }
        }

        @Override
        public void visit(BlockStmt bs, Set<String> collector) {
            try {
                super.visit(bs, collector);
                for (Comment comment : bs.getAllContainedComments()) {
                    collector.add(comment.getContent().trim());
                }
            } catch (Exception e) {
                System.err.println("Error collecting comments in BlockStmt: " + e.getMessage());
            }
        }
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
