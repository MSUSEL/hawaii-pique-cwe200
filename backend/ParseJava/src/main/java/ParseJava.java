import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.StringLiteralExpr;
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

        

        try (FileInputStream in = new FileInputStream(filePath)) {
            JavaParser javaParser = new JavaParser();
            ParseResult<CompilationUnit> result = javaParser.parse(in);

            if (result.isSuccessful() && result.getResult().isPresent()) {
                CompilationUnit cu = result.getResult().get();

                Set<String> variables = new HashSet<>();
                Set<String> comments = new HashSet<>();
                Set<String> strings = new HashSet<>();

                cu.accept(new VariableCollector(), variables);
                cu.accept(new CommentCollector(), comments);
                cu.accept(new StringLiteralCollector(), strings);

                JSONObject jsonOutput = new JSONObject();
                jsonOutput.put("filename", fileName);
                jsonOutput.put("variables", new JSONArray(variables));
                jsonOutput.put("comments", new JSONArray(comments));
                jsonOutput.put("strings", new JSONArray(strings));

                System.out.println(jsonOutput.toString(2));
            } else {
                System.out.println("Parsing failed");

                result.getProblems().forEach(problem -> System.out.println(problem.getMessage()));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Visitor class to collect variable names from the AST
    private static class VariableCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(VariableDeclarator vd, Set<String> collector) {
            super.visit(vd, collector);
            collector.add(vd.getNameAsString());
        }

        @Override
        public void visit(Parameter param, Set<String> collector) {
            super.visit(param, collector);
            collector.add(param.getNameAsString());
        }

        @Override
        public void visit(FieldDeclaration fd, Set<String> collector) {
            super.visit(fd, collector);
            for (VariableDeclarator vd : fd.getVariables()) {
                collector.add(vd.getNameAsString());
            }
        }

        @Override
        public void visit(CatchClause cc, Set<String> collector) {
            super.visit(cc, collector);
            collector.add(cc.getParameter().getNameAsString());
        }
    }

    // Visitor class to collect comments from the AST
    private static class CommentCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(CompilationUnit cu, Set<String> collector) {
            super.visit(cu, collector);
            for (Comment comment : cu.getAllContainedComments()) {
                collector.add(comment.getContent().trim());
            }
        }

        @Override
        public void visit(ClassOrInterfaceDeclaration cid, Set<String> collector) {
            super.visit(cid, collector);
            if (cid.getComment().isPresent()) {
                collector.add(cid.getComment().get().getContent().trim());
            }
        }

        @Override
        public void visit(MethodDeclaration md, Set<String> collector) {
            super.visit(md, collector);
            if (md.getComment().isPresent()) {
                collector.add(md.getComment().get().getContent().trim());
            }
            md.getBody().ifPresent(body -> {
                for (Comment comment : body.getAllContainedComments()) {
                    collector.add(comment.getContent().trim());
                }
            });
        }

        @Override
        public void visit(FieldDeclaration fd, Set<String> collector) {
            super.visit(fd, collector);
            if (fd.getComment().isPresent()) {
                collector.add(fd.getComment().get().getContent().trim());
            }
        }

        @Override
        public void visit(ConstructorDeclaration cd, Set<String> collector) {
            super.visit(cd, collector);
            if (cd.getComment().isPresent()) {
                collector.add(cd.getComment().get().getContent().trim());
            }
            cd.getBody().getAllContainedComments().forEach(comment -> {
                collector.add(comment.getContent().trim());
            });
        }

        @Override
        public void visit(EnumDeclaration ed, Set<String> collector) {
            super.visit(ed, collector);
            if (ed.getComment().isPresent()) {
                collector.add(ed.getComment().get().getContent().trim());
            }
        }

        @Override
        public void visit(BlockStmt bs, Set<String> collector) {
            super.visit(bs, collector);
            for (Comment comment : bs.getAllContainedComments()) {
                collector.add(comment.getContent().trim());
            }
        }
    }

    // Visitor class to collect string literals from the AST
    private static class StringLiteralCollector extends VoidVisitorAdapter<Set<String>> {
        @Override
        public void visit(StringLiteralExpr sle, Set<String> collector) {
            super.visit(sle, collector);
            String value = sle.getValue().replace("\\", "").replace("'", "").trim();
            if (!value.isEmpty() && !value.equals(" ")) {
                collector.add(value);
            }
        }
    }
}



