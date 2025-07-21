import python
import semmle.python.dataflow.new.TaintTracking
import semmle.python.dataflow.new.DataFlow


// Define the extensible predicates
extensible predicate sensitiveVariables(string fileName, string variableName);
extensible predicate sensitiveStrings(string fileName, string variableName);
extensible predicate sensitiveComments(string fileName, string variableName);
extensible predicate sinks(string fileName, string sinkName, string sinkType); 
  
class SensitiveVariableExpr extends Expr {
  SensitiveVariableExpr() {
    exists(Name name, File file |
      (
        // Handle variable name references
        this = name and
        file = name.getLocation().getFile() and
        sensitiveVariables(file.getBaseName(), name.getId()) 
      ) and
      (
        // Exclude variables or fields with generic, non-sensitive names
        name.getId().toLowerCase() != "message" and
        name.getId().toLowerCase() != "messages" and
        name.getId().toLowerCase() != "msg" and
        name.getId().toLowerCase() != "msgs" and
        name.getId().toLowerCase() != "text" and
        name.getId().toLowerCase() != "texts" and
        name.getId().toLowerCase() != "data" and
        name.getId().toLowerCase() != "body" and
        name.getId().toLowerCase() != "request" and
        name.getId().toLowerCase() != "req" and
        name.getId().toLowerCase() != "context" and
        name.getId().toLowerCase() != "contents" and
        name.getId().toLowerCase() != "content" and
        name.getId().toLowerCase() != "id" and
        name.getId().toLowerCase() != "operation" and
        name.getId().toLowerCase() != "op" and
        name.getId().toLowerCase() != "value" and
        name.getId().toLowerCase() != "val" and
        name.getId().toLowerCase() != "parent" and
        name.getId().toLowerCase() != "parents" and
        name.getId().toLowerCase() != "child" and
        name.getId().toLowerCase() != "children" and
        name.getId().toLowerCase() != "xml" and
        name.getId().toLowerCase() != "json" and
        name.getId().toLowerCase() != "html" and
        name.getId().toLowerCase() != "entity" and
        name.getId().toLowerCase() != "entities" and
        not name.getId().toLowerCase().matches("%name%") and
        not name.getId().toLowerCase().matches("%project%") and
        not name.getId().toLowerCase().matches("%id%") and
        not name.getId().toLowerCase().matches("%location%") and
        not name.getId().toLowerCase().matches("%node%") and
        not name.getId().toLowerCase().matches("%subject%") and
        not name.getId().toLowerCase().matches("%object%") and 
        not name.getId().toLowerCase().matches("%script%") 
        // not name.getId().toLowerCase().matches("%path%") and
      ) 
    )
  }
}


class SensitiveStringLiteral extends StrConst {
  SensitiveStringLiteral() {
      // Check for matches against the suspicious patterns
      exists(File f | 
          f = this.getLocation().getFile() and
          sensitiveStrings(f.getBaseName(), this.getText())) and
      not (
          // Exclude common non-sensitive patterns
          this.getText().regexpMatch(".*example.*") or
          this.getText().regexpMatch(".*test.*") or
          this.getText().regexpMatch(".*demo.*") or
          this.getText().regexpMatch(".*foo.*") or 
          this.getText().regexpMatch(".*bar.*") or
          this.getText().regexpMatch(".*baz.*") or
          this.getText().regexpMatch(".*secret.*") or
          // Exclude empty strings
          this.getText() = "" or
          // Exclude whitespace-only strings
          this.getText().regexpMatch("^\\s*$") or
          // Exclude strings with exactly one dot followed by a digit
          this.getText().regexpMatch("^[^.]*\\.[0-9]+$") 
      )
  }
}


class SensitiveComment extends StrConst {
  SensitiveComment() {
    exists(File f, string pattern |
      f = this.getLocation().getFile() and
      sensitiveComments(f.getBaseName(), pattern) and
      this.getText().regexpMatch(pattern)
    )
  }   
}

  
    predicate getSink(DataFlow::Node sink, string sinkType) { 
      exists(File f, Call call, string sinkName | 
        f = call.getLocation().getFile() and
        (
          call.getFunc().(Name).getId() = sinkName or
          call.getFunc().(Attribute).getName() = sinkName
        ) and
        sinks(f.getBaseName(), sinkName, sinkType) and
        sink.asExpr() = call.getAnArg()
      )
    }

    predicate getSinkAny(DataFlow::Node sink) { 
      exists(File f, Call call, string sinkName | 
        f = call.getLocation().getFile() and
        (
          call.getFunc().(Name).getId() = sinkName or
          call.getFunc().(Attribute).getName() = sinkName
        ) and
        (sinks(f.getBaseName(), sinkName, _) and
        not sinks(f.getBaseName(), sinkName, "Log Sink")) and
        sink.asExpr() = call.getAnArg()
      )
    }

    
class DetectedCall extends Call {
  DetectedCall() {
    // Check for matches against the sinks
    exists(File f, string funcName |
      f = this.getLocation().getFile() and
      (
        this.getFunc().(Name).getId() = funcName or
        this.getFunc().(Attribute).getName() = funcName
      ) and
      sinks(f.getBaseName(), funcName, _)
    )
  }
}


string getSinkName() { 
  exists(File f, Call call, string funcName | 
    f = call.getLocation().getFile() and
    (
      call.getFunc().(Name).getId() = funcName or
      call.getFunc().(Attribute).getName() = funcName
    ) and
    (sinks(f.getBaseName(), funcName, "I/O Sink") or
    sinks(f.getBaseName(), funcName, "Print Sink") or
    sinks(f.getBaseName(), funcName, "Network Sink") or
    sinks(f.getBaseName(), funcName, "Log Sink") or
    sinks(f.getBaseName(), funcName, "Database Sink") or
    sinks(f.getBaseName(), funcName, "Email Sink") or
    sinks(f.getBaseName(), funcName, "IPC Sink") or
    sinks(f.getBaseName(), funcName, "Clipboard Sink") or
    sinks(f.getBaseName(), funcName, "GUI Display Sink") or
    sinks(f.getBaseName(), funcName, "RPC Sink") or
    sinks(f.getBaseName(), funcName, "Environment Variable Sink") or
    sinks(f.getBaseName(), funcName, "Command Execution Sink") or
    sinks(f.getBaseName(), funcName, "Configuration File Sink"))
    // Bind the result to the function name
    and result = funcName
  )
}