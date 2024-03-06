import java
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.security.SensitiveActions
import semmle.code.java.controlflow.Guards

// Class for String literals potentially used in observable discrepancies
class SensitiveMessageLiteral extends StringLiteral {
  SensitiveMessageLiteral() {
    this.getValue().regexpMatch(".*Login (Successful|Failed).*|.*Invalid (username|password).*|.*Error.*") // Adjust the regex to match relevant messages
  }
}

from IfStmt outerIf, IfStmt innerIf,SensitiveMessageLiteral str1,SensitiveMessageLiteral str2,SensitiveMessageLiteral str3
where
  // Check if the innerIf is directly within the body of outerIf
  outerIf.getAChild*() = innerIf and
  innerIf.getThen().getBasicBlock().getANode()=str1 and
  innerIf.getElse().getBasicBlock().getANode()=str2 and
  outerIf.getElse().getBasicBlock().getANode()=str3 and
  outerIf != innerIf
  select outerIf.getBasicBlock(), "@ This 'if' statement in a sensitive method has different messages in its branches, potentially indicating observable discrepancies (CWE-204)."
