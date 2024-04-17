import java
import SensitiveInfo.SensitiveInfo // Import from the correct module path
// from SensitiveVariables::SensitiveVariableExpr sve
// select sve


// from SensitiveStrings::SensitiveStringLiteral ssl
// select ssl



class HardCodedSensitiveComments extends Javadoc {
  HardCodedSensitiveComments() {
    exists(SensitiveComments::SensitiveComment sc | 
      this.getAChild().(JavadocText).getText().matches("%" + sc.getValue() + "%")
    )
  }
}

from HardCodedSensitiveComments comment
select comment, "This comment may contain hardcoded sensitive information."