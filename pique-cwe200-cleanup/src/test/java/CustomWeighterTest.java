import calibration.CWE200CustomWeighter;
import org.junit.Test;
import pique.model.QualityModel;
import pique.model.QualityModelImport;

import java.nio.file.Paths;

public class CustomWeighterTest {

    private final String qualityModelPath = "src/main/resources/cwe-200-codeql.json";

    public CustomWeighterTest(){
    }

    @Test
    public void init(){
        CWE200CustomWeighter cwe200CustomWeighter = new CWE200CustomWeighter();
        QualityModelImport qmImport = new QualityModelImport(Paths.get(qualityModelPath));
        QualityModel qmDescription = qmImport.importQualityModel();
        cwe200CustomWeighter.elicitateWeights(qmDescription);
    }
}
