import org.junit.Test;
import runnable.QualityModelDeriver;

public class DeriverTest {

    public DeriverTest(){

    }

    @Test
    public void runDeriver(){
        QualityModelDeriver deriver = new QualityModelDeriver("src/test/resources/pique-test-properties.properties");

    }
}
