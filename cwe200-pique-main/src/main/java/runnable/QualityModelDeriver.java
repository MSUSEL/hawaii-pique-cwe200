package runnable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pique.analysis.ITool;
import pique.model.QualityModel;
import pique.model.QualityModelExport;
import pique.model.QualityModelImport;
import pique.runnable.AQualityModelDeriver;
import pique.utility.PiqueProperties;
import tool.CweCodeQl;


import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class QualityModelDeriver extends AQualityModelDeriver {

    private static final Logger LOGGER = LoggerFactory.getLogger(QualityModelDeriver.class);

    public static void main(String[] args){
        new QualityModelDeriver();
    }

    public QualityModelDeriver(String propertiesPath){
        init(propertiesPath);
    }

    public QualityModelDeriver(){
        init(null);
    }

    private void init(String propertiesPath){
        LOGGER.info("Beginning deriver");

        Properties prop = null;
        try {
            prop = propertiesPath==null ? PiqueProperties.getProperties() : PiqueProperties.getProperties(propertiesPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Path blankqmFilePath = Paths.get(prop.getProperty("blankqm.filepath"));
        Path derivedModelFilePath = Paths.get(prop.getProperty("results.directory"));

        // Initialize objects
        String projectRootFlag = "";
        Path benchmarkRepo = Paths.get(prop.getProperty("benchmark.repo"));

        //ITool gyrpeWrapper = new GrypeWrapper(prop.getProperty("github-token-path"));

        ITool cweCodeQl = new CweCodeQl(prop.getProperty("openai-api-token-path"));
        Set<ITool> tools = Stream.of(cweCodeQl).collect(Collectors.toSet());
        QualityModelImport qmImport = new QualityModelImport(blankqmFilePath);
        QualityModel qmDescription = qmImport.importQualityModel();
        //qmDescription = pique.utility.TreeTrimmingUtility.trimQualityModelTree(qmDescription);

        QualityModel derivedQualityModel = deriveModel(qmDescription, tools, benchmarkRepo, projectRootFlag);

        Path jsonOutput = new QualityModelExport(derivedQualityModel)
                .exportToJson(derivedQualityModel
                        .getName(), derivedModelFilePath);

        LOGGER.info("Quality Model derivation finished. You can find the file at " + jsonOutput.toAbsolutePath().toString());
    }



}
