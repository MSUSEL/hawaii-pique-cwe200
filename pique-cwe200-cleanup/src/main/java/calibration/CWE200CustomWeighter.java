package calibration;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pique.calibration.IWeighter;
import pique.calibration.WeightResult;
import pique.model.ModelNode;
import pique.model.QualityModel;
import runnable.SingleProjectEvaluator;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class CWE200CustomWeighter implements IWeighter {

    private static final Logger LOGGER = LoggerFactory.getLogger(SingleProjectEvaluator.class);

    private String weightsFile = "resources/weights_definition.json";

    @Override
    public Set<WeightResult> elicitateWeights(QualityModel qualityModel, Path... externalInput) {

        List<ModelNode> qualityAspects = qualityModel.getQualityAspects().values();
        List<ModelNode> productFactors = qualityModel.getProductFactors().values();

        if (externalInput.length == 1){
            //if a different path is supplied as an argument in this function, use it and not the one supplied by default
            weightsFile = externalInput[0].toString();
        }else if (externalInput.length > 1){
            //several different paths are supplied as input, this behavior is not expected and will require additional development
            LOGGER.warn("several different paths are supplied as input for the weighting function, " +
                    "this behavior is not expected and will require additional development");
        }

        return null;
    }

    private Map<String, List<String>> parseWeightFile(){
        Map<String, List<String>> parsedWeights = new HashMap<>();




        return parsedWeights;
    }

    @Override
    public String getName() {
        return "CWE200CustomWeighter";
    }
}
