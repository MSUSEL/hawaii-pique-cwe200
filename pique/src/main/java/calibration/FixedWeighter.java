package calibration;

import pique.model.ModelNode;
import pique.model.QualityModel;
import pique.utility.BigDecimalWithContext;
import pique.calibration.IWeighter;
import pique.calibration.WeightResult;

import java.math.BigDecimal;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

public class FixedWeighter implements IWeighter {

    /**
     * For each model node, get the exact weight defined in the JSON configuration.
     * If no weight is defined for a particular child, default to 1.0.
     */
    @Override
    public Set<WeightResult> elicitateWeights(QualityModel qualityModel, Path... externalInput) {
        Set<WeightResult> weights = new HashSet<>();

        // iterate over all nodes in the quality model
        qualityModel.getAllQualityModelNodes().values().forEach(node -> {
            WeightResult weightResult = new WeightResult(node.getName());
            node.getChildren().values().forEach(child -> {
                // Use the fixed weight from the configuration if available,
                // falling back to 1.0 if not defined.
                BigDecimal weight = node.getWeights().getOrDefault(child.getName(), new BigDecimalWithContext("1.0"));
                weightResult.setWeight(child.getName(), weight);
            });
            weights.add(weightResult);
        });
        return weights;
    }

    @Override
    public String getName() {
        return this.getClass().getCanonicalName();
    }
}