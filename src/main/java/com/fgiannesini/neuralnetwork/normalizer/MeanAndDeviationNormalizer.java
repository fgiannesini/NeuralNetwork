package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;

public class MeanAndDeviationNormalizer implements INormalizer {

    private MeanDeviation meanDeviation;

    @Override
    public LayerTypeData normalize(LayerTypeData input) {

        if (meanDeviation == null) {
            meanDeviation = new MeanDeviationProvider().get(input);
        }

        MeanAndDeviationNormalizerVisitor meanAndDeviationNormalizerVisitor = new MeanAndDeviationNormalizerVisitor(meanDeviation);
        input.accept(meanAndDeviationNormalizerVisitor);
        return meanAndDeviationNormalizerVisitor.getNormalizedData();
    }


}
