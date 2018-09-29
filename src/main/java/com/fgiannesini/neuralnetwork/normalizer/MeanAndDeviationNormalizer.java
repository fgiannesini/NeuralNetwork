package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;

public class MeanAndDeviationNormalizer implements INormalizer {

    private MeanDeviation meanDeviation;
    private final MeanDeviationProvider meanDeviationProvider;

    public MeanAndDeviationNormalizer(MeanDeviationProvider meanDeviationProvider) {
        this.meanDeviationProvider = meanDeviationProvider;
    }

    @Override
    public LayerTypeData normalize(LayerTypeData input) {

        if (meanDeviation == null) {
            input.accept(meanDeviationProvider);
            meanDeviation = meanDeviationProvider.getMeanDeviation();
        }

        MeanAndDeviationNormalizerVisitor meanAndDeviationNormalizerVisitor = new MeanAndDeviationNormalizerVisitor(meanDeviation);
        input.accept(meanAndDeviationNormalizerVisitor);
        return meanAndDeviationNormalizerVisitor.getNormalizedData();
    }
}
