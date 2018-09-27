package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;

public class MeanAndDeviationNormalizer implements INormalizer {

    private MeanDeviation meanDeviation;
    private MeanDeviationProvider meanDeviationProvider;

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

    public void setMeanDeviationProvider(MeanDeviationProvider meanDeviationProvider) {
        this.meanDeviationProvider = meanDeviationProvider;
    }


}
