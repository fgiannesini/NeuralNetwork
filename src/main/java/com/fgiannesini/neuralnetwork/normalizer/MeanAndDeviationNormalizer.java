package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.MeanDeviation;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import org.jblas.DoubleMatrix;

public class MeanAndDeviationNormalizer implements INormalizer {

    private MeanDeviation meanDeviation;

    @Override
    public DoubleMatrix normalize(DoubleMatrix input) {
        if (meanDeviation == null) {
            meanDeviation = new MeanDeviationProvider().get(input);
        }
        //(x-mu)/sigma
        return input.subColumnVector(meanDeviation.getMean()).diviColumnVector(meanDeviation.getDeviation());
    }


}
