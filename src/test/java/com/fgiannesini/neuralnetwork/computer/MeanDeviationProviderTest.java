package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.assertions.DoubleMatrixAssertions;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.WeightBiasMeanDeviation;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Test;

class MeanDeviationProviderTest {

    @Test
    void get() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(2, 4, 0, 1, 2, 3, 4, 5, 6, 7));
        MeanDeviationProvider meanDeviationProvider = new MeanDeviationProvider();
        input.accept(meanDeviationProvider);
        WeightBiasMeanDeviation meanDeviation = (WeightBiasMeanDeviation) meanDeviationProvider.getMeanDeviation();
        DoubleMatrixAssertions.assertMatrices(meanDeviation.getMean(), new DoubleMatrix(2, 1, 3, 4));
        DoubleMatrixAssertions.assertMatrices(meanDeviation.getDeviation(), new DoubleMatrix(2, 1, 2.236068, 2.236068));
    }
}