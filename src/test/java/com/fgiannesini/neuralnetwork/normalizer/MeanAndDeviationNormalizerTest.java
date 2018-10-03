package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class MeanAndDeviationNormalizerTest {

    @Test
    void check_on_vector() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 1, -1000, 0, 1000));
        WeightBiasData output = (WeightBiasData) NormalizerType.MEAN_AND_DEVIATION.get(new MeanDeviationProvider()).normalize(input);
        Assertions.assertArrayEquals(new double[]{0, 0, 0}, output.getData().data);
    }

    @Test
    void check_on_matrix() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 2, -1000, 0, 1000, -2000, -1000, 0));
        WeightBiasData output = (WeightBiasData) NormalizerType.MEAN_AND_DEVIATION.get(new MeanDeviationProvider()).normalize(input);
        Assertions.assertArrayEquals(new double[]{1, 1, 1, -1, -1, -1}, output.getData().data, 0.0001);
    }

    @Test
    void check_keep_normalization_params_on_matrices() {
        INormalizer normalizer = NormalizerType.MEAN_AND_DEVIATION.get(new MeanDeviationProvider());
        WeightBiasData input1 = new WeightBiasData(new DoubleMatrix(3, 2, -1000, 0, 1000, -2000, -1000, 0));
        WeightBiasData output1 = (WeightBiasData) normalizer.normalize(input1);
        Assertions.assertArrayEquals(new double[]{1, 1, 1, -1, -1, -1}, output1.getData().data, 0.0001);

        WeightBiasData input2 = new WeightBiasData(new DoubleMatrix(3, 2, -10000, 200, 10000, -20000, -2000, 0));
        WeightBiasData output2 = (WeightBiasData) normalizer.normalize(input2);
        Assertions.assertArrayEquals(new double[]{-17, 1.4, 19, -37, -3, -1}, output2.getData().data, 0.0001);
    }
}