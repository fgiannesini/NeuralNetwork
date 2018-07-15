package com.fgiannesini.neuralnetwork.normalizer;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class MeanAndDeviationNormalizerTest {
    @Test
    void check_on_vector() {
        DoubleMatrix input = new DoubleMatrix(3, 1, -1000, 0, 1000);
        DoubleMatrix output = NormalizerType.MEAN_AND_DEVIATION.get().normalize(input);
        Assertions.assertArrayEquals(new double[]{0, 0, 0}, output.data);
    }

    @Test
    void check_on_matrix() {
        DoubleMatrix input = new DoubleMatrix(3, 2, -1000, 0, 1000, -2000, -1000, 0);
        DoubleMatrix output = NormalizerType.MEAN_AND_DEVIATION.get().normalize(input);
        Assertions.assertArrayEquals(new double[]{0.3163, 0.7072, 0.7072, -0.3163, -0.7072, -0.7072}, output.data, 0.0001);
    }
}