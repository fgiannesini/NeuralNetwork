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
        Assertions.assertArrayEquals(new double[]{1, 1, 1, -1, -1, -1}, output.data, 0.0001);
    }

    @Test
    void check_keep_normalization_params_on_matrices() {
        INormalizer normalizer = NormalizerType.MEAN_AND_DEVIATION.get();
        DoubleMatrix input1 = new DoubleMatrix(3, 2, -1000, 0, 1000, -2000, -1000, 0);
        DoubleMatrix output1 = normalizer.normalize(input1);
        Assertions.assertArrayEquals(new double[]{1, 1, 1, -1, -1, -1}, output1.data, 0.0001);

        DoubleMatrix input2 = new DoubleMatrix(3, 2, -10000, 200, 10000, -20000, -2000, 0);
        DoubleMatrix output2 = normalizer.normalize(input2);
        Assertions.assertArrayEquals(new double[]{-17, 1.4, 19, -37, -3, -1}, output2.data, 0.0001);
    }
}