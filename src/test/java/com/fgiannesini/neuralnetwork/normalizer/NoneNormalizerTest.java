package com.fgiannesini.neuralnetwork.normalizer;

import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class NoneNormalizerTest {

    @Test
    void check_on_vector() {
        DoubleMatrix input = new DoubleMatrix(3, 1, -1000, 0, 1000);
        DoubleMatrix output = NormalizerType.NONE.get().normalize(input);
        Assertions.assertArrayEquals(input.data, output.data);
    }

    @Test
    void check_on_matrix() {
        DoubleMatrix input = new DoubleMatrix(3, 2, -1000, 0, 1000, -2000, -1000, 0);
        DoubleMatrix output = NormalizerType.NONE.get().normalize(input);
        Assertions.assertArrayEquals(input.data, output.data);
    }
}