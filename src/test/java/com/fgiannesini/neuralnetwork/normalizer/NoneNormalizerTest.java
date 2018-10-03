package com.fgiannesini.neuralnetwork.normalizer;

import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class NoneNormalizerTest {

    @Test
    void check_on_vector() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 1, -1000, 0, 1000));
        WeightBiasData output = (WeightBiasData) NormalizerType.NONE.get().normalize(input);
        Assertions.assertArrayEquals(input.getData().data, output.getData().data);
    }

    @Test
    void check_on_matrix() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 2, -1000, 0, 1000, -2000, -1000, 0));
        WeightBiasData output = (WeightBiasData) NormalizerType.NONE.get().normalize(input);
        Assertions.assertArrayEquals(input.getData().data, output.getData().data);
    }
}