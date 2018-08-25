package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class WeightBiasLayerComputerTest {

    @Test
    void compute_on_one_weight_and_weight_bias() {
        WeightBiasLayer layer = new WeightBiasLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = new WeighBiasLayerComputer().computeAFromInput(DoubleMatrix.scalar(3f), layer);
        Assertions.assertArrayEquals(new double[]{4}, output.data);
    }

    @Test
    void compute_on_weight_array_and_weight_bias() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = new WeighBiasLayerComputer().computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{16}, output.data);
    }

    @Test
    void compute_on_weight_matrix_and_weight_bias() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = new WeighBiasLayerComputer().computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{16, 16}, output.data);
    }

}