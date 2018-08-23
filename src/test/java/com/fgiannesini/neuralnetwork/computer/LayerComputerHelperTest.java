package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LayerComputerHelperTest {

    @Test
    void compute_on_one_weight() {
        WeightBiasLayer layer = new WeightBiasLayer(1, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.scalar(3f), layer);
        Assertions.assertArrayEquals(new double[]{4}, output.data);
    }

    @Test
    void compute_on_weight_array() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 1, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{16}, output.data);
    }

    @Test
    void compute_on_weight_matrix() {
        WeightBiasLayer layer = new WeightBiasLayer(5, 2, InitializerType.ONES.getInitializer(), ActivationFunctionType.NONE);
        DoubleMatrix output = LayerComputerHelper.computeAFromInput(DoubleMatrix.ones(5).mul(3f), layer);
        Assertions.assertArrayEquals(new double[]{16, 16}, output.data);
    }

}