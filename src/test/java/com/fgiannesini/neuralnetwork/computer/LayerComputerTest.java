package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.FloatMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class LayerComputerTest {

    @Test
    void compute_on_one_weight() {
        Layer layer = new Layer(1, 1, InitializerType.ONES.getInitializer());
        LayerComputer layerComputer = new LayerComputer(layer, ActivationFunctionType.NONE.getActivationFunction());
        FloatMatrix output = layerComputer.compute(FloatMatrix.scalar(3f));
        Assertions.assertArrayEquals(new float[]{4}, output.data);
    }

    @Test
    void compute_on_weight_array() {
        Layer layer = new Layer(5, 1, InitializerType.ONES.getInitializer());
        LayerComputer layerComputer = new LayerComputer(layer, ActivationFunctionType.NONE.getActivationFunction());
        FloatMatrix output = layerComputer.compute(FloatMatrix.ones(5).mul(3f));
        Assertions.assertArrayEquals(new float[]{16}, output.data);
    }

    @Test
    void compute_on_weight_matrix() {
        Layer layer = new Layer(5, 2, InitializerType.ONES.getInitializer());
        LayerComputer layerComputer = new LayerComputer(layer, ActivationFunctionType.NONE.getActivationFunction());
        FloatMatrix output = layerComputer.compute(FloatMatrix.ones(5).mul(3f));
        Assertions.assertArrayEquals(new float[]{16, 16}, output.data);
    }

}