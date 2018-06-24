package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.FloatMatrix;

class LayerComputer {

    private final Layer layer;
    private final ActivationFunctionApplier activationFunctionApplier;

    public LayerComputer(Layer layer) {
        this.layer = layer;
        activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
    }

    public FloatMatrix compute(FloatMatrix input) {
        //Wt.X + b
        FloatMatrix z = layer.getWeightMatrix().transpose().mmul(input).addiColumnVector(layer.getBiasMatrix());
        return activationFunctionApplier.apply(z);
    }

}
