package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplyer;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.FloatMatrix;

public class LayerComputer {

    private final Layer layer;
    private final ActivationFunctionApplyer activationFunctionApplyer;

    public LayerComputer(Layer layer, ActivationFunctionApplyer activationFunctionApplyer) {
        this.layer = layer;
        this.activationFunctionApplyer = activationFunctionApplyer;
    }

    public FloatMatrix compute(FloatMatrix input) {
        //Wt.X + b
        FloatMatrix z = layer.getWeightMatrix().transpose().mmul(input).addiColumnVector(layer.getBiasMatrix());
        return activationFunctionApplyer.apply(z);
    }

}