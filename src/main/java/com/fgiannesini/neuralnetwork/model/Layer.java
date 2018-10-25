package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

import java.io.Serializable;
import java.util.List;

public abstract class Layer implements Cloneable, Serializable {

    private final ActivationFunctionType activationFunctionType;

    Layer(ActivationFunctionType activationFunctionType) {
        this.activationFunctionType = activationFunctionType;
    }

    public ActivationFunctionType getActivationFunctionType() {
        return activationFunctionType;
    }

    public abstract List<DoubleMatrix> getParametersMatrix();

    public abstract void accept(LayerVisitor layerVisitor);

    @Override
    public Layer clone() {
        try {
            return (Layer) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }
}
