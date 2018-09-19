package com.fgiannesini.neuralnetwork.model;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import org.jblas.DoubleMatrix;

import java.util.List;

public abstract class Layer implements Cloneable {
    private ActivationFunctionType activationFunctionType;

    Layer(ActivationFunctionType activationFunctionType) {
        this.activationFunctionType = activationFunctionType;
    }

    public ActivationFunctionType getActivationFunctionType() {
        return activationFunctionType;
    }

    public List<DoubleMatrix> getParametersMatrix() {
        return null;
    }

    @Override
    public Layer clone() {
        try {
            return (Layer) super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

}
