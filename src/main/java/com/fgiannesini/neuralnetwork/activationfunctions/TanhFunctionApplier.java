package com.fgiannesini.neuralnetwork.activationfunctions;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class TanhFunctionApplier implements ActivationFunctionApplier {

    @Override
    public FloatMatrix apply(FloatMatrix input) {
        return MatrixFunctions.tanh(input);
    }

    @Override
    public FloatMatrix derivate(FloatMatrix input) {
        return FloatMatrix.ones(input.rows,input.columns).addi(MatrixFunctions.pow(input,2));
    }
}
