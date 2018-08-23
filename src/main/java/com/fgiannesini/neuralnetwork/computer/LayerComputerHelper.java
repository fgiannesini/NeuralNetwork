package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionApplier;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class LayerComputerHelper {

    private static double epsilon;

    public LayerComputerHelper() {
        epsilon = Math.pow(10, -8);
    }

    public static DoubleMatrix computeAFromZ(DoubleMatrix z, Layer layer) {
        ActivationFunctionApplier activationFunctionApplier = layer.getActivationFunctionType().getActivationFunction();
        return activationFunctionApplier.apply(z);
    }

    public static DoubleMatrix computeZFromInput(DoubleMatrix input, WeightBiasLayer layer) {
        //W.X + b
        return layer.getWeightMatrix().mmul(input).addiColumnVector(layer.getBiasMatrix());
    }

    public static DoubleMatrix computeAFromInput(DoubleMatrix input, WeightBiasLayer layer) {
        DoubleMatrix z = computeZFromInput(input, layer);
        return computeAFromZ(z, layer);
    }

    public static DoubleMatrix computeAFromInput(DoubleMatrix input, BatchNormLayer layer) {
        DoubleMatrix z = computeZFromInput(input, layer);
        return computeAFromZ(z, layer);
    }

    public static DoubleMatrix computeZFromInput(DoubleMatrix input, BatchNormLayer layer) {
        //Z1 = W.X
        DoubleMatrix z = layer.getWeightMatrix().mmul(input);

        //mean
        DoubleMatrix means = input.rowMeans();

        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(input.subColumnVector(means), 2).rowMeans().addi(epsilon));

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        return z.subColumnVector(means).diviColumnVector(standardDeviation).muliColumnVector(layer.getGammaMatrix()).addi(layer.getBetaMatrix());
    }
}
