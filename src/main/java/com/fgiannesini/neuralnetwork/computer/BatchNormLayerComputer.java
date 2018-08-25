package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class BatchNormLayerComputer implements ILayerComputer<BatchNormLayer> {

    private static final double epsilon = Math.pow(10, -8);

    public DoubleMatrix computeZFromInput(DoubleMatrix input, BatchNormLayer layer) {
        //Z1 = W.X
        DoubleMatrix z = layer.getWeightMatrix().mmul(input);

        //mean
        DoubleMatrix means = z.rowMeans();

        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(z.subColumnVector(means), 2).rowMeans().addi(epsilon));

        //Z2 = (Z1 - mean) / sigma * gamma + beta
        return z.subColumnVector(means).diviColumnVector(standardDeviation).muliColumnVector(layer.getGammaMatrix()).addiColumnVector(layer.getBetaMatrix());
    }
}
