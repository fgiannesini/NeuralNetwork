package com.fgiannesini.neuralnetwork.computer;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MeanDeviationProvider implements DataVisitor {

    private final double epsilon = Math.pow(10, -8);
    private MeanDeviation meanDeviation;

    @Override
    public void visit(WeightBiasData data) {
        //mean
        DoubleMatrix means = data.getData().rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(data.getData().subColumnVector(means), 2).rowMeans()).addi(epsilon);
        meanDeviation = new WeightBiasMeanDeviation(means, standardDeviation);
    }

    @Override
    public void visit(BatchNormData data) {
        //mean
        DoubleMatrix means = data.getInput().rowMeans();
        //sigma
        DoubleMatrix standardDeviation = MatrixFunctions.sqrt(MatrixFunctions.pow(data.getInput().subColumnVector(means), 2).rowMeans()).addi(epsilon);
        meanDeviation = new BatchNormMeanDeviation(means, standardDeviation);
    }

    public MeanDeviation getMeanDeviation() {
        return meanDeviation;
    }
}
