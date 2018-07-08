package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class FinalOutputComputer {

    private final NeuralNetworkModel model;

    FinalOutputComputer(NeuralNetworkModel model) {
        this.model = model;
    }

    public double[] compute(double[] input) {
        return compute(new DoubleMatrix(input)).toArray();
    }

    public double[][] compute(double[][] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input).transpose();
        return compute(inputMatrix).transpose().toArray2();
    }

    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup();
        for (Layer layer : model.getLayers()) {
            currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
        }
        return currentMatrix;
    }
}
