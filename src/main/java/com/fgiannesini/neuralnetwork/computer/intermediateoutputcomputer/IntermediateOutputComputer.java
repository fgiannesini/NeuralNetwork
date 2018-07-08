package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class IntermediateOutputComputer implements IIntermediateOutputComputer {

    private final NeuralNetworkModel model;

    public IntermediateOutputComputer(NeuralNetworkModel model) {
        this.model = model;
    }

    public List<double[]> compute(double[] input) {
        return compute(new DoubleMatrix(input))
                .stream()
                .map(result -> result.toArray())
                .collect(Collectors.toList());
    }

    public List<double[][]> compute(double[][] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input).transpose();
        return compute(inputMatrix)
                .stream()
                .map(result -> result.transpose().toArray2())
                .collect(Collectors.toList());
    }

    public List<DoubleMatrix> compute(DoubleMatrix inputMatrix) {
        List<DoubleMatrix> intermediateMatrix = new ArrayList<>();
        DoubleMatrix currentMatrix = inputMatrix.dup();
        intermediateMatrix.add(currentMatrix);
        for (Layer layer : model.getLayers()) {
            currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
            intermediateMatrix.add(currentMatrix);
        }
        return intermediateMatrix;
    }

}
