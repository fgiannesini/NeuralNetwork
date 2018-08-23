package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputer implements IIntermediateOutputComputer {

    private final NeuralNetworkModel<WeightBiasLayer> model;

    public IntermediateOutputComputer(NeuralNetworkModel<WeightBiasLayer> model) {
        this.model = model;
    }

    public List<DoubleMatrix> compute(DoubleMatrix inputMatrix) {
        List<DoubleMatrix> intermediateMatrix = new ArrayList<>();
        DoubleMatrix currentMatrix = inputMatrix.dup();
        intermediateMatrix.add(currentMatrix);
        for (WeightBiasLayer layer : model.getLayers()) {
            currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
            intermediateMatrix.add(currentMatrix);
        }
        return intermediateMatrix;
    }

}
