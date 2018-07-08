package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class FinalOutputComputer implements IFinalOutputComputer {

    private final NeuralNetworkModel model;

    public FinalOutputComputer(NeuralNetworkModel model) {
        this.model = model;
    }

    @Override
    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup();
        for (Layer layer : model.getLayers()) {
            currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
