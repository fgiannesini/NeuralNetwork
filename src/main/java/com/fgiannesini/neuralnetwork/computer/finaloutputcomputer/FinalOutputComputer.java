package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

public class FinalOutputComputer implements IFinalOutputComputer {

    private final NeuralNetworkModel<WeightBiasLayer> model;

    public FinalOutputComputer(NeuralNetworkModel model) {
        this.model = model;
    }

    @Override
    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup();
        for (WeightBiasLayer layer : model.getLayers()) {
            currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
