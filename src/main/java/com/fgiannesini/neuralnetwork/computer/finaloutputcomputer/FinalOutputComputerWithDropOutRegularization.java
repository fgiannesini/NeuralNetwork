package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization implements IFinalOutputComputer {

    private final NeuralNetworkModel model;
    private List<DoubleMatrix> dropOutMatrixList;

    public FinalOutputComputerWithDropOutRegularization(NeuralNetworkModel model, List<DoubleMatrix> dropOutMatrixList) {
        this.model = model;
        this.dropOutMatrixList = dropOutMatrixList;
    }

    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < model.getLayers().size(); layerIndex++, dropOutIndex++) {
            Layer layer = model.getLayers().get(layerIndex);
            currentMatrix = LayerComputerHelper.computeAFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
        }
        return currentMatrix;
    }

}
