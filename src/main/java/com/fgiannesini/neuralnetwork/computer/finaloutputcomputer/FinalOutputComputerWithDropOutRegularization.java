package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputerWithDropOutRegularization implements IFinalOutputComputer {

    private final NeuralNetworkModel<WeightBiasLayer> model;
    private final List<DoubleMatrix> dropOutMatrixList;

    public FinalOutputComputerWithDropOutRegularization(NeuralNetworkModel model, List<DoubleMatrix> dropOutMatrixList) {
        this.model = model;
        this.dropOutMatrixList = dropOutMatrixList;
    }

    public DoubleMatrix compute(DoubleMatrix inputMatrix) {
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < model.getLayers().size(); layerIndex++, dropOutIndex++) {
            WeightBiasLayer layer = model.getLayers().get(layerIndex);
            currentMatrix = LayerComputerHelper.computeZFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = LayerComputerHelper.computeAFromZ(currentMatrix, layer);
        }
        return currentMatrix;
    }

}
