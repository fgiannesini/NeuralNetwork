package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.LayerComputerHelper;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.WeightBiasLayer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputerWithDropOutRegularization implements IIntermediateOutputComputer {

    private final NeuralNetworkModel<WeightBiasLayer> model;
    private final List<DoubleMatrix> dropOutMatrixList;

    public IntermediateOutputComputerWithDropOutRegularization(NeuralNetworkModel<WeightBiasLayer> model, List<DoubleMatrix> dropOutMatrixList) {
        this.model = model;
        this.dropOutMatrixList = dropOutMatrixList;
    }

    public List<DoubleMatrix> compute(DoubleMatrix inputMatrix) {
        List<DoubleMatrix> intermediateMatrix = new ArrayList<>();
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        intermediateMatrix.add(currentMatrix);
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < model.getLayers().size(); layerIndex++, dropOutIndex++) {
            WeightBiasLayer layer = model.getLayers().get(layerIndex);
            currentMatrix = LayerComputerHelper.computeZFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = LayerComputerHelper.computeAFromZ(currentMatrix, layer);
            intermediateMatrix.add(currentMatrix);
        }
        return intermediateMatrix;
    }

}
