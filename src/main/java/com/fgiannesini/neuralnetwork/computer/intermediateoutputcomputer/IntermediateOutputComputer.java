package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerComputerVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputer implements IIntermediateOutputComputer {

    private final NeuralNetworkModel model;

    public IntermediateOutputComputer(NeuralNetworkModel model) {
        this.model = model;
    }

    public List<IntermediateOutputResult> compute(LayerTypeData data) {
        List<IntermediateOutputResult> intermediateOutputResults = new ArrayList<>();
        DataFunctionApplier dataFunctionApplier = new DataFunctionApplier(matrix -> matrix.dup());
        LayerTypeData firstData = data.accept(dataFunctionApplier);
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstData);
        intermediateOutputResults.add(intermediateOutputResult);
        for (Layer layer : model.getLayers()) {
            LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(intermediateOutputResult.getResult());
            layer.accept(layerComputerVisitor);
            intermediateOutputResult = layerComputerVisitor.getIntermediateOutputResult();
            intermediateOutputResults.add(intermediateOutputResult);
        }
        return intermediateOutputResults;
    }

}
