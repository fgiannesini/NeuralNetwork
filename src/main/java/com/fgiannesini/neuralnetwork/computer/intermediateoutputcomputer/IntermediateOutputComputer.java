package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerComputerVisitor;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.adapter.ForwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputer implements IIntermediateOutputComputer {

    private final NeuralNetworkModel model;

    public IntermediateOutputComputer(NeuralNetworkModel model) {
        this.model = model;
    }

    public List<IntermediateOutputResult> compute(LayerTypeData data) {
        List<IntermediateOutputResult> intermediateOutputResults = new ArrayList<>();
        DataFunctionApplier dataFunctionApplier = new DataFunctionApplier(DoubleMatrix::dup);
        data.accept(dataFunctionApplier);

        LayerTypeData firstData = dataFunctionApplier.getLayerTypeData();
        ForwardDataAdapterVisitor firstDataAdaptorVisitor = new ForwardDataAdapterVisitor(firstData);
        model.getLayers().get(0).accept(firstDataAdaptorVisitor);
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstDataAdaptorVisitor.getData());
        intermediateOutputResults.add(intermediateOutputResult);

        for (Layer layer : model.getLayers()) {
            LayerTypeData previousResult = intermediateOutputResult.getResult();
            ForwardDataAdapterVisitor dataAdaptorVisitor = new ForwardDataAdapterVisitor(previousResult);
            layer.accept(dataAdaptorVisitor);
            LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(dataAdaptorVisitor.getData());
            layer.accept(layerComputerVisitor);
            intermediateOutputResult = layerComputerVisitor.getIntermediateOutputResult();
            intermediateOutputResults.add(intermediateOutputResult);
        }
        return intermediateOutputResults;
    }

}
