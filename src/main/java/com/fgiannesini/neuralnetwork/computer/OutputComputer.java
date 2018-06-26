package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class OutputComputer {

    private final List<LayerComputer> layerComputers;

    OutputComputer(NeuralNetworkModel model) {
        layerComputers = model.getLayers().stream()
                .map(LayerComputer::new)
                .collect(Collectors.toList());
    }

    public float[] compute(float[] input) {
        return computeOutput(new FloatMatrix(input)).toArray();
    }

    public float[][] compute(float[][] input) {
        FloatMatrix inputMatrix = new FloatMatrix(input).transpose();
        return computeOutput(inputMatrix).transpose().toArray2();
    }

    public FloatMatrix computeOutput(FloatMatrix inputMatrix) {
        List<FloatMatrix> layerResults = computeLayerResults(inputMatrix);
        return layerResults.get(layerResults.size() - 1);
    }

    public List<FloatMatrix> computeLayerResults(FloatMatrix inputMatrix) {
        FloatMatrix currentMatrix = inputMatrix.dup();
        List<FloatMatrix> matrixResults = new ArrayList<>();
        matrixResults.add(currentMatrix);
        for (LayerComputer layerComputer : layerComputers) {
            currentMatrix = layerComputer.compute(currentMatrix);
            matrixResults.add(currentMatrix);
        }
        return matrixResults;
    }
}
