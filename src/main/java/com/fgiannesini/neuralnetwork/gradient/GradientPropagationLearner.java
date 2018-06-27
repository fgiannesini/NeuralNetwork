package com.fgiannesini.neuralnetwork.gradient;

import com.fgiannesini.neuralnetwork.computer.OutputComputer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.FloatMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientPropagationLearner {
    private NeuralNetworkModel neuralNetworkModel;
    private OutputComputer outputComputer;

    GradientPropagationLearner(NeuralNetworkModel neuralNetworkModel, OutputComputer outputComputer) {
        this.neuralNetworkModel = neuralNetworkModel.clone();
        this.outputComputer = outputComputer;
    }

    public NeuralNetworkModel learn(float[] input, float[] expected) {
        FloatMatrix inputMatrix = new FloatMatrix(input);
        FloatMatrix y = new FloatMatrix(expected);
        return learn(inputMatrix, y);
    }

    public NeuralNetworkModel learn(float[][] input, float[][] expected) {
        FloatMatrix inputMatrix = new FloatMatrix(input);
        FloatMatrix y = new FloatMatrix(expected);
        return learn(inputMatrix, y);
    }

    NeuralNetworkModel learn(FloatMatrix inputMatrix, FloatMatrix y) {
        List<FloatMatrix> layerResults = getLayerResultMatrices(inputMatrix);
        FloatMatrix a = layerResults.get(0);
        FloatMatrix dz = a.subi(y);
        FloatMatrix dW = dz.mmul(layerResults.get(1).transpose());
        FloatMatrix db = dz;
//        dz = dW.transpose().mmuli(dz).muli()
        return neuralNetworkModel;
    }

    private List<FloatMatrix> getLayerResultMatrices(FloatMatrix input) {
        List<FloatMatrix> layerResults = new ArrayList<>();
        Collections.reverse(layerResults);
        return layerResults;
    }
}
