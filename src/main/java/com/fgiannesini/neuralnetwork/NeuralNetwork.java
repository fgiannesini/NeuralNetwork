package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.batch.BatchIterator;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.converter.DataFormatConverter;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningrate.ILearningRateUpdater;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.normalizer.INormalizer;
import org.jblas.DoubleMatrix;

import java.util.function.Consumer;

public class NeuralNetwork {

    private final LearningAlgorithm learningAlgorithm;
    private final INormalizer normalizer;
    private final CostType costType;
    private final Consumer<NeuralNetworkStats> statsUpdateAction;
    private NeuralNetworkModel neuralNetworkModel;
    private final int epochCount;
    private final ILearningRateUpdater learningRateUpdater;
    private final int batchSize;

    NeuralNetwork(LearningAlgorithm learningAlgorithm, INormalizer normalizer, CostType costType, Consumer<NeuralNetworkStats> statsUpdateAction, HyperParameters hyperParameters) {
        this.learningAlgorithm = learningAlgorithm;
        this.normalizer = normalizer;
        this.costType = costType;
        this.statsUpdateAction = statsUpdateAction;
        batchSize = hyperParameters.getBatchSize();
        this.epochCount = hyperParameters.getEpochCount();
        this.learningRateUpdater = hyperParameters.getLearningRateUpdater();
    }

    void learn(double[] input, double[] expected, double[] testInput, double[] testExpected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromTabToDoubleMatrix(input);
        DoubleMatrix outputMatrix = DataFormatConverter.fromTabToDoubleMatrix(expected);
        DoubleMatrix testInputMatrix = DataFormatConverter.fromTabToDoubleMatrix(testInput);
        DoubleMatrix testExpectedMatrix = DataFormatConverter.fromTabToDoubleMatrix(testExpected);
        learn(inputMatrix, outputMatrix, testInputMatrix, testExpectedMatrix);
    }

    void learn(double[][] input, double[][] expected, double[][] testInput, double[][] testExpected) {
        DoubleMatrix inputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(input);
        DoubleMatrix outputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(expected);
        DoubleMatrix testInputMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(testInput);
        DoubleMatrix testExpectedMatrix = DataFormatConverter.fromDoubleTabToDoubleMatrix(testExpected);
        learn(inputMatrix, outputMatrix, testInputMatrix, testExpectedMatrix);
    }

    public void learn(DoubleMatrix input, DoubleMatrix outpout, DoubleMatrix testInput, DoubleMatrix testOutpout) {
        DoubleMatrix normalizedInput = normalizer.normalize(input);
        DoubleMatrix normalizedOutput = normalizer.normalize(outpout);
        DoubleMatrix normalizedTestInput = normalizer.normalize(testInput);
        DoubleMatrix normalizedTestOutput = normalizer.normalize(testOutpout);

        for (int epochNumber = 0; epochNumber < epochCount; epochNumber++) {
            learningAlgorithm.updateLearningRate(learningRateUpdater.get(epochNumber));
            for (BatchIterator batchIterator = BatchIterator.init(normalizedInput, normalizedOutput, batchSize); batchIterator.hasNext(); batchIterator.next()) {
                DoubleMatrix subInput = batchIterator.getSubInput();
                DoubleMatrix subOutput = batchIterator.getSubOutput();

                neuralNetworkModel = learningAlgorithm.learn(subInput, subOutput);
                CostComputer costComputer = CostComputerBuilder.init()
                        .withNeuralNetworkModel(neuralNetworkModel)
                        .withType(costType)
                        .build();
                double learningCost = costComputer.compute(subInput, subOutput);
                double testCost = costComputer.compute(normalizedTestInput, normalizedTestOutput);

                NeuralNetworkStats stats = new NeuralNetworkStats(learningCost, testCost, batchIterator.getBatchNumber(), epochNumber);
                statsUpdateAction.accept(stats);
            }
        }
    }

    public double[] apply(double[] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input);
        return apply(inputMatrix).toArray();
    }

    public double[][] apply(double[][] input) {
        DoubleMatrix inputMatrix = new DoubleMatrix(input);
        return apply(inputMatrix).transpose().toArray2();
    }

    public DoubleMatrix apply(DoubleMatrix input) {
        DoubleMatrix normalizedInput = normalizer.normalize(input);
        return OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer()
                .compute(normalizedInput);
    }
}
