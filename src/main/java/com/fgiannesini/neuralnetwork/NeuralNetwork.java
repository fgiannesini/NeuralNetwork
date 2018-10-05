package com.fgiannesini.neuralnetwork;

import com.fgiannesini.neuralnetwork.batch.BatchIterator;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningrate.ILearningRateUpdater;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.normalizer.INormalizer;

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

    public void learn(LayerTypeData input, LayerTypeData outpout, LayerTypeData testInput, LayerTypeData testOutpout) {
        LayerTypeData normalizedInput = normalizer.normalize(input);
        LayerTypeData normalizedTestInput = normalizer.normalize(testInput);

        for (int epochNumber = 0; epochNumber < epochCount; epochNumber++) {
            learningAlgorithm.updateLearningRate(learningRateUpdater.get(epochNumber));
            for (BatchIterator batchIterator = BatchIterator.init(normalizedInput, outpout, batchSize); batchIterator.hasNext(); batchIterator.next()) {
                LayerTypeData subInput = batchIterator.getSubInput();
                LayerTypeData subOutput = batchIterator.getSubOutput();

                neuralNetworkModel = learningAlgorithm.learn(subInput, subOutput);
                CostComputer costComputer = CostComputerBuilder.init()
                        .withNeuralNetworkModel(neuralNetworkModel)
                        .withType(costType)
                        .build();
                double learningCost = costComputer.compute(subInput, subOutput);

                CostComputer testCostComputer = CostComputerBuilder.init()
                        .withNeuralNetworkModel(neuralNetworkModel)
                        .withType(costType)
                        .build();
                double testCost = testCostComputer.compute(normalizedTestInput, testOutpout);

                NeuralNetworkStats stats = new NeuralNetworkStats(learningCost, testCost, batchIterator.getBatchNumber(), epochNumber);
                statsUpdateAction.accept(stats);
            }
        }
    }

    public LayerTypeData apply(LayerTypeData input) {
        LayerTypeData normalizedInput = normalizer.normalize(input);
        return OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer()
                .compute(normalizedInput);
    }
}
