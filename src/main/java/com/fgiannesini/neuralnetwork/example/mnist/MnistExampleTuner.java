package com.fgiannesini.neuralnetwork.example.mnist;

import com.fgiannesini.neuralnetwork.HyperParameters;
import com.fgiannesini.neuralnetwork.NeuralNetworkStats;
import com.fgiannesini.neuralnetwork.RegularizationCoeffs;
import com.fgiannesini.neuralnetwork.example.tune.TuneState;
import com.fgiannesini.neuralnetwork.learningrate.ILearningRateUpdater;
import com.fgiannesini.neuralnetwork.learningrate.LearningRateUpdaterType;
import com.fgiannesini.neuralnetwork.model.LayerType;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class MnistExampleTuner {

    private static final int PARAMETER_COUNT = 8;
    private static final int POPULATION = PARAMETER_COUNT * 3;

    public static void main(String[] args) {
        int statePopulation = POPULATION;
        int meanCount = 1;
        int maxIteration = 5;

        List<TuneState> tuneStates = initTuneStates(statePopulation);
        computeMark(meanCount, tuneStates);

        for (int iteration = 0; iteration < maxIteration; iteration++) {
            System.out.println("iteration = " + iteration);
            List<TuneState> mutated = mutate(tuneStates);
            computeMark(meanCount, mutated);
            List<TuneState> merged = merge(tuneStates);
            computeMark(meanCount, merged);

            tuneStates.addAll(mutated);
            tuneStates.addAll(merged);
            tuneStates.sort(Comparator.comparing(TuneState::getMark).reversed());
            tuneStates = tuneStates.subList(0, statePopulation);

            TuneState bestState = tuneStates.get(0);
            System.out.println("Best of iteration " + bestState.getMark() + " with " + bestState.getSuccessRate() + "% in " + bestState.getExecutionTime() + " s " + bestState.getHyperParameters());
        }
    }

    private static List<TuneState> merge(List<TuneState> tuneStates) {
        int population = tuneStates.size();
        Random random = new Random();

        List<TuneState> mergedTuneStates = new ArrayList<>();
        for (int i = 0; i < population / 2; i++) {
            HyperParameters firstParameter = tuneStates.get(random.nextInt(POPULATION / 3)).getHyperParameters().clone();
            HyperParameters secondParameter = tuneStates.get(random.nextInt(population)).getHyperParameters();
            int parameterToMerge = random.nextInt(PARAMETER_COUNT + 1);
            switch (parameterToMerge) {
                case 0:
                    firstParameter.epochCount(secondParameter.getEpochCount());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 1:
                    firstParameter.batchSize(secondParameter.getBatchSize());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 2:
                    firstParameter.hiddenLayerSize(secondParameter.getHiddenLayerSize());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 3:
                    firstParameter.learningRateUpdater(secondParameter.getLearningRateUpdater());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 4:
                    firstParameter.momentumCoeff(secondParameter.getMomentumCoeff());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 5:
                    firstParameter.rmsStopCoeff(secondParameter.getRmsStopCoeff());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 6:
                    firstParameter.layerType(secondParameter.getLayerType());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 7:
                    firstParameter.regularizationCoeff(secondParameter.getRegularizationCoeffs());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                case 8:
                    firstParameter.convolutionLayers(secondParameter.getConvolutionLayers());
                    mergedTuneStates.add(new TuneState(firstParameter));
                    break;
                default:
                    throw new RuntimeException("Missing parameter management");
            }
        }
        return mergedTuneStates;
    }

    private static List<TuneState> mutate(List<TuneState> tuneStates) {
        int population = tuneStates.size();
        Random random = new Random();

        List<TuneState> mutatedTuneStates = new ArrayList<>();
        for (int i = 0; i < population / 2; i++) {
            HyperParameters parameter = tuneStates.get(random.nextInt(POPULATION / 3)).getHyperParameters().clone();
            int parameterToMerge = random.nextInt(PARAMETER_COUNT + 1);
            switch (parameterToMerge) {
                case 0:
                    parameter.epochCount(generateEpochCount(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 1:
                    parameter.batchSize(generateBatchSize(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 2:
                    parameter.hiddenLayerSize(generateHiddenLayerSize(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 3:
                    parameter.learningRateUpdater(generateLearningRateUpdater(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 4:
                    parameter.momentumCoeff(generateMomentumCoeff(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 5:
                    parameter.rmsStopCoeff(generateRmsStopCoeff(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 6:
                    parameter.layerType(generateLayerType(random));
                    mutatedTuneStates.add(new TuneState(parameter));
                    break;
                case 7:
                    RegularizationCoeffs regularizationCoeffs = generateRegularizationCoeffs(random, parameter.getHiddenLayerSize(), parameter.getConvolutionLayers());
                    parameter.regularizationCoeff(regularizationCoeffs);
                    break;
                case 8:
                    parameter.convolutionLayers(generateConvolutionLayer(random));
                    break;
                default:
                    throw new RuntimeException("Missing parameter management");
            }
        }
        return mutatedTuneStates;
    }

    private static void computeMark(int meanCount, List<TuneState> tuneStates) {
        Consumer<NeuralNetworkStats> neuralNetworkStatsConsumer = (unused) -> {
        };

        tuneStates.stream()
                .forEach(tuneState -> {
                    System.out.println("Start computation of " + tuneState.getHyperParameters());
                            MnistExampleLauncher mnistExampleLauncher = new MnistExampleLauncher(neuralNetworkStatsConsumer, tuneState.getHyperParameters());
                            long start = System.nanoTime();
                            double successRate = 0;
                            for (int i = 0; i < meanCount; i++) {
                                try {
                                    successRate += mnistExampleLauncher.launch();
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }
                            double executionTime = (System.nanoTime() - start) / 1_000_000_000d;
                            tuneState.setExecutionTime(executionTime / (double) meanCount);
                            tuneState.setSuccessRate(successRate / (double) meanCount);

                            tuneState.setMark(tuneState.getSuccessRate() + 10 / tuneState.getExecutionTime());
                            System.out.println("Computed " + tuneState.getMark() + " with " + tuneState.getSuccessRate() + "% in " + tuneState.getExecutionTime() + " s " + tuneState.getHyperParameters());
                        }
                );
    }

    private static Double generateRmsStopCoeff(Random random) {
        if (random.nextBoolean()) {
            return random.nextInt(100) * 0.001 + 0.9;
        }
        return null;
    }

    private static Double generateMomentumCoeff(Random random) {
        if (random.nextBoolean()) {
            return random.nextInt(5) * 0.1 + 0.5;
        }
        return null;
    }

    private static ILearningRateUpdater generateLearningRateUpdater(Random random) {
        LearningRateUpdaterType[] learningRateUpdaterTypes = LearningRateUpdaterType.values();
        int updaterIndex = random.nextInt(learningRateUpdaterTypes.length);
        LearningRateUpdaterType learningRateUpdaterType = learningRateUpdaterTypes[updaterIndex];
        int exp = random.nextInt(3) + 1;
        int value = random.nextInt(9) + 1;
        double initialLearningRate = value * Math.pow(10, -exp);
        return learningRateUpdaterType.get(initialLearningRate);
    }

    private static List<TuneState> initTuneStates(int statePopulation) {
        Random random = new Random();
        return IntStream.range(0, statePopulation)
                .mapToObj(i -> {
                    int[] hiddenLayerSize = generateHiddenLayerSize(random);
                    int[] convolutionLayers = generateConvolutionLayer(random);
                    HyperParameters hyperParameters = new HyperParameters()
                            .batchSize(generateBatchSize(random))
                            .hiddenLayerSize(hiddenLayerSize)
                            .learningRateUpdater(generateLearningRateUpdater(random))
                            .epochCount(generateEpochCount(random))
                            .momentumCoeff(generateMomentumCoeff(random))
                            .rmsStopCoeff(generateRmsStopCoeff(random))
                            .layerType(generateLayerType(random))
                            .convolutionLayers(convolutionLayers)
                            .regularizationCoeff(generateRegularizationCoeffs(random, hiddenLayerSize, convolutionLayers));

                    return new TuneState(hyperParameters);
                })
                .collect(Collectors.toList());
    }

    private static int[] generateHiddenLayerSize(Random random) {
        int size = random.nextInt(4) + 1;
        int[] ints = random.ints(size, 10, 784).toArray();
        ints[size - 1] = 10;
        return ints;
    }

    private static int[] generateConvolutionLayer(Random random) {
        int size = random.nextInt(4) + 1;
        return random.ints(size, 1, 64).toArray();
    }

    private static int generateEpochCount(Random random) {
        return random.nextInt(5) + 1;
    }

    private static int generateBatchSize(Random random) {
        return (random.nextInt(200) + 1) * 50;
    }

    private static RegularizationCoeffs generateRegularizationCoeffs(Random random, int[] hiddenLayers, int[] convolutionLayers) {
        int regularizationMethod = random.nextInt(3);
        RegularizationCoeffs regularizationCoeffs = new RegularizationCoeffs();
        switch (regularizationMethod) {
            case 0:
                regularizationCoeffs.setL2RegularizationCoeff(random.nextDouble());
                break;
            case 1:
                double[] dropOutRegularizationCoeffs = DoubleStream.concat(DoubleStream.of(1d), DoubleStream.concat(random.doubles(convolutionLayers.length + hiddenLayers.length), DoubleStream.of(1d))).toArray();
                regularizationCoeffs.setDropOutRegularizationCoeffs(dropOutRegularizationCoeffs);
                break;
            default:
        }
        return regularizationCoeffs;
    }

    private static LayerType generateLayerType(Random random) {
        LayerType[] values = LayerType.values();
        return values[random.nextInt(2) + 3];
    }
}
