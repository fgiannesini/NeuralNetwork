package com.fgiannesini.neuralnetwork.example.tune;

import com.fgiannesini.neuralnetwork.HyperParameters;
import com.fgiannesini.neuralnetwork.NeuralNetworkStats;
import com.fgiannesini.neuralnetwork.example.FloorExampleLauncher;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Tuner {

    public static void main(String[] args) {
        int statePopulation = 10;
        int meanCount = 3;
        int maxIteration = 10;

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
        }
        System.out.println(tuneStates.get(0).getHyperParameters());
    }

    private static List<TuneState> merge(List<TuneState> tuneStates) {
        int population = tuneStates.size();
        Random random = new Random();

        List<TuneState> mergedTuneStates = new ArrayList<>();
        for (int i = 0; i < population / 2; i++) {
            HyperParameters firstParameter = tuneStates.get(random.nextInt(population)).getHyperParameters().clone();
            HyperParameters secondParameter = tuneStates.get(random.nextInt(population)).getHyperParameters();
            int parameterToMerge = random.nextInt(4);
            switch (parameterToMerge) {
                case 0:
                    firstParameter.iterationCount(secondParameter.getIterationCount());
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
                    firstParameter.inputCount(secondParameter.getInputCount());
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
            HyperParameters parameter = tuneStates.get(random.nextInt(population)).getHyperParameters().clone();
            int parameterToMerge = random.nextInt(4);
            switch (parameterToMerge) {
                case 0:
                    parameter.iterationCount(generateBatchSize(random));
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
                    parameter.inputCount(generateInputSize(random));
                    mutatedTuneStates.add(new TuneState(parameter));
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

        tuneStates.parallelStream()
                .forEach(tuneState -> {
                            System.out.println("Mark computation for tuneState with parameters " + tuneState.getHyperParameters());
                            FloorExampleLauncher floorExampleLauncher = new FloorExampleLauncher(neuralNetworkStatsConsumer, tuneState.getHyperParameters());
                            double successRate = 0;
                            for (int i = 0; i < meanCount; i++) {
                                successRate += floorExampleLauncher.launch();
                            }
                            tuneState.setMark(successRate / (double) meanCount);
                            System.out.println("Mark for tuneState with parameters " + tuneState.getHyperParameters() + " " + tuneState.getMark());
                        }
                );
    }

    private static List<TuneState> initTuneStates(int statePopulation) {
        Random random = new Random();
        return IntStream.range(0, statePopulation)
                .mapToObj(i -> {
                    HyperParameters hyperParameters = new HyperParameters()
                            .batchSize(generateBatchSize(random))
                            .hiddenLayerSize(generateHiddenLayerSize(random))
                            .inputCount(generateInputSize(random))
                            .testInputCount(100)
                            .iterationCount(generateIterationCount(random));
                    return new TuneState(hyperParameters);
                })
                .collect(Collectors.toList());
    }

    private static int[] generateHiddenLayerSize(Random random) {
        int streamSize = random.nextInt(4) + 1;
        return random.ints(streamSize, 1, 50).toArray();
    }

    private static int generateIterationCount(Random random) {
        return 150;
    }

    private static int generateBatchSize(Random random) {
        return random.nextInt(999) * 10 + 10;
    }

    private static int generateInputSize(Random random) {
        return random.nextInt(9_999) * 10 + 10;
    }
}
