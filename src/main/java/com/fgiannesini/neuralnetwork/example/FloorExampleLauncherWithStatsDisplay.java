package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.NeuralNetworkStats;
import javafx.application.Application;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.concurrent.Task;
import javafx.scene.Scene;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FloorExampleLauncherWithStatsDisplay extends Application {

    public static void main(String[] args) {
        launch();
    }

    @Override
    public void start(Stage stage) {

        stage.setTitle("Cost evolution");

        Task<NeuralNetworkStats> task = new Task<NeuralNetworkStats>() {
            @Override
            protected NeuralNetworkStats call() {

                FloorExampleLauncher floorExampleLauncher = new FloorExampleLauncher(this::updateValue);
                double successRate = floorExampleLauncher.launch();
                System.out.println("Success Rate: " + successRate + "%");
                return null;
            }
        };

        ObservableList<XYChart.Data<String, Double>> learningCostList = FXCollections.observableArrayList();
        ObservableList<XYChart.Data<String, Double>> testCostList = FXCollections.observableArrayList();
        ObservableList<String> xSerie = FXCollections.observableArrayList();

        task.valueProperty().addListener((observableValue, oldNeuralNetworkStats, newNeuralNetworkStats) -> {
            if (newNeuralNetworkStats == null) {
                return;
            }
            String iterationCount = String.valueOf(newNeuralNetworkStats.getIterationCount());
            xSerie.add(iterationCount);
            learningCostList.add(new XYChart.Data<>(iterationCount, newNeuralNetworkStats.getLearningCost()));
            testCostList.add(new XYChart.Data<>(iterationCount, newNeuralNetworkStats.getTestCost()));

        });

        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Iteration Count");
        final NumberAxis yAxis = new NumberAxis();

        LineChart<String, Number> lineChart = new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Cost evolution");
        lineChart.setAnimated(false);

        xAxis.setCategories(xSerie);
//        xAxis.setAutoRanging(false);

        XYChart.Series xySeries1 = new XYChart.Series<>(learningCostList);
        xySeries1.setName("Learning Cost Series");

        XYChart.Series xySeries2 = new XYChart.Series<>(testCostList);
        xySeries2.setName("Test Cost Series");

        lineChart.getData().addAll(xySeries1, xySeries2);

        Scene scene = new Scene(lineChart, 800, 600);

        stage.setScene(scene);
        stage.show();

        ExecutorService executor = Executors.newSingleThreadExecutor();
        executor.submit(task);

        stage.setOnCloseRequest(windowEvent -> task.cancel());
    }
}
