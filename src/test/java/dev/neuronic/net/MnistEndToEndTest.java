package dev.neuronic.net;

import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.common.Utils;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetInt;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestMethodOrder;
import org.knowm.datasets.mnist.Mnist;
import org.knowm.datasets.mnist.MnistDAO;

import java.util.List;
import java.util.ArrayList;

import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end test that trains a neural network on MNIST and verifies accuracy > 90%.
 * This test requires the MNIST dataset to be available in src/test/resources/datasets/
 */
@TestMethodOrder(org.junit.jupiter.api.MethodOrderer.OrderAnnotation.class)
class MnistEndToEndTest {

    static {
        System.setProperty("OMP_NUM_THREADS", "16");
    }

    private static NeuralNet trainedModel = null;
    private static float[][] testData = null;

    @BeforeAll
    static void setupDataset() {
        // Suppress verbose logging
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "WARN");
        System.setProperty("com.zaxxer.hikari.HikariConfig", "OFF");

        // Check if dataset exists in test resources
        File datasetDir = new File("src/test/resources/datasets");
        if (!datasetDir.exists()) {
            System.err.println("MNIST dataset not found in test resources!");
            System.err.println("Please copy the 'datasets' folder to: src/test/resources/");
            System.err.println("The folder should contain the MNIST database files.");
        }
    }

    @Test
    @org.junit.jupiter.api.Order(1)
    void testMnistTrainingAchievesHighAccuracy() {
        // Skip test if dataset is not available
        File datasetDir = new File("src/test/resources/datasets");
        if (!datasetDir.exists()) {
            System.out.println("Skipping MNIST test - dataset not available");
            return;
        }

        // Initialize dataset from test resources
        MnistDAO.init("src/test/resources/datasets");
        int split = MnistDAO.getTrainTestSplit();

        // Use same configuration as Main
        SgdOptimizer optimizer = new SgdOptimizer(0.025f);

        NeuralNet net = NeuralNet.newBuilder()
                .input(784)
                .layer(Layers.hiddenDenseRelu(64, optimizer))
                .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));

        // Load training data into Lists for SimpleNetInt
        System.out.printf("Loading training data...\n");
        List<Object> trainInputsList = new ArrayList<>();
        List<Integer> trainLabelsList = new ArrayList<>();

        for (int i = 0; i < split; i++) {
            Mnist entry = MnistDAO.selectSingle(i);
            float[] input = Utils.flatten(entry.getImageMatrix(), true);
            trainInputsList.add(input);
            trainLabelsList.add(entry.getLabel());
        }
        System.out.printf("Loaded %d training samples\n", split);

        // Load test data
        int testSize = (int) MnistDAO.selectCount() - split;
        List<Object> testInputsList = new ArrayList<>();
        List<Integer> testLabelsList = new ArrayList<>();

        for (int i = 0; i < testSize; i++) {
            Mnist entry = MnistDAO.selectSingle(split + i);
            float[] input = Utils.flatten(entry.getImageMatrix(), true);
            testInputsList.add(input);
            testLabelsList.add(entry.getLabel());
        }
        System.out.printf("Loaded %d test samples\n", testSize);

        // Use BatchTrainer for proper bulk training with all data loaded at once
        long startTime = System.currentTimeMillis();

        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
                .batchSize(16)  // Balanced batch size for performance and accuracy
                .epochs(3)
                .verbosity(1)
                .build();

        SimpleNetInt simpleNet = SimpleNet.ofIntClassification(net);

        // Train with TRAIN data, not test data!
        simpleNet.trainBulk(trainInputsList, trainLabelsList, config);

        long trainingTime = System.currentTimeMillis() - startTime;
        System.out.printf("Training completed in %.1f seconds\n", trainingTime / 1000.0);

        // Evaluate accuracy using SimpleNetInt
        int correct = 0;
        for (int i = 0; i < testInputsList.size(); i++) {
            int predicted = simpleNet.predictInt(testInputsList.get(i));
            if (predicted == testLabelsList.get(i)) {
                correct++;
            }
        }
        float accuracy = (correct * 100.0f) / testInputsList.size();
        System.out.printf("Test accuracy: %.1f%%\n", accuracy);

        // Save trained model and test data for throughput test
        trainedModel = net;
        // Convert List<Object> to float[][]
        testData = new float[testInputsList.size()][];
        for (int i = 0; i < testInputsList.size(); i++) {
            testData[i] = (float[]) testInputsList.get(i);
        }

        // Assert accuracy > 91% (larger batch sizes may need more epochs for full convergence)
        assertTrue(accuracy > 91.0f,
                String.format("Expected accuracy > 91%%, but got %.1f%%", accuracy));
    }

    @Test
    @org.junit.jupiter.api.Order(2)
    void testPredictionThroughput() {
        // This test should run after testTrainingAccuracy
        if (trainedModel == null || testData == null) {
            System.out.println("Skipping throughput test - no trained model available");
            System.out.println("Run testTrainingAccuracy first to train the model");
            return;
        }

        // Use the trained model from the accuracy test
        NeuralNet net = trainedModel;

        // Use first 1000 test samples for throughput testing
        float[][] testInputs = new float[Math.min(1000, testData.length)][];
        System.arraycopy(testData, 0, testInputs, 0, testInputs.length);

        // Warmup phase - let JVM optimize
        System.out.println("Warming up JVM...");
        for (int warmup = 0; warmup < 50; warmup++) {
            net.predictBatch(testInputs);
        }

        // Test batch predictions for better throughput
        // Use more iterations for stable measurement
        int iterations = 100;
        long startTime = System.nanoTime();
        int totalPredictions = 0;

        for (int iter = 0; iter < iterations; iter++) {
            // Use batch predictions for efficiency
            net.predictBatch(testInputs);
            totalPredictions += testInputs.length;
        }

        long duration = System.nanoTime() - startTime;

        double seconds = duration / 1_000_000_000.0;
        double throughput = totalPredictions / seconds;

        System.out.printf("Prediction throughput: %.0f predictions/second (%.1f ms per batch)\n",
                throughput, (duration / iterations) / 1_000_000.0);

        // Should achieve reasonable throughput - lowered threshold for reliability
        assertTrue(throughput > 40_000,
                String.format("Expected throughput > 40k/sec, but got %.0f/sec", throughput));
    }

    @Test
    @org.junit.jupiter.api.Order(3)
    void testSinglePredictionParallelThroughput() throws Exception {
        if (trainedModel == null || testData == null) {
            System.out.println("Skipping single‐prediction throughput test – no model available");
            return;
        }

        NeuralNet net       = trainedModel;
        float[][] inputs    = testData;
        int      numThreads = Runtime.getRuntime().availableProcessors();
        int      passes     = 1;          // how many times each thread loops over the full set
        int      N          = inputs.length;

        ExecutorService exec  = Executors.newFixedThreadPool(numThreads);
        CountDownLatch ready  = new CountDownLatch(numThreads);
        CountDownLatch start  = new CountDownLatch(1);
        CountDownLatch done   = new CountDownLatch(numThreads);
        AtomicInteger counter = new AtomicInteger(0);

        // each worker will wait for the “start” latch, then run passes×N single predicts
        for (int t = 0; t < numThreads; t++) {
            exec.submit(() -> {
                ready.countDown();
                try { start.await(); } catch (InterruptedException ignored) {}
                for (int p = 0; p < passes; p++) {
                    for (int i = 0; i < N; i++) {
                        net.predict(inputs[i]);
                        counter.incrementAndGet();
                    }
                }
                done.countDown();
            });
        }

        // wait until everyone’s ready, then time the burst
        ready.await();
        long t0 = System.nanoTime();
        start.countDown();
        done.await();
        long elapsed = System.nanoTime() - t0;
        exec.shutdown();

        double seconds    = elapsed / 1e9;
        double throughput = counter.get() / seconds;
        System.out.printf(
                "Single‐sample parallel throughput: %.0f predictions/sec  (%d threads × %d samples × %d passes = %,d)\n",
                throughput, numThreads, N, passes, counter.get());

        // simple sanity check – tweak to suit your hardware
        assertTrue(throughput > numThreads * 5_000,
                String.format("Expected >%d preds/sec, but got %.0f",
                        numThreads * 5_000, throughput));
    }
}