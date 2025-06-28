package dev.neuronic.net.training;

import dev.neuronic.net.*;
import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetInt;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the new bulk training and metrics collection functionality.
 * Demonstrates the complete performance tracking pipeline.
 */
class BulkTrainingIntegrationTest {
    
    @TempDir
    Path tempDir;
    
    @Test
    void testBulkTrainingWithMetricsCollection() {
        // Create a simple neural network for testing
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(8))
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Generate synthetic classification data (3 classes, 4 features)
        List<Object> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < 300; i++) {
            float[] features = new float[4];
            for (int j = 0; j < 4; j++) {
                features[j] = (float) random.nextGaussian();
            }
            
            // Create separable classes based on features
            int label = (features[0] + features[1] > 0) ? 
                       ((features[2] > 0) ? 0 : 1) : 2;
            
            inputs.add(features);
            labels.add(label);
        }
        
        // Configure training with callbacks
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(5)
            .build();
        
        // Train with bulk method
        SimpleNetTrainingResult result = classifier.trainBulk(inputs, labels, config);
        
        // Get metrics from the result
        TrainingMetrics metrics = result.getMetrics();
        double finalValAccuracy = result.getFinalValidationAccuracy();
        
        // Verify training completed successfully
        assertTrue(finalValAccuracy > 0.0, "Final validation accuracy should be positive");
        assertEquals(5, metrics.getEpochCount(), "Should have completed 5 epochs");
        // Note: progressMessages tracking removed as the new API doesn't support custom callbacks directly
        
        // Verify metrics were collected
        assertNotNull(metrics.getFinalAccuracy(), "Should have final training accuracy");
        assertNotNull(metrics.getFinalValidationAccuracy(), "Should have final validation accuracy");
        assertTrue(metrics.getTotalSamplesSeen() > 0, "Should have processed training samples");
        assertTrue(metrics.getTotalTrainingTime().toNanos() > 0, "Should have recorded training time");
        
        // Verify epoch metrics
        TrainingMetrics.EpochMetrics firstEpoch = metrics.getEpochMetrics(0);
        assertNotNull(firstEpoch, "Should have first epoch metrics");
        assertEquals(0, firstEpoch.getEpochNumber(), "First epoch should be number 0");
        assertTrue(firstEpoch.getTrainingAccuracy() >= 0.0, "Training accuracy should be non-negative");
        assertTrue(firstEpoch.getValidationAccuracy() >= 0.0, "Validation accuracy should be non-negative");
        
        // Test summary generation
        String summary = metrics.getSummary();
        assertNotNull(summary, "Should generate summary");
        assertTrue(summary.contains("Training Summary"), "Summary should contain header");
        assertTrue(summary.contains("Epochs: 5"), "Summary should show epoch count");
    }
    
    @Test
    void testMetricsExportToJson() throws IOException {
        // Create minimal training setup
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Minimal training data (more samples to avoid empty training set after split)
        List<Object> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            inputs.add(new float[]{i % 2 == 0 ? 1.0f : -1.0f, i % 2 == 0 ? 2.0f : -2.0f});
            labels.add(i % 2);
        }
        
        // Train and collect metrics
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(2)
            .build();
        SimpleNetTrainingResult result = classifier.trainBulk(inputs, labels, config);
        TrainingMetrics metrics = result.getMetrics();
        
        // Test JSON export
        Path jsonFile = tempDir.resolve("test_metrics.json");
        Map<String, Object> metadata = Map.of(
            "test_name", "testMetricsExportToJson",
            "learning_rate", 0.01,
            "optimizer", "AdamW"
        );
        
        assertDoesNotThrow(() -> 
            MetricsLogger.exportToJson(metrics, jsonFile, metadata),
            "JSON export should not throw exception");
        
        assertTrue(jsonFile.toFile().exists(), "JSON file should be created");
        assertTrue(jsonFile.toFile().length() > 0, "JSON file should not be empty");
        
        // Test CSV export
        Path csvFile = tempDir.resolve("test_metrics.csv");
        assertDoesNotThrow(() -> 
            MetricsLogger.exportToCsv(metrics, csvFile),
            "CSV export should not throw exception");
        
        assertTrue(csvFile.toFile().exists(), "CSV file should be created");
        assertTrue(csvFile.toFile().length() > 0, "CSV file should not be empty");
    }
    
    @Test
    void testValidationEvaluatorSplitting() {
        // Create test data with known distribution
        List<Object> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        // Add 30 samples of each class (0, 1, 2) for stratified testing
        for (int label = 0; label < 3; label++) {
            for (int i = 0; i < 30; i++) {
                inputs.add(new float[]{label * 1.0f, i * 0.1f});
                labels.add(label);
            }
        }
        
        // Test stratified split
        ValidationEvaluator<Object, Integer> evaluator = 
            ValidationEvaluator.forClassification(inputs, labels)
                .withSplit(0.8)
                .stratified(true);
        
        ValidationEvaluator.TrainingData<Object, Integer> trainData = evaluator.getTrainingData();
        ValidationEvaluator.TrainingData<Object, Integer> valData = evaluator.getValidationData();
        
        // Verify split sizes
        assertEquals(72, trainData.size(), "Training set should be ~80% of 90 samples");
        assertEquals(18, valData.size(), "Validation set should be ~20% of 90 samples");
        assertEquals(90, trainData.size() + valData.size(), "Total should equal original size");
        
        // Verify stratification (each class should be represented proportionally)
        Map<Integer, Long> trainClassCounts = trainData.getOutputs().stream()
            .collect(java.util.stream.Collectors.groupingBy(x -> x, java.util.stream.Collectors.counting()));
        Map<Integer, Long> valClassCounts = valData.getOutputs().stream()
            .collect(java.util.stream.Collectors.groupingBy(x -> x, java.util.stream.Collectors.counting()));
        
        assertEquals(3, trainClassCounts.size(), "Training set should have all 3 classes");
        assertEquals(3, valClassCounts.size(), "Validation set should have all 3 classes");
        
        // Each class should have roughly the same proportion in train/val
        for (int label = 0; label < 3; label++) {
            long trainCount = trainClassCounts.getOrDefault(label, 0L);
            long valCount = valClassCounts.getOrDefault(label, 0L);
            assertEquals(30, trainCount + valCount, "Total for class " + label + " should be 30");
            assertTrue(trainCount >= 20, "Training count for class " + label + " should be at least 20");
            assertTrue(valCount >= 4, "Validation count for class " + label + " should be at least 4");
        }
    }
    
    @Test
    void testProgressCallbacks() {
        // Test basic training metrics collection without custom callbacks
        
        // Create minimal network and data
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(3))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Need enough samples for proper train/val split
        List<Object> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            inputs.add(new float[]{i % 2 == 0 ? 1.0f : 0.0f, i % 2 == 0 ? 0.0f : 1.0f});
            labels.add(i % 2);
        }
        
        // Train with configuration
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(3)
            .build();
        SimpleNetTrainingResult result = classifier.trainBulk(inputs, labels, config);
        
        // Note: Custom callback testing removed as the new API doesn't support custom callbacks directly
        // Verify training completed
        assertEquals(3, result.getEpochsTrained(), "Should have completed 3 epochs");
    }
    
    @Test
    void testModelEvaluationMethods() {
        // Create and train a simple model
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(4))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Training data
        List<Object> trainInputs = Arrays.asList(
            new float[]{1.0f, 1.0f},
            new float[]{-1.0f, -1.0f},
            new float[]{1.0f, -1.0f},
            new float[]{-1.0f, 1.0f}
        );
        List<Integer> trainLabels = Arrays.asList(1, 0, 1, 0);
        
        // Train the model
        TrainingMetrics metrics = new TrainingMetrics();
        classifier.trainBulk(trainInputs, trainLabels, 3, metrics, 
                           MetricsLogger.ProgressCallbacks.silent());
        
        // Test evaluation on separate test set
        List<Object> testInputs = Arrays.asList(
            new float[]{0.5f, 0.5f},
            new float[]{-0.5f, -0.5f}
        );
        List<Integer> testLabels = Arrays.asList(1, 0);
        
        ValidationEvaluator.ValidationResults results = classifier.evaluate(testInputs, testLabels);
        assertNotNull(results, "Evaluation results should not be null");
        assertTrue(results.getAccuracy() >= 0.0, "Accuracy should be non-negative");
        assertTrue(results.getAccuracy() <= 1.0, "Accuracy should not exceed 1.0");
        assertEquals(2, results.getTotalSamples(), "Should evaluate 2 test samples");
        
        // Test performance metrics
        Map<String, Double> perfMetrics = classifier.getPerformanceMetrics(testInputs, testLabels);
        assertNotNull(perfMetrics, "Performance metrics should not be null");
        assertTrue(perfMetrics.containsKey("accuracy"), "Should contain overall accuracy");
        assertTrue(perfMetrics.containsKey("num_classes"), "Should contain class count");
        assertTrue(perfMetrics.containsKey("total_samples"), "Should contain sample count");
        
        assertEquals(2.0, perfMetrics.get("total_samples"), "Should report 2 samples");
        assertTrue(perfMetrics.get("num_classes") >= 1.0, "Should have at least 1 class");
    }
    
    @Test
    void testOnlineLearningPreservation() {
        // Verify that simple train() methods still work for online learning
        AdamWOptimizer optimizer = new AdamWOptimizer(0.01f, 0.01f);
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(3))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Online training (original simple API preserved)
        float[] input1 = {1.0f, 0.0f};
        int label1 = 1;
        
        assertDoesNotThrow(() -> classifier.train(input1, label1),
                          "Simple train() method should still work");
        
        int prediction = classifier.predictInt(input1);
        assertTrue(prediction >= 0, "Prediction should be non-negative class");
        
        // Verify class tracking still works
        assertTrue(classifier.hasSeenLabel(label1), "Should track seen labels");
        assertEquals(1, classifier.getClassCount(), "Should count 1 class seen");
    }
}