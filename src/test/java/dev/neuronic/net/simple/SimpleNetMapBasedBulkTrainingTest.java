package dev.neuronic.net.simple;

import dev.neuronic.net.*;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.training.BatchTrainer;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Map-based bulk training in SimpleNet classes.
 * Ensures the new type-safe trainBulk API works correctly.
 */
class SimpleNetMapBasedBulkTrainingTest {
    
    private final AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
    
    @Test
    void testMapBasedBulkTrainingWithIntClassification() {
        // Create classifier with mixed features
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.embedding(100, 32, "category"),
                Feature.oneHot(5, "type"),
                Feature.passthrough("score")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Prepare training data
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        // Add training samples
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = Map.of(
                "category", "cat_" + (i % 10),
                "type", i % 5,
                "score", (float)(i % 100) / 100.0f
            );
            inputs.add(input);
            labels.add(i % 3);  // 3 classes
        }
        
        // Configure training
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(5)
            .build();
        
        // Train using the new Map-based API
        SimpleNetTrainingResult result = classifier.trainBulk(inputs, labels, config);
        
        // Verify training occurred
        assertNotNull(result);
        assertEquals(5, result.getEpochsTrained());
        assertTrue(result.getTrainingTimeMs() > 0);
        
        // Test prediction with Map input
        Map<String, Object> testInput = Map.of(
            "category", "cat_5",
            "type", 2,
            "score", 0.75f
        );
        int prediction = classifier.predictInt(testInput);
        assertTrue(prediction >= 0 && prediction < 3);
    }
    
    @Test
    void testMapBasedBulkTrainingWithStringClassification() {
        // Create classifier with mixed features
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.embedding(1000, 64, "text"),
                Feature.oneHot(3, "source"),
                Feature.passthrough("length")
            ))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        SimpleNetString classifier = SimpleNet.ofStringClassification(net);
        
        // Prepare training data
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        String[] sentiments = {"positive", "negative", "neutral"};
        
        // Add training samples
        for (int i = 0; i < 150; i++) {
            Map<String, Object> input = Map.of(
                "text", "text_sample_" + i,
                "source", i % 3,
                "length", (float)(50 + i % 100)
            );
            inputs.add(input);
            labels.add(sentiments[i % 3]);
        }
        
        // Configure training
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(16)
            .epochs(3)
            .build();
        
        // Train using the new Map-based API
        SimpleNetTrainingResult result = classifier.trainBulk(inputs, labels, config);
        
        // Verify training occurred
        assertNotNull(result);
        assertEquals(3, result.getEpochsTrained());
        
        // Test prediction
        Map<String, Object> testInput = Map.of(
            "text", "new_text_sample",
            "source", 1,
            "length", 75.0f
        );
        String sentiment = classifier.predictString(testInput);
        assertTrue(Arrays.asList(sentiments).contains(sentiment));
    }
    
    @Test
    void testMapBasedBulkTrainingWithFloatRegression() {
        // Create regressor with mixed features
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.oneHot(10, "location"),
                Feature.passthrough("sqft"),
                Feature.passthrough("bedrooms")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        
        // Prepare training data
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        
        // Add training samples (house price prediction)
        for (int i = 0; i < 100; i++) {
            float sqft = 1000 + i * 20;
            int bedrooms = 1 + (i % 4);
            int location = i % 10;
            
            Map<String, Object> input = Map.of(
                "location", location,
                "sqft", sqft,
                "bedrooms", (float)bedrooms
            );
            inputs.add(input);
            
            // Simple price formula for testing
            float price = 100000 + sqft * 100 + bedrooms * 10000 + location * 5000;
            targets.add(price);
        }
        
        // Configure training
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(20)
            .epochs(10)
            .build();
        
        // Train using the new Map-based API
        SimpleNetTrainingResult result = regressor.trainBulk(inputs, targets, config);
        
        // Verify training occurred
        assertNotNull(result);
        assertEquals(10, result.getEpochsTrained());
        
        // Test prediction
        Map<String, Object> testInput = Map.of(
            "location", 5,
            "sqft", 2000.0f,
            "bedrooms", 3.0f
        );
        float predictedPrice = regressor.predictFloat(testInput);
        assertTrue(predictedPrice > 0, "Price should be positive");
    }
    
    @Test
    void testMapBasedBulkTrainingWithMultiFloatRegression() {
        // Create multi-output regressor with mixed features
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.oneHot(5, "risk_profile"),
                Feature.passthrough("age"),
                Feature.passthrough("amount")
            ))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputLinearRegression(4));
        
        String[] assetNames = {"stocks", "bonds", "real_estate", "cash"};
        SimpleNetMultiFloat allocator = SimpleNet.ofMultiFloatRegression(net, assetNames);
        
        // Prepare training data
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<float[]> targets = new ArrayList<>();
        
        // Add training samples (portfolio allocation)
        for (int i = 0; i < 100; i++) {
            Map<String, Object> input = Map.of(
                "risk_profile", i % 5,
                "age", 20.0f + (i % 50),
                "amount", 10000.0f + i * 1000
            );
            inputs.add(input);
            
            // Create allocation that sums to 1.0
            float[] allocation = new float[4];
            float remaining = 1.0f;
            for (int j = 0; j < 3; j++) {
                allocation[j] = remaining * (0.2f + (i + j) % 3 * 0.1f);
                remaining -= allocation[j];
            }
            allocation[3] = remaining;  // Ensure sum is exactly 1.0
            targets.add(allocation);
        }
        
        // Configure training
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(25)
            .epochs(5)
            .build();
        
        // Train using the new Map-based API
        SimpleNetTrainingResult result = allocator.trainBulk(inputs, targets, config);
        
        // Verify training occurred
        assertNotNull(result);
        assertEquals(5, result.getEpochsTrained());
        
        // Test prediction with named outputs
        Map<String, Object> testInput = Map.of(
            "risk_profile", 2,
            "age", 35.0f,
            "amount", 50000.0f
        );
        
        Map<String, Float> namedAllocation = allocator.predictNamed(testInput);
        assertNotNull(namedAllocation);
        assertEquals(4, namedAllocation.size());
        assertTrue(namedAllocation.containsKey("stocks"));
        assertTrue(namedAllocation.containsKey("bonds"));
        assertTrue(namedAllocation.containsKey("real_estate"));
        assertTrue(namedAllocation.containsKey("cash"));
        
        // Also test regular array prediction
        float[] allocation = allocator.predictMultiFloat(testInput);
        assertEquals(4, allocation.length);
    }
    
    @Test
    void testSerializationWithMapBasedModels() throws Exception {
        // Create classifier with mixed features
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.embedding(50, 16, "word"),
                Feature.passthrough("confidence")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Train with some data
        List<Map<String, Object>> inputs = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        
        for (int i = 0; i < 50; i++) {
            Map<String, Object> input = Map.of(
                "word", "word_" + (i % 10),
                "confidence", (float)i / 50.0f
            );
            inputs.add(input);
            labels.add(i % 2);  // Binary classification
        }
        
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(10)
            .epochs(3)
            .build();
        
        classifier.trainBulk(inputs, labels, config);
        
        // Save the model
        Path tempFile = Files.createTempFile("test_model", ".nn");
        try {
            classifier.save(tempFile);
            
            // Load the model
            SimpleNetInt loaded = SimpleNetInt.load(tempFile);
            
            // Test that loaded model works with Map input
            Map<String, Object> testInput = Map.of(
                "word", "word_5",
                "confidence", 0.8f
            );
            
            // Compare predictions
            int originalPred = classifier.predictInt(testInput);
            int loadedPred = loaded.predictInt(testInput);
            assertEquals(originalPred, loadedPred, "Predictions should match after serialization");
            
            // Verify the model can still train
            loaded.train(testInput, 1);
            
        } finally {
            Files.deleteIfExists(tempFile);
        }
    }
    
    @Test
    void testErrorHandlingForWrongInputTypes() {
        // Create classifier expecting Map inputs
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.embedding(50, 16, "feature1"),
                Feature.passthrough("feature2")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Try to use trainBulk with a mixed feature model
        List<float[]> arrayInputs = List.of(
            new float[]{1.0f, 2.0f},
            new float[]{3.0f, 4.0f}
        );
        List<Integer> labels = List.of(0, 1);
        
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(10)
            .epochs(1)
            .build();
        
        // This should work - the model accepts both array and map inputs
        SimpleNetTrainingResult result = classifier.trainBulk(arrayInputs, labels, config);
        assertNotNull(result);
    }
    
    @Test
    void testLanguageModelRejectsMapBasedTraining() {
        // Create language model
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputSequenceEmbedding(10, 100, 32))
            .layer(Layers.hiddenGruLast(64))
            .output(Layers.outputSoftmaxCrossEntropy(100));
        
        SimpleNetLanguageModel lm = SimpleNet.ofLanguageModel(net);
        
        // Try to use Map-based training
        List<Map<String, Object>> mapInputs = List.of(
            Map.of("text", "hello"),
            Map.of("text", "world")
        );
        List<String> targets = List.of("there", "!");
        
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(10)
            .epochs(1)
            .build();
        
        // Should throw UnsupportedOperationException
        assertThrows(UnsupportedOperationException.class, () -> {
            lm.trainBulk(mapInputs, targets, config);
        });
    }
}