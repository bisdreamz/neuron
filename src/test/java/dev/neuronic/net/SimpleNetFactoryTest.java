package dev.neuronic.net;

import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetInt;
import dev.neuronic.net.simple.SimpleNetString;
import dev.neuronic.net.simple.SimpleNetFloat;
import dev.neuronic.net.simple.SimpleNetMultiFloat;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.LinkedHashSet;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for the new type-safe SimpleNet factory API.
 * Tests factory methods, type safety, validation, and serialization.
 */
class SimpleNetFactoryTest {
    
    @TempDir
    Path tempDir;
    
    // ===============================
    // FACTORY METHOD TESTS
    // ===============================
    
    @Test
    void testIntClassificationFactory() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network for integer classification
        NeuralNet net = NeuralNet.newBuilder()
            .input(784)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSoftmaxCrossEntropy(10));  // 10 classes
        
        // Test factory method
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        assertNotNull(classifier, "Factory should create SimpleNetInt instance");
        
        // Test type safety - returns primitive int
        float[] input = createRandomInput(784);
        int prediction = classifier.predictInt(input);
        assertTrue(prediction >= 0, "Prediction should be non-negative class index");
        
        // Test training with automatic label discovery
        assertDoesNotThrow(() -> {
            classifier.train(input, 7);  // First time seeing label 7
            classifier.train(input, 3);  // First time seeing label 3
            classifier.train(input, 7);  // Label 7 again
        });
        
        assertEquals(2, classifier.getClassCount(), "Should have seen 2 unique classes");
        assertTrue(classifier.hasSeenLabel(7), "Should have seen label 7");
        assertTrue(classifier.hasSeenLabel(3), "Should have seen label 3");
        assertFalse(classifier.hasSeenLabel(5), "Should not have seen label 5");
    }
    
    @Test
    void testStringClassificationFactory() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network for string classification
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(1000, 64),  // text_tokens
                Feature.oneHot(4),            // source_type
                Feature.passthrough()         // text_length
            ))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputSoftmaxCrossEntropy(3));  // 3 sentiment classes
        
        // Test factory method
        SimpleNetString classifier = SimpleNet.ofStringClassification(net);
        assertNotNull(classifier, "Factory should create SimpleNetString instance");
        
        // Test type safety - returns String
        Map<String, Object> input = Map.of(
            "feature_0", "This is amazing!",
            "feature_1", 1,
            "feature_2", 127.5f
        );
        
        // Train with string labels
        assertDoesNotThrow(() -> {
            classifier.train(input, "positive");
            classifier.train(input, "negative");
            classifier.train(input, "neutral");
        });
        
        String prediction = classifier.predictString(input);
        assertNotNull(prediction, "Prediction should not be null");
        assertTrue(prediction instanceof String, "Prediction should be String type");
        
        assertEquals(3, classifier.getClassCount(), "Should have seen 3 unique classes");
        assertTrue(classifier.hasSeenLabel("positive"), "Should have seen 'positive' label");
        assertTrue(classifier.hasSeenLabel("negative"), "Should have seen 'negative' label");
        assertTrue(classifier.hasSeenLabel("neutral"), "Should have seen 'neutral' label");
    }
    
    @Test
    void testFloatRegressionFactory() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network for single regression
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(500, 32),  // neighborhood
                Feature.oneHot(5),           // house_type
                Feature.passthrough(),       // square_feet
                Feature.passthrough()        // bedrooms
            ))
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputLinearRegression(1));  // Single output
        
        // Test factory method
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        assertNotNull(regressor, "Factory should create SimpleNetFloat instance");
        
        // Test type safety - returns primitive float
        Map<String, Object> input = Map.of(
            "feature_0", "downtown",
            "feature_1", 2,
            "feature_2", 2400.0f,
            "feature_3", 3.0f
        );
        
        // Train with float targets
        assertDoesNotThrow(() -> {
            regressor.train(input, 485000.0f);
            regressor.train(input, 520000.0f);
            regressor.train(input, 450000.0f);
        });
        
        float prediction = regressor.predictFloat(input);
        assertTrue(Float.isFinite(prediction), "Prediction should be a finite float");
        
        // Test batch prediction
        Object[] inputs = {input, input, input};
        float[] predictions = regressor.predict(inputs);
        assertEquals(3, predictions.length, "Should return array of same length as input");
        for (float pred : predictions) {
            assertTrue(Float.isFinite(pred), "All predictions should be finite");
        }
    }
    
    // ===============================
    // VALIDATION TESTS
    // ===============================
    
    @Test
    void testClassificationValidation() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create regression network (wrong for classification)
        NeuralNet regressionNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        // Should throw for classification factories
        IllegalArgumentException exception1 = assertThrows(IllegalArgumentException.class, 
            () -> SimpleNet.ofIntClassification(regressionNet));
        assertTrue(exception1.getMessage().contains("classification"), 
            "Error should mention classification requirement");
        
        IllegalArgumentException exception2 = assertThrows(IllegalArgumentException.class, 
            () -> SimpleNet.ofStringClassification(regressionNet));
        assertTrue(exception2.getMessage().contains("classification"), 
            "Error should mention classification requirement");
    }
    
    @Test
    void testRegressionValidation() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create classification network (wrong for regression)
        NeuralNet classificationNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        // Should throw for regression factory
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, 
            () -> SimpleNet.ofFloatRegression(classificationNet));
        assertTrue(exception.getMessage().contains("regression"), 
            "Error should mention regression requirement");
        
        // Multi-output regression should also fail for single regression
        NeuralNet multiOutputNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(3));  // Multiple outputs
        
        IllegalArgumentException exception2 = assertThrows(IllegalArgumentException.class, 
            () -> SimpleNet.ofFloatRegression(multiOutputNet));
        assertTrue(exception2.getMessage().contains("single regression"), 
            "Error should mention single regression requirement");
    }
    
    @Test
    void testPrivateConstructor() {
        // SimpleNet is abstract and cannot be directly instantiated
        assertTrue(java.lang.reflect.Modifier.isAbstract(SimpleNet.class.getModifiers()),
                  "SimpleNet should be abstract to prevent direct instantiation");
    }
    
    // ===============================
    // TYPE SAFETY TESTS
    // ===============================
    
    @Test
    void testIntClassificationTypeSafety() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        float[] input = createRandomInput(10);
        
        // Train and test type safety
        classifier.train(input, 0);
        classifier.train(input, 1);
        classifier.train(input, 4);
        
        // predict() returns primitive int - no casting needed
        int prediction = classifier.predictInt(input);
        assertTrue(prediction >= 0 && prediction <= 4, "Prediction should be valid class index");
        
        // predictTopK() returns int[] - no casting needed
        int[] topK = classifier.predictTopK(input, 3);
        assertEquals(3, topK.length, "Should return requested number of predictions");
        for (int pred : topK) {
            assertTrue(pred >= 0 && pred <= 4, "All top-K predictions should be valid");
        }
        
        // predictConfidence() returns primitive float - no casting needed
        float confidence = classifier.predictConfidence(input);
        assertTrue(confidence >= 0.0f && confidence <= 1.0f, "Confidence should be a probability");
    }
    
    @Test
    void testStringClassificationTypeSafety() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(5)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        SimpleNetString classifier = SimpleNet.ofStringClassification(net);
        float[] input = createRandomInput(5);
        
        // Train with string labels
        classifier.train(input, "spam");
        classifier.train(input, "legitimate");
        classifier.train(input, "promotional");
        
        // predict() returns String - no casting needed
        String prediction = classifier.predictString(input);
        assertTrue(prediction.equals("spam") || prediction.equals("legitimate") || 
                  prediction.equals("promotional") || prediction.startsWith("class_"),
                  "Prediction should be a known label or default class name");
        
        // predictTopK() returns String[] - no casting needed
        String[] topK = classifier.predictTopK(input, 2);
        assertEquals(2, topK.length, "Should return requested number of predictions");
        for (String pred : topK) {
            assertNotNull(pred, "All top-K predictions should be non-null");
        }
    }
    
    @Test
    void testFloatRegressionTypeSafety() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(8)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        float[] input = createRandomInput(8);
        
        // Train with various numeric types
        regressor.train(input, 100.5f);      // float
        regressor.train(input, 200);         // int (via Number overload)
        regressor.train(input, 150.75);      // double (via Number overload)
        
        // predict() returns primitive float - no casting needed
        float prediction = regressor.predictFloat(input);
        assertTrue(Float.isFinite(prediction), "Prediction should be finite");
        
        // Arithmetic works directly on the result
        float doubled = prediction * 2;
        float incremented = prediction + 1;
        assertTrue(Float.isFinite(doubled) && Float.isFinite(incremented), 
                  "Arithmetic should work directly on primitive result");
    }
    
    // ===============================
    // MIXED FEATURES TESTS
    // ===============================
    
    @Test
    void testMixedFeaturesWithIntClassification() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 16),  // category
                Feature.oneHot(5),           // priority
                Feature.passthrough(),       // score
                Feature.embedding(50, 8)     // source
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        Map<String, Object> input = Map.of(
            "feature_0", "category_A",
            "feature_1", 2,
            "feature_2", 0.75f,
            "feature_3", "source_X"
        );
        
        // Test training and prediction with mixed features
        assertDoesNotThrow(() -> {
            classifier.train(input, 0);
            classifier.train(input, 1);
            classifier.train(input, 2);
        });
        
        int prediction = classifier.predictInt(input);
        assertTrue(prediction >= 0 && prediction <= 2, "Prediction should be valid class");
    }
    
    @Test
    void testMixedFeaturesValidation() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 16),
                Feature.oneHot(3),
                Feature.passthrough()  // Must be numerical
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetString classifier = SimpleNet.ofStringClassification(net);
        
        // Valid input
        Map<String, Object> validInput = Map.of(
            "feature_0", "text_value",
            "feature_1", 1,
            "feature_2", 42.5f
        );
        assertDoesNotThrow(() -> classifier.predictString(validInput));
        
        // Invalid - wrong number of features
        Map<String, Object> wrongSize = Map.of(
            "feature_0", "text_value",
            "feature_1", 1
            // Missing feature_2
        );
        IllegalArgumentException exception1 = assertThrows(IllegalArgumentException.class, 
            () -> classifier.predictString(wrongSize));
        assertTrue(exception1.getMessage().contains("exactly 3 features"), 
                  "Should complain about feature count");
        
        // Invalid - non-numerical value for passthrough
        Map<String, Object> invalidType = Map.of(
            "feature_0", "text_value",
            "feature_1", 1,
            "feature_2", "not_a_number"
        );
        IllegalArgumentException exception2 = assertThrows(IllegalArgumentException.class, 
            () -> classifier.predictString(invalidType));
        assertTrue(exception2.getMessage().contains("PASSTHROUGH"), 
                  "Should complain about passthrough feature type");
        
        // Invalid - missing feature
        Map<String, Object> missingFeature = Map.of(
            "feature_0", "text_value",
            "feature_1", 1,
            "wrong_name", 42.5f
        );
        IllegalArgumentException exception3 = assertThrows(IllegalArgumentException.class, 
            () -> classifier.predictString(missingFeature));
        assertTrue(exception3.getMessage().contains("Missing required feature"), 
                  "Should complain about missing required feature");
    }
    
    // ===============================
    // SERIALIZATION TESTS
    // ===============================
    
    @Test
    void testSimpleNetIntSerialization() throws IOException {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create and train original model
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(784)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(128))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        
        SimpleNetInt originalClassifier = SimpleNet.ofIntClassification(originalNet);
        
        // Train with some data to build dictionaries
        float[] input1 = createRandomInput(784);
        float[] input2 = createRandomInput(784);
        originalClassifier.train(input1, 7);
        originalClassifier.train(input2, 3);
        originalClassifier.train(input1, 9);
        
        // Test predictions before serialization
        int originalPred1 = originalClassifier.predictInt(input1);
        int originalPred2 = originalClassifier.predictInt(input2);
        
        // Serialize the complete SimpleNetInt (including dictionaries)
        Path modelFile = tempDir.resolve("simplenet_int.bin");
        originalClassifier.save(modelFile);
        
        // Deserialize complete SimpleNetInt
        SimpleNetInt loadedClassifier = SimpleNetInt.load(modelFile);
        
        // Test predictions after serialization - should be identical
        int loadedPred1 = loadedClassifier.predictInt(input1);
        int loadedPred2 = loadedClassifier.predictInt(input2);
        
        // State should be completely preserved
        assertEquals(originalPred1, loadedPred1, "Prediction 1 should be identical after serialization");
        assertEquals(originalPred2, loadedPred2, "Prediction 2 should be identical after serialization");
        assertEquals(3, loadedClassifier.getClassCount(), "Should have same number of classes");
        assertTrue(loadedClassifier.hasSeenLabel(7), "Should remember label 7");
        assertTrue(loadedClassifier.hasSeenLabel(3), "Should remember label 3");
        assertTrue(loadedClassifier.hasSeenLabel(9), "Should remember label 9");
    }
    
    @Test
    void testSimpleNetStringSerialization() throws IOException {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create and train original model
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 16),
                Feature.oneHot(4),
                Feature.passthrough()
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputSoftmaxCrossEntropy(3));
        
        SimpleNetString originalClassifier = SimpleNet.ofStringClassification(originalNet);
        
        // Train with string labels
        Map<String, Object> input1 = Map.of("feature_0", "text1", "feature_1", 1, "feature_2", 0.5f);
        Map<String, Object> input2 = Map.of("feature_0", "text2", "feature_1", 2, "feature_2", 0.8f);
        Map<String, Object> input3 = Map.of("feature_0", "text3", "feature_1", 0, "feature_2", 0.3f);
        originalClassifier.train(input1, "positive");
        originalClassifier.train(input2, "negative");
        originalClassifier.train(input3, "neutral");
        originalClassifier.train(input1, "positive"); // Train again to stabilize
        
        // Test predictions before serialization
        String originalPred1 = originalClassifier.predictString(input1);
        String originalPred2 = originalClassifier.predictString(input2);
        String originalPred3 = originalClassifier.predictString(input3);
        float originalConf1 = originalClassifier.predictConfidence(input1);
        
        // Serialize the complete SimpleNetString (including dictionaries)
        Path modelFile = tempDir.resolve("simplenet_string.bin");
        originalClassifier.save(modelFile);
        
        // Deserialize complete SimpleNetString
        SimpleNetString loadedClassifier = SimpleNetString.load(modelFile);
        
        // Test predictions after serialization - should be identical
        String loadedPred1 = loadedClassifier.predictString(input1);
        String loadedPred2 = loadedClassifier.predictString(input2);
        String loadedPred3 = loadedClassifier.predictString(input3);
        float loadedConf1 = loadedClassifier.predictConfidence(input1);
        
        // State should be completely preserved
        assertEquals(originalPred1, loadedPred1, "Prediction 1 should be identical after serialization");
        assertEquals(originalPred2, loadedPred2, "Prediction 2 should be identical after serialization");
        assertEquals(originalPred3, loadedPred3, "Prediction 3 should be identical after serialization");
        assertEquals(originalConf1, loadedConf1, 0.001f, "Confidence should be nearly identical after serialization");
        
        // Test class information
        assertEquals(3, loadedClassifier.getClassCount(), "Should have same number of classes");
        assertTrue(loadedClassifier.hasSeenLabel("positive"), "Should remember 'positive' label");
        assertTrue(loadedClassifier.hasSeenLabel("negative"), "Should remember 'negative' label");
        assertTrue(loadedClassifier.hasSeenLabel("neutral"), "Should remember 'neutral' label");
        
        // Test top-K predictions
        String[] originalTopK = originalClassifier.predictTopK(input1, 2);
        String[] loadedTopK = loadedClassifier.predictTopK(input1, 2);
        assertArrayEquals(originalTopK, loadedTopK, "Top-K predictions should be identical after serialization");
    }
    
    @Test
    void testSimpleNetFloatSerialization() throws IOException {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create and train original model
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(50, 8),
                Feature.oneHot(3),
                Feature.passthrough(),
                Feature.passthrough()
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat originalRegressor = SimpleNet.ofFloatRegression(originalNet);
        
        // Train with regression data
        Map<String, Object> input1 = Map.of("feature_0", "house1", "feature_1", 1, "feature_2", 2400.0f, "feature_3", 3.0f);
        Map<String, Object> input2 = Map.of("feature_0", "house2", "feature_1", 2, "feature_2", 3200.0f, "feature_3", 4.0f);
        originalRegressor.train(input1, 485000.0f);
        originalRegressor.train(input2, 625000.0f);
        originalRegressor.train(input1, 520000.0f);
        
        // Test predictions before serialization
        float originalPred1 = originalRegressor.predictFloat(input1);
        float originalPred2 = originalRegressor.predictFloat(input2);
        
        // Serialize the complete SimpleNetFloat (including dictionaries)
        Path modelFile = tempDir.resolve("simplenet_float.bin");
        originalRegressor.save(modelFile);
        
        // Deserialize complete SimpleNetFloat
        SimpleNetFloat loadedRegressor = SimpleNetFloat.load(modelFile);
        
        // Test predictions after serialization - should be identical
        float loadedPred1 = loadedRegressor.predictFloat(input1);
        float loadedPred2 = loadedRegressor.predictFloat(input2);
        
        // State should be completely preserved
        assertEquals(originalPred1, loadedPred1, 0.001f, "Prediction 1 should be nearly identical after serialization");
        assertEquals(originalPred2, loadedPred2, 0.001f, "Prediction 2 should be nearly identical after serialization");
        
        // Test batch predictions
        Object[] inputs = {input1, input2};
        float[] originalBatch = originalRegressor.predict(inputs);
        float[] loadedBatch = loadedRegressor.predict(inputs);
        
        assertEquals(originalBatch.length, loadedBatch.length, "Batch predictions should have same length");
        for (int i = 0; i < originalBatch.length; i++) {
            assertEquals(originalBatch[i], loadedBatch[i], 0.001f, 
                "Batch prediction " + i + " should be nearly identical after serialization");
        }
    }
    
    @Test
    void testSimpleNetMultiFloatFactory() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create neural network for multi-output regression
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100, 16, "market_sector"),
                Feature.oneHot(5, "risk_profile"),
                Feature.passthrough("investment_amount"),
                Feature.passthrough("time_horizon")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(4));  // 4 asset allocations
        
        // Test factory method with named outputs
        String[] assetNames = {"stocks", "bonds", "real_estate", "cash"};
        SimpleNetMultiFloat allocator = SimpleNet.ofMultiFloatRegression(net, assetNames);
        assertNotNull(allocator, "Factory should create SimpleNetMultiFloat instance");
        assertTrue(allocator.hasNamedOutputs(), "Should have named outputs");
        assertEquals(new LinkedHashSet<>(Arrays.asList(assetNames)), allocator.getOutputNames(), "Should preserve output names");
        assertEquals(4, allocator.getOutputCount(), "Should have 4 outputs");
        
        // Test factory method without named outputs
        SimpleNetMultiFloat unnamedAllocator = SimpleNet.ofMultiFloatRegression(net);
        assertNotNull(unnamedAllocator, "Factory should create SimpleNetMultiFloat instance");
        assertFalse(unnamedAllocator.hasNamedOutputs(), "Should not have named outputs");
        assertNull(unnamedAllocator.getOutputNames(), "Should return null for output names");
        assertEquals(4, unnamedAllocator.getOutputCount(), "Should have 4 outputs");
        
        // Test training and prediction
        Map<String, Object> input = Map.of(
            "market_sector", "technology",
            "risk_profile", 2,  // moderate risk
            "investment_amount", 100000.0f,
            "time_horizon", 10.0f  // 10 year horizon
        );
        
        float[] targetAllocation = {0.6f, 0.2f, 0.15f, 0.05f};
        
        // Test training
        assertDoesNotThrow(() -> {
            allocator.train(input, targetAllocation);
            unnamedAllocator.train(input, targetAllocation);
        });
        
        // Test prediction
        float[] allocation = allocator.predictMultiFloat(input);
        assertEquals(4, allocation.length, "Should return 4 allocation values");
        for (float value : allocation) {
            assertTrue(Float.isFinite(value), "All allocation values should be finite");
        }
        
        // Test named prediction
        Map<String, Float> namedAllocation = allocator.predictNamed(input);
        assertEquals(4, namedAllocation.size(), "Should return 4 named allocations");
        assertTrue(namedAllocation.containsKey("stocks"), "Should contain stocks allocation");
        assertTrue(namedAllocation.containsKey("bonds"), "Should contain bonds allocation");
        assertTrue(namedAllocation.containsKey("real_estate"), "Should contain real_estate allocation");
        assertTrue(namedAllocation.containsKey("cash"), "Should contain cash allocation");
        
        // Named prediction should fail for unnamed allocator
        assertThrows(IllegalStateException.class, 
            () -> unnamedAllocator.predictNamed(input),
            "Named prediction should fail without output names");
    }
    
    @Test
    void testSimpleNetMultiFloatSerialization() throws IOException {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create and train original model
        NeuralNet originalNet = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(50, 8, "location"),
                Feature.passthrough("lat"),
                Feature.passthrough("lon")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(3));  // 3D coordinates
        
        String[] coordinateNames = {"x", "y", "z"};
        SimpleNetMultiFloat originalRegressor = SimpleNet.ofMultiFloatRegression(originalNet, coordinateNames);
        
        // Train with coordinate data
        Map<String, Object> input1 = Map.of("location", "location1", "lat", 2.5f, "lon", 1.8f);
        Map<String, Object> input2 = Map.of("location", "location2", "lat", 3.2f, "lon", 0.9f);
        originalRegressor.train(input1, new float[]{1.0f, 2.0f, 3.0f});
        originalRegressor.train(input2, new float[]{4.0f, 5.0f, 6.0f});
        originalRegressor.train(input1, new float[]{1.1f, 2.1f, 3.1f}); // Train again
        
        // Test predictions before serialization
        float[] originalCoords1 = originalRegressor.predictMultiFloat(input1);
        float[] originalCoords2 = originalRegressor.predictMultiFloat(input2);
        Map<String, Float> originalNamed1 = originalRegressor.predictNamed(input1);
        
        // Serialize the complete SimpleNetMultiFloat (including dictionaries and output names)
        Path modelFile = tempDir.resolve("simplenet_multi_float.bin");
        originalRegressor.save(modelFile);
        
        // Deserialize complete SimpleNetMultiFloat
        SimpleNetMultiFloat loadedRegressor = SimpleNetMultiFloat.load(modelFile);
        
        // Test predictions after serialization - should be identical
        float[] loadedCoords1 = loadedRegressor.predictMultiFloat(input1);
        float[] loadedCoords2 = loadedRegressor.predictMultiFloat(input2);
        Map<String, Float> loadedNamed1 = loadedRegressor.predictNamed(input1);
        
        // State should be completely preserved
        assertArrayEquals(originalCoords1, loadedCoords1, 0.001f, "Coordinates 1 should be nearly identical after serialization");
        assertArrayEquals(originalCoords2, loadedCoords2, 0.001f, "Coordinates 2 should be nearly identical after serialization");
        
        // Test output configuration preservation
        assertEquals(3, loadedRegressor.getOutputCount(), "Should have same output count");
        assertTrue(loadedRegressor.hasNamedOutputs(), "Should preserve named outputs flag");
        assertEquals(new LinkedHashSet<>(Arrays.asList(coordinateNames)), loadedRegressor.getOutputNames(), "Should preserve output names");
        
        // Test named predictions preservation
        assertEquals(originalNamed1.size(), loadedNamed1.size(), "Named predictions should have same size");
        for (String key : originalNamed1.keySet()) {
            assertTrue(loadedNamed1.containsKey(key), "Should contain key: " + key);
            assertEquals(originalNamed1.get(key), loadedNamed1.get(key), 0.001f, 
                "Named prediction for " + key + " should be nearly identical");
        }
        
        // Test batch predictions
        Object[] inputs = {input1, input2};
        float[][] originalBatch = originalRegressor.predict(inputs);
        float[][] loadedBatch = loadedRegressor.predict(inputs);
        
        assertEquals(originalBatch.length, loadedBatch.length, "Batch predictions should have same length");
        for (int i = 0; i < originalBatch.length; i++) {
            assertArrayEquals(originalBatch[i], loadedBatch[i], 0.001f, 
                "Batch prediction " + i + " should be nearly identical after serialization");
        }
    }
    
    @Test
    void testSimpleNetMultiFloatValidation() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create classification network (wrong for multi-float regression)
        NeuralNet classificationNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        // Should throw for multi-float regression factory
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, 
            () -> SimpleNet.ofMultiFloatRegression(classificationNet));
        assertTrue(exception.getMessage().contains("regression"), 
            "Error should mention regression requirement");
        
        // Test mismatched output names
        NeuralNet regressionNet = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(3));  // 3 outputs
        
        String[] wrongNames = {"a", "b"};  // Only 2 names for 3 outputs
        IllegalArgumentException exception2 = assertThrows(IllegalArgumentException.class, 
            () -> SimpleNet.ofMultiFloatRegression(regressionNet, wrongNames));
        assertTrue(exception2.getMessage().contains("Output names length"), 
            "Error should mention output names length mismatch");
        
        // Test wrong target array size
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(regressionNet);
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        float[] wrongTargets = {1.0f, 2.0f};  // Only 2 targets for 3 outputs
        
        IllegalArgumentException exception3 = assertThrows(IllegalArgumentException.class, 
            () -> regressor.train(input, wrongTargets));
        assertTrue(exception3.getMessage().contains("Target array length"), 
            "Error should mention target array length mismatch");
    }
    
    // ===============================
    // EDGE CASES AND ROBUSTNESS
    // ===============================
    
    @Test
    void testEmptyModel() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Minimal valid model
        NeuralNet net = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(optimizer)
            .output(Layers.outputSoftmaxCrossEntropy(1));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Should work with minimal input
        float[] input = {1.0f};
        assertDoesNotThrow(() -> {
            classifier.train(input, 0);
            int prediction = classifier.predictInt(input);
            assertEquals(0, prediction, "Single class should always predict 0");
        });
    }
    
    @Test
    void testLargeBatchPrediction() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        
        // Create large batch
        int batchSize = 1000;
        Object[] inputs = new Object[batchSize];
        for (int i = 0; i < batchSize; i++) {
            inputs[i] = createRandomInput(10);
        }
        
        // Should handle large batch efficiently
        long startTime = System.currentTimeMillis();
        float[] predictions = regressor.predict(inputs);
        long endTime = System.currentTimeMillis();
        
        assertEquals(batchSize, predictions.length, "Should return prediction for each input");
        assertTrue(endTime - startTime < 5000, "Should complete large batch in reasonable time");
        
        for (float pred : predictions) {
            assertTrue(Float.isFinite(pred), "All predictions should be finite");
        }
    }
    
    // ===============================
    // HELPER METHODS
    // ===============================
    
    private float[] createRandomInput(int size) {
        float[] input = new float[size];
        for (int i = 0; i < size; i++) {
            input[i] = (float) Math.random();
        }
        return input;
    }
}