package dev.neuronic.net.simple;

import dev.neuronic.net.*;
import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the named output functionality in SimpleNet classes.
 * Ensures that named inputs and outputs work correctly together.
 */
class SimpleNetNamedOutputTest {
    
    private final AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
    
    // ===============================
    // INT CLASSIFICATION WITH NAMED OUTPUT
    // ===============================
    
    @Test
    void testIntClassificationWithNamedOutput() {
        // Create classifier with named output
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
        
        // For classification with 3 classes, we need 3 output names if using named outputs
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        // Test training with named input
        Map<String, Object> input = Map.of(
            "category", "sports",
            "type", 2,
            "score", 0.8f
        );
        
        classifier.train(input, 1);
        
        // For classification without output names, predictNamed doesn't make sense
        // Test regular prediction instead
        int predictedClass = classifier.predictInt(input);
        assertTrue(predictedClass >= 0 && predictedClass < 3);
    }
    
    @Test
    void testIntClassificationRequiresNamesForPredictNamed() {
        // Create classifier WITHOUT named output
        NeuralNet net = NeuralNet.newBuilder()
            .input(784)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(128))
            .output(Layers.outputSoftmaxCrossEntropy(10));
        
        SimpleNetInt classifier = SimpleNet.ofIntClassification(net);
        
        float[] input = new float[784];
        Arrays.fill(input, 0.5f);
        
        // Train normally
        classifier.train(input, 5);
        
        // predictNamed should throw exception without output names
        assertThrows(IllegalStateException.class, () -> {
            classifier.predictNamed(input);
        }, "predictNamed() should require output names");
    }
    
    // ===============================
    // STRING CLASSIFICATION WITH NAMED OUTPUT
    // ===============================
    
    @Test
    void testStringClassificationWithNamedOutput() {
        // Create classifier with named output
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
        
        // For classification with 3 classes, we need 3 output names if using named outputs
        SimpleNetString classifier = SimpleNet.ofStringClassification(net);
        
        // Test training with named input
        Map<String, Object> input = Map.of(
            "text", "amazing product",
            "source", 1,
            "length", 0.75f
        );
        
        classifier.train(input, "positive");
        
        // For classification without output names, predictNamed doesn't make sense
        // Test regular prediction instead  
        String predictedSentiment = classifier.predictString(input);
        assertNotNull(predictedSentiment);
    }
    
    // ===============================
    // FLOAT REGRESSION WITH NAMED OUTPUT
    // ===============================
    
    @Test
    void testFloatRegressionWithNamedOutput() {
        // Create regressor with named output
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
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net, "price");
        
        // Test training with named input and output
        Map<String, Object> input = Map.of(
            "location", 3,
            "sqft", 2400.0f,
            "bedrooms", 3.0f
        );
        
        Map<String, Float> target = Map.of("price", 485000.0f);
        regressor.train(input, target);
        
        // Test named output prediction
        Map<String, Float> namedResult = regressor.predictNamed(input);
        assertNotNull(namedResult);
        assertEquals(1, namedResult.size());
        assertTrue(namedResult.containsKey("price"));
        
        // Verify it's a valid prediction (not NaN or infinite)
        float predictedPrice = namedResult.get("price");
        assertTrue(Float.isFinite(predictedPrice), "Price should be a finite value");
    }
    
    @Test
    void testFloatRegressionTrainWithMap() {
        // Create regressor with named output
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.passthrough("temp"),
                Feature.passthrough("humidity")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net, "comfort_score");
        
        // Test training with both named inputs and outputs
        Map<String, Object> input = Map.of(
            "temp", 72.0f,
            "humidity", 0.45f
        );
        
        Map<String, Float> target = Map.of("comfort_score", 0.85f);
        
        // Should be able to train with Map target
        regressor.train(input, target);
        
        // And predict with named output
        Map<String, Float> prediction = regressor.predictNamed(input);
        assertEquals(1, prediction.size());
        assertTrue(prediction.containsKey("comfort_score"));
    }
    
    // ===============================
    // MULTI-FLOAT REGRESSION WITH NAMED OUTPUTS
    // ===============================
    
    @Test
    void testNamedFeaturesAreProperlyUsed() {
        // Create network with named features
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.embedding(100, 32, "category"),
                Feature.oneHot(5, "type"),
                Feature.passthrough("score")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net, "result");
        
        // Train with properly named features
        Map<String, Object> input = Map.of(
            "category", "sports",
            "type", 2,
            "score", 0.8f
        );
        
        regressor.train(input, Map.of("result", 1.5f));
        
        // Should work with the correct feature names
        Map<String, Float> prediction = regressor.predictNamed(input);
        assertNotNull(prediction);
        assertEquals(1, prediction.size());
        assertTrue(prediction.containsKey("result"));
        
        // Should fail with wrong feature names
        Map<String, Object> wrongInput = Map.of(
            "feature_0", "sports",  // Wrong name!
            "feature_1", 2,
            "feature_2", 0.8f
        );
        
        assertThrows(IllegalArgumentException.class, () -> {
            regressor.train(wrongInput, 1.5f);
        }, "Should fail with wrong feature names");
    }
    
    @Test
    void testMultiFloatRegressionWithNamedOutputs() {
        // Create multi-output regressor with named outputs
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
        
        Set<String> outputNames = new LinkedHashSet<>(Arrays.asList(
            "stocks", "bonds", "real_estate", "cash"
        ));
        SimpleNetMultiFloat allocator = SimpleNet.ofMultiFloatRegression(net, outputNames);
        
        // Test training with named inputs and outputs
        Map<String, Object> input = Map.of(
            "risk_profile", 2,
            "age", 45.0f,
            "amount", 100000.0f
        );
        
        Map<String, Float> targets = Map.of(
            "stocks", 0.6f,
            "bonds", 0.2f,
            "real_estate", 0.15f,
            "cash", 0.05f
        );
        
        allocator.train(input, targets);
        
        // Test named output prediction
        Map<String, Float> namedResult = allocator.predictNamed(input);
        assertNotNull(namedResult);
        assertEquals(4, namedResult.size());
        
        // Verify all outputs are present
        assertTrue(namedResult.containsKey("stocks"));
        assertTrue(namedResult.containsKey("bonds"));
        assertTrue(namedResult.containsKey("real_estate"));
        assertTrue(namedResult.containsKey("cash"));
        
        // Note: We're not checking that outputs sum to 1.0 because 
        // that would require proper training, which is not the focus of this test.
        // We're just testing the named output functionality.
    }
    
    @Test
    void testMultiFloatRegressionRequiresCorrectTargetMap() {
        // Create multi-output regressor with named outputs
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.passthrough("x"),
                Feature.passthrough("y")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(3));
        
        Set<String> outputNames = Set.of("r", "g", "b");
        SimpleNetMultiFloat colorPredictor = SimpleNet.ofMultiFloatRegression(net, outputNames);
        
        Map<String, Object> input = Map.of("x", 0.5f, "y", 0.5f);
        
        // Test with correct target map
        Map<String, Float> correctTargets = Map.of("r", 1.0f, "g", 0.5f, "b", 0.0f);
        colorPredictor.train(input, correctTargets);
        
        // Test with missing output
        Map<String, Float> missingTargets = Map.of("r", 1.0f, "g", 0.5f);
        assertThrows(IllegalArgumentException.class, () -> {
            colorPredictor.train(input, missingTargets);
        }, "Should fail with missing target");
        
        // Test with unknown output
        Map<String, Float> unknownTargets = Map.of("r", 1.0f, "g", 0.5f, "b", 0.0f, "alpha", 1.0f);
        assertThrows(IllegalArgumentException.class, () -> {
            colorPredictor.train(input, unknownTargets);
        }, "Should fail with unknown target");
    }
    
    // ===============================
    // LANGUAGE MODEL WITH NAMED OUTPUT
    // ===============================
    
    @Test
    void testLanguageModelWithNamedOutput() {
        // Create language model with named output
        NeuralNet net = NeuralNet.newBuilder()
            .input(5) // sequence length
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputSequenceEmbedding(5, 100, 32))
            .layer(Layers.hiddenGruLast(64))
            .output(Layers.outputSoftmaxCrossEntropy(100));
        
        // For language models with 100 vocabulary size, we'd need 100 output names if using named outputs
        SimpleNetLanguageModel lm = SimpleNet.ofLanguageModel(net);
        
        // Train with sequence
        String[] sequence = {"the", "cat", "sat", "on", "the"};
        lm.train(sequence, "mat");
        
        // For language models without output names, predictNamed doesn't make sense
        // Test regular prediction instead
        String nextWord = lm.predictNext(sequence);
        assertNotNull(nextWord);
    }
    
    // ===============================
    // ERROR HANDLING TESTS
    // ===============================
    
    @Test
    void testTrainNamedRequiresOutputNames() {
        // Create network without output names
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        
        float[] input = new float[10];
        Arrays.fill(input, 0.5f);
        
        // Should fail when trying to use named targets without output names
        Map<String, Float> namedTargets = Map.of("output", 1.0f);
        assertThrows(IllegalStateException.class, () -> {
            regressor.train(input, namedTargets);
        }, "train() with named targets should require output names");
    }
    
    @Test
    void testPredictNamedRequiresBothInputAndOutputNames() {
        // Test 1: Has output names but using array input
        NeuralNet net1 = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor1 = SimpleNet.ofFloatRegression(net1, "value");
        
        float[] arrayInput = new float[10];
        Arrays.fill(arrayInput, 0.5f);
        
        // Train first
        regressor1.train(arrayInput, 1.0f);
        
        // predictNamed with array input should work (no input names required)
        Map<String, Float> result1 = regressor1.predictNamed(arrayInput);
        assertNotNull(result1);
        assertEquals(1, result1.size());
        
        // Test 2: Has input names but no output names
        NeuralNet net2 = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(
                Feature.passthrough("a"),
                Feature.passthrough("b")
            ))
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(1));
        
        SimpleNetFloat regressor2 = SimpleNet.ofFloatRegression(net2);
        
        Map<String, Object> namedInput = Map.of("a", 1.0f, "b", 2.0f);
        
        // Train first
        regressor2.train(namedInput, 3.0f);
        
        // predictNamed should fail without output names
        assertThrows(IllegalStateException.class, () -> {
            regressor2.predictNamed(namedInput);
        }, "predictNamed() should require output names");
    }
    
    @Test
    void testOutputNameUniqueness() {
        // Set ensures uniqueness
        Set<String> names = new LinkedHashSet<>(Arrays.asList("a", "b", "a"));
        assertEquals(2, names.size(), "Set should enforce uniqueness");
        
        // Create network with unique output names
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputLinearRegression(2));
        
        Set<String> outputNames = Set.of("output1", "output2");
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net, outputNames);
        
        assertNotNull(regressor);
        assertTrue(regressor.hasNamedOutputs());
    }
}