package dev.neuronic.net.simple;

import dev.neuronic.net.Layers;
import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.layers.Feature;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that SimpleNet properly supports HuberRegressionOutput layers.
 */
class SimpleNetHuberRegressionTest {
    
    @Test
    void testSimpleNetFloatWithHuberRegression() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create a network with Huber regression output
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputHuberRegression(1)); // Huber regression instead of Linear
        
        // This should work now that we check for RegressionOutput interface
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        
        // Train with some data
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        regressor.train(input, 5.0f);
        
        // Make prediction
        float prediction = regressor.predictFloat(input);
        assertNotNull(prediction);
    }
    
    @Test
    void testSimpleNetMultiFloatWithHuberRegression() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create a multi-output network with Huber regression
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .setDefaultOptimizer(optimizer)
            .layer(Layers.hiddenDenseRelu(32))
            .output(Layers.outputHuberRegression(3)); // Multiple outputs
        
        // This should work with multi-output regression too
        SimpleNetMultiFloat regressor = SimpleNet.ofMultiFloatRegression(net, 
            new String[]{"x", "y", "z"});
        
        // Train with some data
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targets = {5.0f, 6.0f, 7.0f};
        regressor.train(input, targets);
        
        // Make prediction
        float[] predictions = regressor.predictMultiFloat(input);
        assertEquals(3, predictions.length);
    }
    
    @Test
    void testHuberRegressionWithMixedFeatures() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // Create network with mixed features and Huber regression
        NeuralNet net = NeuralNet.newBuilder()
            .setDefaultOptimizer(optimizer)
            .layer(Layers.inputMixed(optimizer,
                Feature.embedding(100_000, 4, "publisher_id"),
                Feature.oneHot(200, "placement_type"),
                Feature.autoScale(0f, 20f, "bid_floor"),
                Feature.passthrough("other_feature")
            ))
            .layer(Layers.hiddenDenseRelu(64))
            .output(Layers.outputHuberRegression(1)); // Use default delta
        
        SimpleNetFloat regressor = SimpleNet.ofFloatRegression(net);
        
        // Train with map input
        Map<String, Object> input = Map.of(
            "publisher_id", 12345,
            "placement_type", 2,
            "bid_floor", 5.5f,
            "other_feature", 0.8f
        );
        
        regressor.train(input, 2.5f);
        
        // Predict
        float prediction = regressor.predictFloat(input);
        assertNotNull(prediction);
    }
    
    @Test
    void testAllFeaturesNamedRequirement() {
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        // This should fail because not all features are named
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> {
            NeuralNet.newBuilder()
                .setDefaultOptimizer(optimizer)
                .layer(Layers.inputMixed(optimizer,
                    Feature.embedding(100_000, 4, "publisher_id"),
                    Feature.oneHot(200, "placement_type"),
                    Feature.autoScale(0f, 20f, "bid_floor"),
                    Feature.passthrough() // Missing name!
                ))
                .layer(Layers.hiddenDenseRelu(64))
                .output(Layers.outputHuberRegression(1));
        });
        
        assertTrue(ex.getMessage().contains("Feature naming must be all-or-nothing"));
        assertTrue(ex.getMessage().contains("Feature.passthrough(\"your_feature_name\")"));
    }
}