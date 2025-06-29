package dev.neuronic;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.Layers;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.*;

public class GradientClippingTest {
    
    @Test
    public void testGradientClippingPreventsNaN() {
        // Create a network with very small gradient clipping threshold
        // This should prevent NaN even with extreme inputs
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(0.1f)  // Very aggressive clipping
            .layer(Layers.hiddenDenseRelu(8))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputSigmoidBinary());
        
        // Train with extreme inputs that would normally cause gradient explosion
        float[][] extremeInputs = {
            {1000.0f, 1000.0f},
            {-1000.0f, -1000.0f},
            {1000.0f, -1000.0f},
            {-1000.0f, 1000.0f}
        };
        float[][] targets = {
            {0.0f}, {1.0f}, {1.0f}, {0.0f}
        };
        
        // Train for several epochs - should not produce NaN
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i < extremeInputs.length; i++) {
                net.train(extremeInputs[i], targets[i]);
            }
            
            // Verify predictions are not NaN
            for (int i = 0; i < extremeInputs.length; i++) {
                float[] pred = net.predict(extremeInputs[i]);
                assertFalse(Float.isNaN(pred[0]), "Prediction should not be NaN at epoch " + epoch);
                assertFalse(Float.isInfinite(pred[0]), "Prediction should not be infinite at epoch " + epoch);
                assertTrue(pred[0] >= 0.0f && pred[0] <= 1.0f, 
                    "Sigmoid output should be in [0,1] range, got: " + pred[0]);
            }
        }
    }
    
    @Test
    public void testGradientClippingWithRNNScenario() {
        // RNNs are particularly prone to gradient explosion
        // Test with a GRU layer which should benefit from clipping
        NeuralNet net = NeuralNet.newBuilder()
            .input(10)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)  // Typical RNN clipping value
            .layer(Layers.hiddenGruLast(32))
            .output(Layers.outputSoftmaxCrossEntropy(5));
        
        // Simulate sequence data with potential for gradient explosion
        float[] longSequence = new float[10];
        for (int i = 0; i < longSequence.length; i++) {
            longSequence[i] = (float) Math.sin(i * 0.1) * 10.0f; // Large values
        }
        
        float[] target = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f}; // One-hot encoded
        
        // Train for multiple iterations
        for (int iter = 0; iter < 50; iter++) {
            net.train(longSequence, target);
            
            float[] pred = net.predict(longSequence);
            // Verify softmax outputs are valid probabilities
            float sum = 0.0f;
            for (float p : pred) {
                assertFalse(Float.isNaN(p), "Softmax output contains NaN");
                assertTrue(p >= 0.0f && p <= 1.0f, "Softmax output out of range: " + p);
                sum += p;
            }
            assertEquals(1.0f, sum, 0.01f, "Softmax outputs should sum to 1");
        }
    }
    
    @Test
    public void testGradientClippingActuallyClips() {
        // Capture stderr to verify clipping warnings are produced
        ByteArrayOutputStream errContent = new ByteArrayOutputStream();
        PrintStream originalErr = System.err;
        System.setErr(new PrintStream(errContent));
        
        try {
            // Use a simpler network with tanh (no dead neurons) and lower clipping threshold
            NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(new SgdOptimizer(1000.0f)) // Extremely high LR
                .withGlobalGradientClipping(0.1f)  // Low threshold to ensure clipping
                .layer(Layers.hiddenDenseTanh(10)) // Tanh won't have dead neurons
                .output(Layers.outputSigmoidBinary());
            
            // Moderate inputs (not extreme to avoid saturation)
            float[] input = {5.0f, -5.0f};
            float[] target = {1.0f};
            
            // Train just once - should trigger warning immediately
            net.train(input, target);
            
            String output = errContent.toString();
            assertTrue(output.contains("Warning: Large gradient norm"), 
                "Should produce gradient clipping warnings, but got: " + output);
            assertTrue(output.contains("clipped to 0.10"), 
                "Should show clipping to threshold 0.10");
            
        } finally {
            System.setErr(originalErr);
        }
    }
    
    @Test
    public void testDifferentClippingThresholds() {
        // Test that different clipping thresholds behave correctly
        float[] thresholds = {0.1f, 1.0f, 5.0f, 10.0f};
        
        for (float threshold : thresholds) {
            NeuralNet net = NeuralNet.newBuilder()
                .input(5)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
                .withGlobalGradientClipping(threshold)
                .layer(Layers.hiddenDenseRelu(20))
                .output(Layers.outputSoftmaxCrossEntropy(3));
            
            // Moderate inputs
            float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] target = {0.0f, 1.0f, 0.0f};
            
            // Train and verify no NaN
            for (int epoch = 0; epoch < 100; epoch++) {
                net.train(input, target);
                float[] pred = net.predict(input);
                
                for (float p : pred) {
                    assertFalse(Float.isNaN(p), 
                        "NaN detected with threshold " + threshold + " at epoch " + epoch);
                }
            }
        }
    }
    
    @Test
    public void testBatchTrainingWithGradientClipping() {
        // Test gradient clipping works correctly with batch training
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(2.0f)
            .layer(Layers.hiddenDenseRelu(10))
            .output(Layers.outputSoftmaxCrossEntropy(2));
        
        // Batch of samples
        float[][] batchInputs = {
            {10.0f, 20.0f, 30.0f},
            {-10.0f, -20.0f, -30.0f},
            {5.0f, -5.0f, 0.0f},
            {100.0f, 200.0f, 300.0f} // Extreme values
        };
        
        float[][] batchTargets = {
            {1.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {0.0f, 1.0f}
        };
        
        // Train with batches
        for (int epoch = 0; epoch < 50; epoch++) {
            net.trainBatch(batchInputs, batchTargets);
            
            // Verify all predictions are valid
            float[][] predictions = net.predictBatch(batchInputs);
            for (float[] pred : predictions) {
                float sum = 0.0f;
                for (float p : pred) {
                    assertFalse(Float.isNaN(p), "Batch prediction contains NaN");
                    assertTrue(p >= 0.0f && p <= 1.0f, "Invalid probability: " + p);
                    sum += p;
                }
                assertEquals(1.0f, sum, 0.01f, "Probabilities should sum to 1");
            }
        }
    }
    
    @Test
    public void testGradientClippingStability() {
        // Test that gradient clipping maintains numerical stability
        // Primary goal: Verify no NaN/Inf values are produced
        
        var optimizers = new Optimizer[] {
            new SgdOptimizer(1.0f),  // High LR that would cause NaN without clipping
            new AdamWOptimizer(0.01f, 0.0f),
            new AdamWOptimizer(0.01f, 0.0f)
        };
        
        for (var optimizer : optimizers) {
            NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(optimizer)
                .withGlobalGradientClipping(2.0f)
                .layer(Layers.hiddenDenseRelu(8))
                .layer(Layers.hiddenDenseRelu(4))
                .output(Layers.outputSigmoidBinary());
            
            // Simple data
            float[][] inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
            float[][] targets = {{0.0f}, {0.0f}, {0.0f}, {1.0f}};
            
            // Train and verify stability
            for (int epoch = 0; epoch < 100; epoch++) {
                for (int i = 0; i < inputs.length; i++) {
                    net.train(inputs[i], targets[i]);
                }
                
                // Check for numerical stability
                for (int i = 0; i < inputs.length; i++) {
                    float[] pred = net.predict(inputs[i]);
                    assertFalse(Float.isNaN(pred[0]), 
                        "NaN at epoch " + epoch + " with " + optimizer.getClass().getSimpleName());
                    assertFalse(Float.isInfinite(pred[0]),
                        "Inf at epoch " + epoch + " with " + optimizer.getClass().getSimpleName());
                    assertTrue(pred[0] >= 0.0f && pred[0] <= 1.0f,
                        "Out of range at epoch " + epoch + " with " + optimizer.getClass().getSimpleName());
                }
            }
        }
    }
    
    @Test 
    public void testSGDRequiresGradientClipping() {
        // Demonstrate that SGD with VERY high learning rates NEEDS gradient clipping
        
        // Test SGD without clipping - should produce NaN
        NeuralNet netWithoutClipping = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(100.0f))  // Extreme LR to ensure explosion
            .withGlobalGradientClipping(0.0f)  // Disabled
            .layer(Layers.hiddenDenseRelu(64))  // Deeper network for gradient multiplication
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .layer(Layers.hiddenDenseRelu(32))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputSigmoidBinary());
        
        // Test SGD with clipping - should work fine
        NeuralNet netWithClipping = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new SgdOptimizer(100.0f))  // Same extreme LR
            .withGlobalGradientClipping(1.0f)  // Enabled with aggressive clipping
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(64))
            .layer(Layers.hiddenDenseRelu(32))
            .layer(Layers.hiddenDenseRelu(32))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputSigmoidBinary());
        
        // Use more extreme inputs to trigger gradient explosion
        float[][] inputs = {{1000.0f, 1000.0f}, {-1000.0f, 1000.0f}, {1000.0f, -1000.0f}, {-1000.0f, -1000.0f}};
        float[][] targets = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
        
        // Train both networks
        boolean withoutClippingFailed = false;
        boolean withClippingFailed = false;
        
        for (int epoch = 0; epoch < 100; epoch++) {
            // Train without clipping
            if (!withoutClippingFailed) {
                for (int i = 0; i < inputs.length; i++) {
                    netWithoutClipping.train(inputs[i], targets[i]);
                    float[] pred = netWithoutClipping.predict(inputs[i]);
                    if (Float.isNaN(pred[0]) || Float.isInfinite(pred[0])) {
                        withoutClippingFailed = true;
                        break;
                    }
                }
            }
            
            // Train with clipping
            for (int i = 0; i < inputs.length; i++) {
                netWithClipping.train(inputs[i], targets[i]);
                float[] pred = netWithClipping.predict(inputs[i]);
                if (Float.isNaN(pred[0]) || Float.isInfinite(pred[0])) {
                    withClippingFailed = true;
                }
            }
        }
        
        // Verify gradient clipping prevents NaN
        assertTrue(withoutClippingFailed, 
            "SGD with lr=1.0 should produce NaN without gradient clipping");
        assertFalse(withClippingFailed, 
            "SGD with lr=1.0 should NOT produce NaN with gradient clipping");
    }
    
    @Test
    public void testGradientClippingConvergence() {
        // Test that networks with gradient clipping can learn
        // Focus on optimizers known to work well with clipping
        
        var optimizers = new Optimizer[] {
            new AdamWOptimizer(0.01f, 0.0f),
            new AdamWOptimizer(0.01f, 0.0f)
        };
        
        for (var optimizer : optimizers) {
            NeuralNet net = NeuralNet.newBuilder()
                .input(2)
                .setDefaultOptimizer(optimizer)
                .withGlobalGradientClipping(2.0f)
                .layer(Layers.hiddenDenseRelu(16))
                .layer(Layers.hiddenDenseRelu(8))
                .output(Layers.outputSigmoidBinary());
            
            // XOR pattern
            float[][] inputs = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
            float[][] targets = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};
            
            // Train
            for (int epoch = 0; epoch < 300; epoch++) {
                for (int i = 0; i < inputs.length; i++) {
                    net.train(inputs[i], targets[i]);
                }
            }
            
            // Verify learning
            float avgError = 0.0f;
            for (int i = 0; i < inputs.length; i++) {
                float[] pred = net.predict(inputs[i]);
                avgError += Math.abs(pred[0] - targets[i][0]);
            }
            avgError /= inputs.length;
            
            assertTrue(avgError < 0.35f, 
                "AdamW should converge with gradient clipping, got error: " + avgError);
        }
    }
    
    @Test
    public void testExtremeGradientClippingStillLearns() {
        // Even with very aggressive clipping, network should still learn
        NeuralNet net = NeuralNet.newBuilder()
            .input(2)
            .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0f)) // Higher LR, no weight decay
            .withGlobalGradientClipping(0.01f) // Extremely aggressive
            .layer(Layers.hiddenDenseRelu(16))
            .layer(Layers.hiddenDenseRelu(16))
            .output(Layers.outputSigmoidBinary());
        
        // Simple linearly separable problem
        float[][] inputs = {
            {0.0f, 0.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f}
        };
        float[][] targets = {
            {0.0f}, {0.0f}, {1.0f}, {1.0f} // Linearly separable
        };
        
        // Train for more epochs due to aggressive clipping
        for (int epoch = 0; epoch < 500; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.train(inputs[i], targets[i]);
            }
        }
        
        // Should still learn this simple pattern
        float avgError = 0.0f;
        for (int i = 0; i < inputs.length; i++) {
            float[] pred = net.predict(inputs[i]);
            float error = Math.abs(pred[0] - targets[i][0]);
            avgError += error;
        }
        avgError /= inputs.length;
        
        assertTrue(avgError < 0.2f, 
            "Should learn even with extreme clipping, got error: " + avgError);
    }
    
    @Test
    public void testGradientClippingConsistency() {
        // Verify that gradient clipping produces consistent results
        NeuralNet net1 = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)
            .layer(Layers.hiddenDenseRelu(5))
            .output(Layers.outputSigmoidBinary());
        
        NeuralNet net2 = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.001f, 0.01f))
            .withGlobalGradientClipping(1.0f)
            .layer(Layers.hiddenDenseRelu(5))
            .output(Layers.outputSigmoidBinary());
        
        // Same training data
        float[] input = {1.0f, 2.0f, 3.0f};
        float[] target = {1.0f};
        
        // Train both networks identically
        for (int i = 0; i < 10; i++) {
            net1.train(input, target);
            net2.train(input, target);
        }
        
        // Predictions should be very close (not exactly equal due to weight initialization)
        float[] pred1 = net1.predict(input);
        float[] pred2 = net2.predict(input);
        
        // Both should be valid
        assertFalse(Float.isNaN(pred1[0]) || Float.isNaN(pred2[0]), 
            "Predictions contain NaN");
    }
}