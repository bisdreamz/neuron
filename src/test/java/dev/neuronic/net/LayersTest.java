package dev.neuronic.net;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.outputs.*;
import dev.neuronic.net.activators.*;
import dev.neuronic.net.outputs.LinearRegressionOutput;
import dev.neuronic.net.outputs.MultiLabelSigmoidOutput;
import dev.neuronic.net.outputs.SigmoidBinaryCrossEntropyOutput;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LayersTest {

    private final SgdOptimizer optimizer = new SgdOptimizer(0.01f);

    // Hidden Layer Tests
    
    @Test
    void testHiddenDenseRelu() {
        Layer.Spec spec = Layers.hiddenDenseRelu(128, optimizer);
        
        assertEquals(128, spec.getOutputSize());
        
        Layer layer = spec.create(784);
        assertNotNull(layer);
        
        // Test forward pass
        float[] input = new float[784];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() - 0.5f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(128, context.outputs().length);
        
        // ReLU should produce non-negative outputs
        for (float output : context.outputs()) {
            assertTrue(output >= 0.0f, "ReLU output should be non-negative");
        }
    }

    @Test
    void testHiddenDenseTanh() {
        Layer.Spec spec = Layers.hiddenDenseTanh(64, optimizer);
        
        assertEquals(64, spec.getOutputSize());
        
        Layer layer = spec.create(128);
        assertNotNull(layer);
        
        // Test forward pass
        float[] input = new float[128];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 2.0f - 1.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(64, context.outputs().length);
        
        // Tanh should produce outputs in range (-1, 1)
        for (float output : context.outputs()) {
            assertTrue(output > -1.0f && output < 1.0f, 
                "Tanh output should be in range (-1, 1): " + output);
        }
    }

    @Test
    void testHiddenDenseSigmoid() {
        Layer.Spec spec = Layers.hiddenDenseSigmoid(32, optimizer);
        
        assertEquals(32, spec.getOutputSize());
        
        Layer layer = spec.create(64);
        assertNotNull(layer);
        
        // Test forward pass
        float[] input = new float[64];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 4.0f - 2.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(32, context.outputs().length);
        
        // Sigmoid should produce outputs in range (0, 1)
        for (float output : context.outputs()) {
            assertTrue(output > 0.0f && output < 1.0f, 
                "Sigmoid output should be in range (0, 1): " + output);
        }
    }

    @Test
    void testHiddenDenseLinear() {
        Layer.Spec spec = Layers.hiddenDenseLinear(16, optimizer);
        
        assertEquals(16, spec.getOutputSize());
        
        Layer layer = spec.create(32);
        assertNotNull(layer);
        
        // Test forward pass
        float[] input = new float[32];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 2.0f - 1.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(16, context.outputs().length);
        
        // Linear activation should not constrain output range
        // Just verify we get reasonable outputs (not NaN/Infinite)
        for (float output : context.outputs()) {
            assertFalse(Float.isNaN(output), "Linear output should not be NaN");
            assertFalse(Float.isInfinite(output), "Linear output should not be infinite");
        }
    }

    // Output Layer Tests

    @Test
    void testOutputSoftmaxCrossEntropy() {
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(10, optimizer);
        
        assertEquals(10, spec.getOutputSize());
        
        Layer layer = spec.create(128);
        assertNotNull(layer);
        assertInstanceOf(SoftmaxCrossEntropyOutput.class, layer);
        
        // Test forward pass - should produce softmax probabilities
        float[] input = new float[128];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 2.0f - 1.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(10, context.outputs().length);
        
        // Softmax outputs should sum to 1
        float sum = 0;
        for (float output : context.outputs()) {
            assertTrue(output > 0.0f, "Softmax output should be positive");
            sum += output;
        }
        assertEquals(1.0f, sum, 0.001f, "Softmax outputs should sum to 1");
    }

    @Test
    void testOutputLinearRegression() {
        Layer.Spec spec = Layers.outputLinearRegression(1, optimizer);
        
        assertEquals(1, spec.getOutputSize());
        
        Layer layer = spec.create(64);
        assertNotNull(layer);
        assertInstanceOf(LinearRegressionOutput.class, layer);
        
        // Test forward pass
        float[] input = new float[64];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 2.0f - 1.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(1, context.outputs().length);
        
        // Linear regression can output any value
        assertFalse(Float.isNaN(context.outputs()[0]), "Output should not be NaN");
        assertFalse(Float.isInfinite(context.outputs()[0]), "Output should not be infinite");
    }

    @Test
    void testOutputSigmoidBinaryCrossEntropy() {
        Layer.Spec spec = Layers.outputSigmoidBinary(optimizer);
        
        assertEquals(1, spec.getOutputSize());
        
        Layer layer = spec.create(32);
        assertNotNull(layer);
        assertInstanceOf(SigmoidBinaryCrossEntropyOutput.class, layer);
        
        // Test forward pass - should produce sigmoid probability
        float[] input = new float[32];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 4.0f - 2.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(1, context.outputs().length);
        
        // Sigmoid output should be in range (0, 1)
        float output = context.outputs()[0];
        assertTrue(output > 0.0f && output < 1.0f, 
            "Sigmoid output should be in range (0, 1): " + output);
    }

    @Test
    void testOutputMultiLabelSigmoid() {
        Layer.Spec spec = Layers.outputMultiLabel(5, optimizer);
        
        assertEquals(5, spec.getOutputSize());
        
        Layer layer = spec.create(64);
        assertNotNull(layer);
        assertInstanceOf(MultiLabelSigmoidOutput.class, layer);
        
        // Test forward pass - should produce independent sigmoid probabilities
        float[] input = new float[64];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random() * 4.0f - 2.0f;
        }
        
        Layer.LayerContext context = layer.forward(input);
        assertEquals(5, context.outputs().length);
        
        // Each output should be sigmoid probability in range (0, 1)
        for (float output : context.outputs()) {
            assertTrue(output > 0.0f && output < 1.0f, 
                "Multi-label sigmoid output should be in range (0, 1): " + output);
        }
        
        // Unlike softmax, multi-label outputs don't need to sum to 1
        float sum = 0;
        for (float output : context.outputs()) {
            sum += output;
        }
        // Sum can be anything from 0 to 5 (number of labels)
        assertTrue(sum >= 0.0f && sum <= 5.0f, "Sum should be reasonable");
    }

    // Integration Tests

    @Test
    void testNeuralNetworkConstruction() {
        // Test building a complete network using Layers utility
        NeuralNet net = NeuralNet.newBuilder()
            .input(784)
            .layer(Layers.hiddenDenseRelu(256, optimizer))
            .layer(Layers.hiddenDenseTanh(128, optimizer))
            .layer(Layers.hiddenDenseSigmoid(64, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(10, optimizer));
        
        assertNotNull(net);
        
        // Test prediction
        float[] input = new float[784];
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.random();
        }
        
        float[] prediction = net.predict(input);
        assertEquals(10, prediction.length);
        
        // Final output should be softmax probabilities
        float sum = 0;
        for (float prob : prediction) {
            assertTrue(prob > 0.0f, "Probability should be positive");
            sum += prob;
        }
        assertEquals(1.0f, sum, 0.001f, "Probabilities should sum to 1");
    }

    @Test
    void testNeuralNetworkTraining() {
        // Test training a network built with Layers utility
        NeuralNet net = NeuralNet.newBuilder()
            .input(4)
            .layer(Layers.hiddenDenseRelu(8, optimizer))
            .output(Layers.outputSoftmaxCrossEntropy(3, optimizer));
        
        assertNotNull(net);
        
        // Get initial prediction
        float[] input = {0.5f, -0.3f, 0.8f, -0.1f};
        float[] initialPrediction = net.predict(input).clone();
        
        // Train on one sample
        float[] targets = {1.0f, 0.0f, 0.0f}; // One-hot for class 0
        net.train(input, targets);
        
        // Get prediction after training
        float[] trainedPrediction = net.predict(input);
        
        // Predictions should be different after training
        boolean changed = false;
        for (int i = 0; i < initialPrediction.length; i++) {
            if (Math.abs(initialPrediction[i] - trainedPrediction[i]) > 0.001f) {
                changed = true;
                break;
            }
        }
        assertTrue(changed, "Network should learn and change predictions");
        
        // Final output should still be valid softmax probabilities
        float sum = 0;
        for (float prob : trainedPrediction) {
            assertTrue(prob > 0.0f, "Probability should be positive");
            sum += prob;
        }
        assertEquals(1.0f, sum, 0.001f, "Probabilities should sum to 1");
    }

    @Test
    void testValidParameters() {
        // Test that normal parameters work
        assertDoesNotThrow(() -> {
            Layers.hiddenDenseRelu(10, optimizer);
        }, "Should work with valid parameters");
        
        assertDoesNotThrow(() -> {
            Layers.outputSoftmaxCrossEntropy(10, optimizer);
        }, "Should work with valid parameters");
        
        // Test edge cases that should work
        assertDoesNotThrow(() -> {
            Layers.hiddenDenseRelu(1, optimizer); // Single neuron should work
        }, "Should work with single neuron");
    }
}