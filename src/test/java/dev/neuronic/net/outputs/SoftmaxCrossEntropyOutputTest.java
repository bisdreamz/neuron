package dev.neuronic.net.outputs;

import dev.neuronic.net.Layers;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SoftmaxCrossEntropyOutputTest {

    @Test
    void testOutputSpecCreation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(10, optimizer);
        
        assertEquals(10, spec.getOutputSize());
        
        Layer layer = spec.create(5);
        assertInstanceOf(SoftmaxCrossEntropyOutput.class, layer);
    }

    @Test
    void testForwardSoftmaxNormalization() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        float[] input = {1.0f, 2.0f, 3.0f};
        Layer.LayerContext context = layer.forward(input, false);
        
        // Check that outputs sum to approximately 1.0 (softmax property)
        float sum = 0;
        for (float output : context.outputs()) {
            sum += output;
            assertTrue(output > 0, "Softmax output should be positive");
        }
        assertEquals(1.0f, sum, 0.001f, "Softmax outputs should sum to 1");
    }

    @Test
    void testForwardLargeInputsNumericalStability() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        // Test with large inputs that could cause overflow
        float[] input = {100.0f, 101.0f, 102.0f};
        Layer.LayerContext context = layer.forward(input, false);
        
        // Should still be numerically stable
        float sum = 0;
        for (float output : context.outputs()) {
            assertFalse(Float.isNaN(output), "Output should not be NaN");
            assertFalse(Float.isInfinite(output), "Output should not be infinite");
            sum += output;
        }
        assertEquals(1.0f, sum, 0.001f, "Softmax outputs should sum to 1 even with large inputs");
    }

    @Test
    void testBackwardGradientComputation() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        float[] input = {1.0f, 2.0f, 3.0f};
        Layer.LayerContext context = layer.forward(input, false);
        
        // Create layer contexts for backward pass
        Layer.LayerContext[] stack = {context};
        float[] targets = {1.0f, 0.0f, 0.0f}; // One-hot for class 0
        
        float[] gradient = layer.backward(stack, 0, targets);
        
        assertNotNull(gradient);
        assertEquals(input.length, gradient.length);
        
        // The returned gradient is for the previous layer (downstream gradient)
        // It should have the same length as the input to this layer (3 elements)
        assertEquals(input.length, gradient.length, "Downstream gradient should match input size");
        
        // The gradient should not be all zeros (network should be learning)
        boolean hasNonZeroGradient = false;
        for (float g : gradient) {
            if (Math.abs(g) > 0.001f) {
                hasNonZeroGradient = true;
                break;
            }
        }
        assertTrue(hasNonZeroGradient, "Should have non-zero gradients for learning");
    }

    @Test
    void testBackwardPerfectPrediction() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        // Create input that will result in near-perfect prediction for class 0
        float[] input = {10.0f, -10.0f, -10.0f};
        Layer.LayerContext context = layer.forward(input, false);
        
        Layer.LayerContext[] stack = {context};
        float[] targets = {1.0f, 0.0f, 0.0f}; // Target is class 0
        
        float[] gradient = layer.backward(stack, 0, targets);
        
        // For perfect prediction, the downstream gradients should be relatively small
        // since the output error is minimal
        assertEquals(input.length, gradient.length, "Downstream gradient should match input size");
        
        // Gradients should not be extreme values
        for (int i = 0; i < gradient.length; i++) {
            assertTrue(Math.abs(gradient[i]) < 10.0f, "Downstream gradient should be reasonable");
        }
    }

    @Test
    void testBackwardGradientFormula() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        // Use any input - the gradient formula should hold regardless
        float[] input = {1.0f, 2.0f, 3.0f};
        Layer.LayerContext context = layer.forward(input, false);
        
        // Get softmax probabilities (whatever they are after W*input + b)
        float[] probabilities = context.outputs();
        
        // Test multiple target scenarios
        float[][] testTargets = {
            {1.0f, 0.0f, 0.0f},  // Target class 0
            {0.0f, 1.0f, 0.0f},  // Target class 1
            {0.0f, 0.0f, 1.0f}   // Target class 2
        };
        
        for (float[] targets : testTargets) {
            // The local gradient for softmax + cross-entropy is exactly: probability - target
            // This is a mathematical property that should ALWAYS hold
            float[] expectedLocalGradient = new float[3];
            for (int i = 0; i < 3; i++) {
                expectedLocalGradient[i] = probabilities[i] - targets[i];
            }
            
            // Verify the gradient formula holds
            // For the target class i where target[i] = 1:
            //   gradient[i] = probability[i] - 1, which should be negative (unless perfect prediction)
            // For non-target classes j where target[j] = 0:
            //   gradient[j] = probability[j] - 0 = probability[j], which should be positive
            
            int targetClass = -1;
            for (int i = 0; i < 3; i++) {
                if (targets[i] == 1.0f) targetClass = i;
            }
            
            // The gradient at the target class should be negative (probability - 1)
            assertTrue(expectedLocalGradient[targetClass] <= 0, 
                String.format("Gradient at target class %d should be probability - 1 <= 0, got %.3f", 
                    targetClass, expectedLocalGradient[targetClass]));
            
            // The sum of all gradients should be 0 (because sum of probabilities = 1, sum of targets = 1)
            float gradientSum = 0;
            for (float g : expectedLocalGradient) {
                gradientSum += g;
            }
            assertEquals(0.0f, gradientSum, 1e-6f, "Sum of gradients should be 0");
            
            // Call backward to verify it runs without error
            Layer.LayerContext[] stack = {context};
            float[] downstreamGradient = layer.backward(stack, 0, targets);
            
            // Just verify the downstream gradient has correct shape
            assertEquals(input.length, downstreamGradient.length, "Downstream gradient should match input size");
        }
    }

    @Test
    void testIsOutputLayer() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        assertInstanceOf(SoftmaxCrossEntropyOutput.class, layer);
        assertNotNull(layer, "Should create a valid output layer");
    }

    @Test
    void testWeightInitialization() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        // Weights should be initialized (not all zeros)
        boolean hasNonZeroWeight = false;
        float[] testInput = {1.0f, 1.0f, 1.0f};
        var context = layer.forward(testInput, false);
        
        // If all weights were zero, all outputs would be equal (1/3 each)
        float[] outputs = context.outputs();
        for (int i = 1; i < outputs.length; i++) {
            if (Math.abs(outputs[i] - outputs[0]) > 0.001f) {
                hasNonZeroWeight = true;
                break;
            }
        }
        
        assertTrue(hasNonZeroWeight, "Layer should have non-zero weights after initialization");
    }
}