package dev.neuronic.net.outputs;

import dev.neuronic.net.Layers;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import org.junit.jupiter.api.Test;

public class SoftmaxCrossEntropyDebugTest {
    
    @Test
    void debugBackwardWorstPrediction() {
        SgdOptimizer optimizer = new SgdOptimizer(0.01f);
        Layer.Spec spec = Layers.outputSoftmaxCrossEntropy(3, optimizer);
        SoftmaxCrossEntropyOutput layer = (SoftmaxCrossEntropyOutput) spec.create(3);
        
        // Create input that will result in wrong prediction (class 2 instead of 0)
        float[] input = {-10.0f, -10.0f, 10.0f};
        Layer.LayerContext context = layer.forward(input);
        
        // Print forward pass results
        System.out.println("Input: [-10, -10, 10]");
        // context doesn't have intermediates() method, logits are internal to the layer
        System.out.println("Probabilities: " + java.util.Arrays.toString(context.outputs()));
        
        Layer.LayerContext[] stack = {context};
        float[] targets = {1.0f, 0.0f, 0.0f}; // Target is class 0, but logits favor class 2
        
        float[] gradient = layer.backward(stack, 0, targets);
        
        System.out.println("\nTarget: [1, 0, 0]");
        System.out.println("Gradient to previous layer: " + java.util.Arrays.toString(gradient));
        
        // Also check the internal gradient (probabilities - targets)
        float[] internalGrad = new float[3];
        for (int i = 0; i < 3; i++) {
            internalGrad[i] = context.outputs()[i] - targets[i];
        }
        System.out.println("Internal gradient (probs - targets): " + java.util.Arrays.toString(internalGrad));
        
        // Check gradient magnitude
        float maxGrad = 0.0f;
        for (float g : gradient) {
            maxGrad = Math.max(maxGrad, Math.abs(g));
        }
        System.out.println("\nMax downstream gradient magnitude: " + maxGrad);
        System.out.println("Test threshold: 0.0001f");
        System.out.println("Passes test: " + (maxGrad > 0.0001f));
        
        // Print weight matrix shape
        System.out.println("\nWeight matrix is 3x3 (inputs x neurons)");
        System.out.println("Computing: weights^T @ internal_gradient");
        
        // Compute expected downstream gradient manually
        float[] expected = new float[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // weights[i][j] * internalGrad[j]
                System.out.printf("  expected[%d] += weights[%d][%d] * internalGrad[%d]\n", i, i, j, j);
            }
        }
    }
}