package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug the embedding layer forward pass.
 */
public class EmbeddingLayerDebugTest {
    
    @Test
    public void testEmbeddingLayerForward() {
        // Create embedding layer directly
        AdamWOptimizer optimizer = new AdamWOptimizer(0.1f, 0.0f);
        InputSequenceEmbeddingLayer embedLayer = new InputSequenceEmbeddingLayer(
            optimizer, 3, 10, 8, WeightInitStrategy.XAVIER
        );
        
        // Build vocabulary
        embedLayer.getTokenId("a"); // Should be 1
        embedLayer.getTokenId("b"); // Should be 2
        
        // Test forward pass with different inputs
        float[] inputA = {1.0f, 1.0f, 1.0f}; // "a a a"
        float[] inputB = {2.0f, 2.0f, 2.0f}; // "b b b"
        
        Layer.LayerContext contextA = embedLayer.forward(inputA, false);
        Layer.LayerContext contextB = embedLayer.forward(inputB, false);
        
        float[] outputA = contextA.outputs();
        float[] outputB = contextB.outputs();
        
        System.out.println("Output A length: " + outputA.length);
        System.out.println("Output B length: " + outputB.length);
        System.out.println("First 8 values of output A: " + Arrays.toString(Arrays.copyOf(outputA, 8)));
        System.out.println("First 8 values of output B: " + Arrays.toString(Arrays.copyOf(outputB, 8)));
        
        // Check if outputs are different
        boolean allSame = true;
        for (int i = 0; i < outputA.length; i++) {
            if (Math.abs(outputA[i] - outputB[i]) > 1e-6) {
                allSame = false;
                break;
            }
        }
        
        assertFalse(allSame, "Embedding outputs should be different for different inputs");
    }
    
    @Test
    public void testSimpleDenseNetwork() {
        // Test a simple dense network without embeddings
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputSoftmaxCrossEntropy(5));
            
        // Test with different inputs
        float[] inputA = {1.0f, 1.0f, 1.0f};
        float[] inputB = {2.0f, 2.0f, 2.0f};
        
        float[] outputA = net.predict(inputA);
        float[] outputB = net.predict(inputB);
        
        System.out.println("\nSimple dense network test:");
        System.out.println("Output A: " + Arrays.toString(outputA));
        System.out.println("Output B: " + Arrays.toString(outputB));
        
        // Check if outputs are different
        boolean allSame = true;
        for (int i = 0; i < outputA.length; i++) {
            if (Math.abs(outputA[i] - outputB[i]) > 1e-6) {
                allSame = false;
                break;
            }
        }
        
        assertFalse(allSame, "Network outputs should be different for different inputs");
    }
}